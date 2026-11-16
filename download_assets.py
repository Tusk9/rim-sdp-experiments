#!/usr/bin/env python
"""
download_assets.py — fetch model + datasets for RIM + SDP replication (Zhao et al. 2026).

What it does:
  * Resolves every HF repo to an exact commit SHA and downloads AT that SHA (reproducible).
  * Detects gating at runtime (repo_info().gated) — does not trust a hardcoded list.
  * Builds the training subset: first 10,000 AM-DeepSeek examples, byte-exact, hashed.
  * Pulls AdvBench from the original llm-attacks GitHub CSV (has the `target` compliance
    strings needed for the safety direction S, context file §3.1).
  * Runs count / schema sanity checks. Never prints prompt or response text.
  * Writes runs/assets/manifest.json + manifest.csv.

Exit code: 0 if all REQUIRED assets succeeded, 1 otherwise. Gated assets you have not been
granted yet are reported as GATED_NO_ACCESS with the URL to request access.

Usage:
  hf auth login
  python download_assets.py                 # everything
  python download_assets.py --only model am  # subset by name
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

# =============================== CONFIG ======================================
SEED = 3407  # paper seed; nothing here samples, set for hygiene
ROOT = Path(os.environ.get("RIM_DATA_ROOT", "assets"))
LOG_DIR = Path("runs/assets")

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_EXPECT = {"num_hidden_layers": 36, "hidden_size": 2048}  # context file §2
# Scale-up run (run_7b.sh, not Phase 1/2): the paper's second model. Fetched only when named
# (`--only model_7b`), ~15 GB.
MODEL_7B_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_7B_EXPECT = {"num_hidden_layers": 28, "hidden_size": 3584}

AM_REPO = "a-m-team/AM-DeepSeek-R1-Distilled-1.4M"
# [UNSPECIFIED] The paper says "first 10,000 examples" but the repo is split into configs
# (am_0.5M / am_0.9M). Which config, and the shard order, are not stated.
# MY CHOICE: am_0.5M, shards in sorted filename order, lines in file order.
AM_CONFIG = "am_0.5M"
AM_N = 10_000

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)

# Step 1 follow-up (not a paper asset): XSTest over-refusal set (Röttger et al. 2024).
# 250 safe prompts that look unsafe + 200 unsafe contrast prompts. Eval only.
XSTEST_URL = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
XSTEST_SHA256 = "11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a"  # 2026-11-10

# Step 4 baseline (not a paper asset): STAR-1, 1k safety-reasoning examples (Apache-2.0),
# pinned to a commit of the authors' GitHub repo (same file as the HF dataset UCSC-VLAA/STAR-1).
STAR1_URL = ("https://raw.githubusercontent.com/UCSC-VLAA/STAR-1/"
             "b49333598887845f148ed1b4d2576dc2bff7581a/data/STAR-1.json")
STAR1_SHA256 = "cfff9b23307914ecf8a94a1d5526f71f7144fcfc5f651741bea1ac41b3184db4"

# name -> spec. `required` = needed for Phase 1/2 of the replication.
HF_ASSETS = {
    "model": dict(
        repo_id=MODEL_ID, repo_type="model", required=True,
        allow=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json", "LICENSE*"],
        license="Qwen-Research (non-commercial research)",
    ),
    "model_7b": dict(
        repo_id=MODEL_7B_ID, repo_type="model", required=False, named_only=True,
        allow=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json", "LICENSE*"],
        license="Apache-2.0", note="run_7b.sh only.",
    ),
    "hexphi": dict(
        repo_id="LLM-Tuning-Safety/HEx-PHI", repo_type="dataset", required=True,
        allow=None, license="Gated; research use per dataset terms",
        note="Eval only. Public release omits one category -> expect 300 prompts.",
    ),
    "gpqa": dict(
        repo_id="Idavidrein/gpqa", repo_type="dataset", required=True,
        allow=None, license="CC-BY-4.0, gated; authors ask that examples not be posted online",
        note="Subset [UNSPECIFIED]; paper numbers (29.7/28.6/25.9) only fit N=448 -> GPQA main.",
    ),
    "safetybench": dict(
        repo_id="thu-coai/SafetyBench", repo_type="dataset", required=True,
        allow=None, license="MIT (per repo)",
        note="Check log: if test split has no 'answer' field, labels require official submission.",
    ),
    "aime24": dict(repo_id="math-ai/aime24", repo_type="dataset", required=True, allow=None, license="see repo"),
    "aime25": dict(repo_id="math-ai/aime25", repo_type="dataset", required=True, allow=None, license="see repo"),
    "alpaca": dict(
        repo_id="yahma/alpaca-cleaned", repo_type="dataset", required=False,
        allow=None, license="CC-BY-4.0", note="Diagnostics only (CKA, §3.4).",
    ),
}
# =============================================================================

random.seed(SEED)
sys.path  # noqa
from huggingface_hub import HfApi, get_token, hf_hub_download, snapshot_download  # noqa: E402

try:
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
except ImportError:  # older hub
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

API = HfApi()


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def hf_url(spec) -> str:
    prefix = "" if spec["repo_type"] == "model" else "datasets/"
    return f"https://huggingface.co/{prefix}{spec['repo_id']}"


def resolve(spec) -> dict:
    info = API.repo_info(spec["repo_id"], repo_type=spec["repo_type"])
    return {"sha": info.sha, "gated": info.gated or False}  # False | "auto" | "manual"


# ------------------------------ fetchers -------------------------------------
def fetch_hf_snapshot(name: str, spec: dict) -> dict:
    rec = {"name": name, "source": hf_url(spec), "required": spec["required"],
           "license": spec.get("license"), "note": spec.get("note", "")}
    try:
        meta = resolve(spec)
    except RepositoryNotFoundError:
        rec.update(status="NOT_FOUND_OR_GATED", detail="repo_info failed; check ID or access")
        return rec
    rec.update(revision=meta["sha"], gated=meta["gated"])
    local = ROOT / name
    try:
        path = snapshot_download(
            repo_id=spec["repo_id"], repo_type=spec["repo_type"], revision=meta["sha"],
            local_dir=local, allow_patterns=spec["allow"],
        )
    except GatedRepoError:
        rec.update(status="GATED_NO_ACCESS", detail=f"Request access at {hf_url(spec)}, then rerun")
        return rec
    root = Path(path).resolve()
    files = sorted(str(p.relative_to(root)) for p in root.rglob("*")
                   if p.is_file() and ".cache" not in p.relative_to(root).parts)
    rec.update(status="OK", local_dir=str(local), n_files=len(files))
    return rec


def fetch_am_subset() -> dict:
    spec = dict(repo_id=AM_REPO, repo_type="dataset")
    rec = {"name": "am_subset", "source": hf_url(spec), "required": True,
           "license": "CC-BY-NC-4.0 (non-commercial)", "config": AM_CONFIG, "n_target": AM_N,
           "note": "[UNSPECIFIED] config + ordering; MY CHOICE recorded in 'config'."}
    try:
        meta = resolve(spec)
    except RepositoryNotFoundError:
        rec.update(status="NOT_FOUND_OR_GATED"); return rec
    rec.update(revision=meta["sha"], gated=meta["gated"])

    files = API.list_repo_files(AM_REPO, repo_type="dataset", revision=meta["sha"])
    shards = sorted(f for f in files if AM_CONFIG in f and (".jsonl" in f or f.endswith(".json")))
    if not shards:
        rec.update(status="NO_SHARDS", detail=f"No files matching {AM_CONFIG}. Repo files: {files[:20]}")
        return rec

    out = ROOT / "am_subset" / f"am_first{AM_N}_{AM_CONFIG}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0; used = []
    with open(out, "wb") as w:
        for shard in shards:
            try:
                p = hf_hub_download(AM_REPO, shard, repo_type="dataset", revision=meta["sha"])
            except GatedRepoError:
                rec.update(status="GATED_NO_ACCESS", detail=f"Request access at {hf_url(spec)}")
                return rec
            used.append(shard)
            if shard.endswith(".zst"):
                import zstandard
                fh = io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(open(p, "rb")))
            elif shard.endswith(".gz"):
                fh = gzip.open(p, "rb")
            else:
                fh = open(p, "rb")
            with fh:
                for line in fh:
                    if not line.strip():
                        continue
                    w.write(line if line.endswith(b"\n") else line + b"\n")
                    n += 1
                    if n >= AM_N:
                        break
            if n >= AM_N:
                break

    # Format checks (context file §2: all examples contain <think>...</think> then <answer>)
    stats = {"rows": 0, "has_think_open": 0, "has_think_close": 0, "has_answer": 0,
             "parse_errors": 0, "asst_chars_total": 0}
    with open(out, "rb") as f:
        for line in f:
            stats["rows"] += 1
            try:
                obj = json.loads(line)
                msgs = obj.get("messages") or obj.get("conversations") or []
                asst = next((m.get("content", "") for m in msgs
                             if m.get("role") in ("assistant", "gpt")), "")
            except Exception:
                stats["parse_errors"] += 1
                continue
            stats["has_think_open"] += "<think>" in asst
            stats["has_think_close"] += "</think>" in asst
            stats["has_answer"] += "<answer>" in asst
            stats["asst_chars_total"] += len(asst)
    stats["asst_chars_mean"] = round(stats.pop("asst_chars_total") / max(stats["rows"], 1))

    rec.update(status="OK" if n == AM_N else "SHORT", local_file=str(out), rows=n,
               shards_used=used, sha256=sha256_file(out), checks=stats)
    return rec


def fetch_advbench() -> dict:
    rec = {"name": "advbench", "source": ADVBENCH_URL, "required": True, "license": "MIT",
           "note": "Direction extraction only (§3.1). Columns: goal, target."}
    out = ROOT / "advbench" / "harmful_behaviors.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(ADVBENCH_URL, timeout=60) as r:
            out.write_bytes(r.read())
    except Exception as e:
        rec.update(status="DOWNLOAD_FAILED", detail=repr(e)); return rec
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = list(rows[0].keys()) if rows else []
    ok = len(rows) == 520 and {"goal", "target"} <= set(cols)
    rec.update(status="OK" if ok else "CHECK_FAILED", local_file=str(out), rows=len(rows),
               columns=cols, sha256=sha256_file(out), expected_rows=520)
    return rec


def fetch_xstest() -> dict:
    rec = {"name": "xstest", "source": XSTEST_URL, "required": False,
           "license": "see XSTest repo", "note": "Step 1 over-refusal eval only."}
    out = ROOT / "xstest" / "xstest_prompts.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(XSTEST_URL, timeout=60) as r:
            out.write_bytes(r.read())
    except Exception as e:
        rec.update(status="DOWNLOAD_FAILED", detail=repr(e)); return rec
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = list(rows[0].keys()) if rows else []
    labels = {}
    for r in rows:
        labels[r.get("label")] = labels.get(r.get("label"), 0) + 1
    digest = sha256_file(out)
    ok = len(rows) == 450 and {"prompt", "label", "type"} <= set(cols)
    rec.update(status="OK" if ok else "CHECK_FAILED", local_file=str(out), rows=len(rows),
               columns=cols, labels=labels, sha256=digest,
               sha256_matches_pinned=digest == XSTEST_SHA256)
    return rec


def fetch_star1() -> dict:
    rec = {"name": "star1", "source": STAR1_URL, "required": False, "license": "Apache-2.0",
           "note": "Step 4 safety-mixing baseline only."}
    out = ROOT / "star1" / "STAR-1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(STAR1_URL, timeout=120) as r:
            out.write_bytes(r.read())
    except Exception as e:
        rec.update(status="DOWNLOAD_FAILED", detail=repr(e)); return rec
    digest = sha256_file(out)
    rows = json.loads(out.read_text(encoding="utf-8"))
    ok = (digest == STAR1_SHA256 and len(rows) == 1000
          and {"question", "response"} <= set(rows[0]))
    rec.update(status="OK" if ok else "CHECK_FAILED", local_file=str(out), rows=len(rows),
               sha256=digest, sha256_matches_pinned=digest == STAR1_SHA256)
    return rec


# ------------------------------ checks ---------------------------------------
def check(name: str, rec: dict) -> None:
    if rec.get("status") != "OK":
        return
    d = Path(rec["local_dir"])
    c = {}
    try:
        if name in ("model", "model_7b"):
            cfg = json.loads((d / "config.json").read_text())
            for k, v in (MODEL_EXPECT if name == "model" else MODEL_7B_EXPECT).items():
                c[k] = cfg.get(k)
                if cfg.get(k) != v:
                    rec["status"] = "CHECK_FAILED"
            tok = json.loads((d / "tokenizer_config.json").read_text())
            c["has_chat_template"] = "chat_template" in tok
            c["safetensors_files"] = len(list(d.glob("*.safetensors")))
        elif name == "hexphi":
            per = {}
            for p in sorted(d.rglob("category_*.csv")):
                with open(p, newline="") as f:
                    per[p.stem] = sum(1 for r in csv.reader(f) if r and r[0].strip())
            c["per_category"] = per
            c["total"] = sum(per.values())
            c["expected_total"] = 300
        elif name == "gpqa":
            import pandas as pd
            for p in sorted(d.rglob("*.csv")):
                c[p.name] = len(pd.read_csv(p))
            c["expected_main"] = 448
        elif name == "safetybench":
            for p in sorted(d.rglob("*.json")):
                obj = json.loads(p.read_text())
                items = obj if isinstance(obj, list) else [x for v in obj.values() for x in (v if isinstance(v, list) else [])]
                if items and isinstance(items[0], dict):
                    c[p.name] = {"n": len(items), "has_answer_field": "answer" in items[0]}
        elif name in ("aime24", "aime25"):
            from datasets import load_dataset
            ds = load_dataset(str(d))
            c["splits"] = {k: len(v) for k, v in ds.items()}
            c["expected_per_year"] = 30
        elif name == "alpaca":
            for p in d.rglob("*.json"):
                c[p.name] = len(json.loads(p.read_text()))
    except Exception as e:
        c["check_error"] = repr(e)
    rec["checks"] = c


# ------------------------------- main ----------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="subset: model hexphi gpqa safetybench aime24 aime25 "
                    "alpaca am advbench xstest star1 model_7b (xstest/star1/model_7b only when named)")
    args = ap.parse_args()
    want = set(args.only) if args.only else None

    ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"HF token present: {get_token() is not None}  (needed for gated repos)")

    records = []
    t0 = time.time()
    for name, spec in HF_ASSETS.items():
        if want and name not in want:
            continue
        if spec.get("named_only") and not (want and name in want):
            continue                       # e.g. model_7b: only with --only model_7b
        print(f"-> {name}: {spec['repo_id']}")
        rec = fetch_hf_snapshot(name, spec)
        check(name, rec)
        records.append(rec)
        print(f"   {rec['status']}  gated={rec.get('gated')}  rev={str(rec.get('revision'))[:10]}")
    if not want or "am" in want:
        print(f"-> am_subset: {AM_REPO} [{AM_CONFIG}] first {AM_N}")
        rec = fetch_am_subset(); records.append(rec)
        print(f"   {rec['status']}  gated={rec.get('gated')}  rows={rec.get('rows')}")
    if not want or "advbench" in want:
        print("-> advbench (GitHub)")
        rec = fetch_advbench(); records.append(rec)
        print(f"   {rec['status']}  rows={rec.get('rows')}")
    if want and "star1" in want:           # Step 4 only: fetched on request, not by default
        print("-> star1 (GitHub, pinned commit)")
        rec = fetch_star1(); records.append(rec)
        print(f"   {rec['status']}  rows={rec.get('rows')}  pinned_sha_match={rec.get('sha256_matches_pinned')}")
    if want and "xstest" in want:          # Step 1 only: fetched on request, not by default
        print("-> xstest (GitHub)")
        rec = fetch_xstest(); records.append(rec)
        print(f"   {rec['status']}  rows={rec.get('rows')}  labels={rec.get('labels')}  "
              f"pinned_sha_match={rec.get('sha256_matches_pinned')}")

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED, "elapsed_s": round(time.time() - t0, 1),
        "choices_not_in_paper": {
            "am_config": AM_CONFIG, "am_ordering": "sorted shard filenames, file line order",
            "gpqa_subset": "main (derived: only N=448 fits paper numbers)",
        },
        "assets": records,
    }
    (LOG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    with open(LOG_DIR / "manifest.csv", "w", newline="") as f:
        cols = ["name", "status", "required", "gated", "revision", "rows", "sha256", "source", "license"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)

    print("\n=== SUMMARY ===")
    gated = [r for r in records if r.get("gated")]
    blocked = [r for r in records if r["status"] == "GATED_NO_ACCESS"]
    failed = [r for r in records if r["required"] and r["status"] != "OK"]
    print("Gated repos:", ", ".join(f"{r['name']}({r['gated']})" for r in gated) or "none")
    for r in blocked:
        print(f"  NEED ACCESS: {r['name']} -> {r['source']}")
    for r in records:
        if "checks" in r:
            print(f"  {r['name']}: {json.dumps(r['checks'], default=str)[:300]}")
    print(f"Manifest: {LOG_DIR/'manifest.json'}")
    if failed:
        print("REQUIRED assets not OK:", [r["name"] for r in failed])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
