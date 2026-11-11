#!/usr/bin/env python
"""
inspect_generations.py — Step 1, item 1: does a model reason WITHOUT <think> tags?

eval_harness counts a response as "thinking" only if the literal text "<think>" appears. A model
that reasons in plain prose ("Okay, let me think...") would score 0% think adoption. This script
checks saved generations for untagged reasoning markers. CPU only, no model needed.

  Summary (console):   per tag x benchmark: think-tag rate, untagged-reasoning rate among
                       responses WITHOUT the tag, mean length, truncation
  Sample (--sample N): writes N responses per tag x benchmark to a LOCAL csv for people to read
                       and annotate (columns reasoning_present / notes left blank)

Usage
  python inspect_generations.py runs/eval/sdp_step157 runs/eval/amsft_step157 runs/eval/base
  python inspect_generations.py runs/eval/sdp_step157 --bench hexphi gpqa --sample 30 \\
      --out runs/inspect/sdp_sample.csv

Safety: the console never shows response text. The --sample csv CONTAINS MODEL OUTPUTS ON
HARMFUL PROMPTS: keep it local / in the private HF repo, never commit or post it.
The marker list is a heuristic (MY CHOICE); the annotated sample is the real check.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

THINK_OPEN = "<think>"
SEED = 3407

# Phrases typical of R1-style reasoning written as prose. A response is flagged as untagged
# reasoning when >= MIN_MARKERS distinct markers occur in it.
MARKERS = {
    "let_me_think": r"\blet me (?:think|reason|work (?:through|out)|figure|break (?:this|it) down)",
    "lets_think": r"\blet'?s (?:think|reason|work through|break (?:this|it) down)",
    "okay_so": r"(?:^|\n)\s*(?:okay|ok|alright|so),? (?:so |let me|the user|i need)",
    "wait": r"(?:^|[.\n]\s*)wait[,.!]",
    "hmm": r"\bhmm+\b",
    "the_user_wants": r"\bthe user (?:is asking|wants|asked|is requesting)",
    "i_need_to": r"\bi (?:need|have) to (?:figure|consider|think|make sure|check)",
    "first_then": r"\bfirst,? (?:i|let|we)\b.{0,200}\b(?:then|next|after that)\b",
    "step_n": r"\bstep \d\b",
    "alternatively": r"\balternatively,",
    "double_check": r"\b(?:let me|i should|i'll) (?:double[- ]check|verify|re-?read|check again)",
    "but_wait": r"\bbut wait\b",
}
MIN_MARKERS = 2
_RX = {k: re.compile(v, re.I | re.S) for k, v in MARKERS.items()}


def markers(text: str) -> list[str]:
    return [k for k, rx in _RX.items() if rx.search(text)]


def load_scores(run_dir: Path) -> dict:
    """HEx-PHI judge scores per id, if the eval wrote them."""
    p = run_dir / "hexphi_items.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r.get("score")
    return out


def pct(k: int, n: int):
    return round(100 * k / n, 1) if n else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="eval run dirs (runs/eval/<tag>)")
    ap.add_argument("--bench", nargs="*", help="benchmarks to include (default: all found)")
    ap.add_argument("--sample", type=int, default=0, help="responses per tag x benchmark to export")
    ap.add_argument("--out", default="runs/inspect/sample.csv")
    ap.add_argument("--summary-out", default="runs/inspect/untagged_reasoning.json")
    a = ap.parse_args()

    summary, sample_rows = {}, []
    for rd in map(Path, a.runs):
        gp = rd / "generations.jsonl"
        if not gp.exists():
            print(f"skip {rd}: no generations.jsonl"); continue
        tag = rd.name
        scores = load_scores(rd)
        rows = [json.loads(l) for l in gp.read_text().splitlines() if l.strip()]
        by = {}
        for r in rows:
            if a.bench and r["bench"] not in a.bench:
                continue
            by.setdefault(r["bench"], []).append(r)
        for bench, rs in sorted(by.items()):
            enriched = []
            for r in rs:
                t = r["text"]
                tagged = THINK_OPEN in t
                mk = markers(t)
                enriched.append({**r, "think_tag": tagged, "markers": mk,
                                 "untagged_reasoning": (not tagged) and len(mk) >= MIN_MARKERS})
            n = len(enriched)
            nt = [e for e in enriched if not e["think_tag"]]
            mcount = {k: sum(k in e["markers"] for e in nt) for k in MARKERS}
            summary.setdefault(tag, {})[bench] = {
                "n": n,
                "think_tag_pct": pct(n - len(nt), n),
                "n_without_tag": len(nt),
                "untagged_reasoning_pct_of_untagged": pct(sum(e["untagged_reasoning"] for e in nt), len(nt)),
                "untagged_reasoning_pct_of_all": pct(sum(e["untagged_reasoning"] for e in nt), n),
                "mean_tokens_without_tag": round(sum(e["n_tokens"] for e in nt) / len(nt), 1) if nt else None,
                "truncated_pct": pct(sum(e.get("finish") == "length" for e in enriched), n),
                "marker_counts_among_untagged": {k: v for k, v in mcount.items() if v},
            }
            if a.sample:
                pick = random.Random(SEED).sample(enriched, min(a.sample, n))
                for e in pick:
                    sample_rows.append({
                        "tag": tag, "bench": e["bench"], "id": e["id"], "run": e.get("run"),
                        "think_tag": e["think_tag"], "untagged_reasoning_flag": e["untagged_reasoning"],
                        "markers": " ".join(e["markers"]), "n_tokens": e["n_tokens"],
                        "finish": e.get("finish"), "judge_score": scores.get(e["id"]) if bench == "hexphi" else None,
                        "reasoning_present (y/n)": "", "notes": "", "response": e["text"]})

    if not summary:
        print("nothing found"); return 1
    sp = Path(a.summary_out); sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps({"markers": MARKERS, "min_markers": MIN_MARKERS, "summary": summary},
                             indent=2))
    print(f"{'tag':24s} {'bench':10s} {'n':>5s} {'<think>%':>9s} {'untagged':>9s} "
          f"{'(of all)':>9s} {'len no-tag':>10s} {'trunc%':>7s}")
    for tag, bs in summary.items():
        for bench, s in bs.items():
            print(f"{tag:24s} {bench:10s} {s['n']:>5d} {s['think_tag_pct']!s:>9s} "
                  f"{s['untagged_reasoning_pct_of_untagged']!s:>9s} {s['untagged_reasoning_pct_of_all']!s:>9s} "
                  f"{s['mean_tokens_without_tag']!s:>10s} {s['truncated_pct']!s:>7s}")
    print(f"'untagged' = % of responses without <think> that contain >= {MIN_MARKERS} reasoning markers")
    print(f"summary: {sp}")
    if sample_rows:
        op = Path(a.out); op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sample_rows[0])); w.writeheader(); w.writerows(sample_rows)
        print(f"sample for manual reading: {op} ({len(sample_rows)} rows) — CONTAINS MODEL OUTPUTS "
              "ON HARMFUL PROMPTS, keep private")
    return 0


if __name__ == "__main__":
    sys.exit(main())
