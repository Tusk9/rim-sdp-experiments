#!/usr/bin/env python
"""
judge_disagreement.py — Step 1 follow-up (CPU only, no API calls).

Under forced thinking, HEx-PHI harm judged on the FINAL ANSWER ONLY (text after </think>) is far
higher than harm judged on the FULL response (base 30.7 vs 2.0, SDP 22.7 vs 4.7). This script
lists the prompts where the two judgments disagree, with structural facts and the judge's own
one-line reasons (recovered from runs/eval/judge_cache.jsonl, no new judge calls), so a human
can decide which metric to trust.

For each tag it writes runs/inspect/disagree_<tag>.csv:
  id, category, full_score, final_score, kind, think_closed, finish, n_tokens,
  think_chars, final_chars, judge_trunc, full_reason, final_reason[, final_head]
and prints a summary: how often final=5 & full<5 (and the reverse), split by think_closed.

Response text is NOT written unless --show-chars N is given (then the first N characters of the
final answer go into final_head). Those files contain model outputs on harmful prompts:
keep them local, never commit or post them. Default N = 0.

Usage:  python judge_disagreement.py base_forcethink amsft_step157_forcethink sdp_step157_forcethink
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import eval_harness as E


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--hexphi-dir", help="override assets/hexphi")
    ap.add_argument("--cache", default="runs/eval/judge_cache.jsonl")
    ap.add_argument("--show-chars", type=int, default=0,
                    help="include the first N chars of the final answer (default 0 = none)")
    ap.add_argument("--out-dir", default="runs/inspect")
    a = ap.parse_args()

    cfg = E.Config()
    cfg.judge_backend = "mock"                    # only to build the Judge (template, keys)
    if a.hexphi_dir:
        cfg.hexphi_dir = a.hexphi_dir
    first = json.load(open(Path("runs/eval") / a.tags[0] / "results.json")).get("config", {})
    for k in ("judge_model", "judge_prompt_file"):          # use the run's own judge settings
        if first.get(k) is not None:
            setattr(cfg, k, first[k])
    judge = E.Judge(cfg, Path(a.cache))           # same template => same cache keys
    cache = judge.cache
    print(f"judge cache: {len(cache)} entries; judge model {cfg.judge_model}; "
          f"template sha {judge.template_sha}")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for tag in a.tags:
        d = Path("runs/eval") / tag
        res = json.load(open(d / "results.json"))
        rc = res.get("config", {})
        suffix = rc.get("hexphi_suffix") or ""
        cfg.hexphi_suffix = suffix
        items = {it["id"]: it for it in E.load_hexphi(cfg)}
        gens = {g["id"]: g for g in map(json.loads, open(d / "generations.jsonl"))
                if g["bench"] == "hexphi" and g.get("run", 0) == 0}
        rows = [json.loads(l) for l in open(d / "hexphi_items.jsonl") if l.strip()]
        kinds, out_rows, missing = Counter(), [], 0
        for r in rows:
            it, g = items.get(r["id"]), gens.get(r["id"])
            if it is None or g is None:
                missing += 1; continue
            instr = it.get("instr", it["user"])
            fin, unclosed = E.final_segment(g["text"])
            jf = cache.get(judge._key(instr, g["text"]), {})
            jl = cache.get(judge._key(instr, fin), {})
            fs, ls = r.get("score"), r.get("final_score")
            if fs is None or ls is None:
                kind = "missing_score"
            elif ls >= 5 > fs:
                kind = "final5_full<5"
            elif fs >= 5 > ls:
                kind = "full5_final<5"
            else:
                kind = "agree"
            kinds[(kind, "closed" if r.get("think_closed") else
                   ("unclosed" if r.get("think_used") else "no_think"))] += 1
            if kind == "agree":
                continue
            think_chars = len(g["text"]) - len(fin) if not unclosed else len(g["text"])
            row = {"id": r["id"], "category": r.get("category"), "full_score": fs,
                   "final_score": ls, "kind": kind, "think_closed": r.get("think_closed"),
                   "finish": r.get("finish"), "n_tokens": r.get("n_tokens"),
                   "think_chars": think_chars, "final_chars": len(fin),
                   "judge_trunc": r.get("judge_trunc"),
                   "full_reason": jf.get("reason", "(not in cache)"),
                   "final_reason": jl.get("reason", "(not in cache)")}
            if a.show_chars > 0:
                row["final_head"] = fin[: a.show_chars].replace("\n", " ")
            out_rows.append(row)
        p = out / f"disagree_{tag}.csv"
        if out_rows:
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
        n = len(rows)
        print(f"\n== {tag}  (n={n}{', suffix=' + repr(suffix.strip()) if suffix else ''}"
              f"{f', {missing} unmatched' if missing else ''})")
        for (k, t), c in sorted(kinds.items()):
            print(f"  {k:16s} {t:9s} {c:4d}  ({100 * c / max(n, 1):.1f}%)")
        if out_rows:
            fl = [x for x in out_rows if x["kind"] == "final5_full<5"]
            if fl:
                import statistics as st
                print(f"  final5_full<5: median final_chars {st.median(x['final_chars'] for x in fl):.0f}, "
                      f"median think_chars {st.median(x['think_chars'] for x in fl):.0f}, "
                      f"full scores {dict(Counter(x['full_score'] for x in fl))}")
            print(f"  -> {p}")
    print("\nRead the reasons (and, locally only, --show-chars 300) for ~20 rows per tag. "
          "Do not share response text.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
