#!/usr/bin/env python
"""
diagnose_loops.py — CPU-only diagnosis of the AM-SFT repetition loops (Phase 1b).

Questions answered:
  1. Does the AM training data itself contain loops / repetitive traces?
  2. In the AM-SFT generations, WHERE do loops start (token position) relative to the length
     of the training traces? Late onset -> length generalisation; early onset -> over-fitting.
  3. Do loops start inside <think> (reasoning never closes) or after it?

Inputs (all local; nothing is uploaded):
  assets/am_subset/am_first10000_am_0.5M.jsonl         training data (from download_assets.py)
  runs/data_prep/per_example_L16384_n10000_truncate.csv  think-token lengths (from prep_am_data.py)
  runs/eval/{amsft,amsft_r1,base_r1}/generations.jsonl   from the private HF backup
Output: runs/diagnose/loops.json + console summary. NO response text is printed (HEx-PHI
outputs are only counted, never shown).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

# =============================== CONFIG ======================================
AM_FILE = "assets/am_subset/am_first10000_am_0.5M.jsonl"
PREP_CSV = "runs/data_prep/per_example_L16384_n10000_truncate.csv"
TOKENIZER = "assets/model"
GEN = {"amsft": "runs/eval/amsft/generations.jsonl",
       "amsft_r1": "runs/eval/amsft_r1/generations.jsonl",
       "base_r1": "runs/eval/base_r1/generations.jsonl"}
BENCHES = ("gpqa", "aime24", "aime25", "hexphi")
MIN_REPS = 3          # a "loop" = the same block repeated >= 3 times back to back at the end
PROBE = 100           # chars used to find the repeating period
OUT = "runs/diagnose/loops.json"
# =============================================================================


def loop_onset(t: str):
    """Start of the exact periodic tail. Returns (onset_char, period_chars, reps) or None."""
    if len(t) < 3 * PROBE:
        return None
    key = t[-PROBE:]
    j = t.rfind(key, 0, len(t) - PROBE)
    if j < 0:
        return None
    period = len(t) - PROBE - j
    if period <= 0:
        return None
    i = len(t) - period - 1
    while i >= 0 and t[i] == t[i + period]:
        i -= 1
    onset = i + 1
    reps = (len(t) - onset) / period
    return (onset, period, reps) if reps >= MIN_REPS else None


_SENT = re.compile(r"(?<=[.!?\n])\s+")


def max_repeat_run(t: str, min_len: int = 10) -> int:
    """Longest run of the same sentence repeated back to back, anywhere in the text."""
    best, run, prev = 1, 1, None
    for s in _SENT.split(t):
        s = s.strip()
        if len(s) >= min_len and s == prev:
            run += 1
            best = max(best, run)
        else:
            run = 1
        prev = s
    return best


def distinct4(t: str) -> float:
    w = t.split()
    grams = list(zip(w, w[1:], w[2:], w[3:]))
    return len(set(grams)) / len(grams) if grams else 1.0


def pct(a, qs=(10, 50, 90, 95, 99)):
    a = np.asarray([x for x in a if x is not None], dtype=float)
    if a.size == 0:
        return None
    return {f"p{q}": int(np.percentile(a, q)) for q in qs} | {"n": int(a.size)}


def main() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    report = {}

    # ---- 1. training data -----------------------------------------------------
    think_lens = []
    if Path(PREP_CSV).exists():
        with open(PREP_CSV) as f:
            think_lens = [int(r["think_tokens"]) for r in csv.DictReader(f) if r["think_tokens"]]
    loops, mid_loops, d4, n = 0, 0, [], 0
    for line in open(AM_FILE, encoding="utf-8"):
        if not line.strip():
            continue
        o = json.loads(line)
        msgs = o.get("messages") or o.get("conversations") or []
        a = next((m.get("content", m.get("value", "")) for m in msgs
                  if m.get("role", m.get("from")) in ("assistant", "gpt")), "")
        n += 1
        if loop_onset(a):
            loops += 1
        if max_repeat_run(a) >= MIN_REPS:
            mid_loops += 1
        if n <= 2000:                       # distinct-4gram on a 2k sample (speed)
            d4.append(distinct4(a))
    think_p = pct(think_lens)
    report["training_data"] = {"n": n, "pct_with_tail_loop": round(100 * loops / max(n, 1), 2),
                               "pct_with_repeat_run_anywhere": round(100 * mid_loops / max(n, 1), 2),
                               "think_tokens": think_p,
                               "distinct4_mean": round(float(np.mean(d4)), 3) if d4 else None}
    train_p95 = think_p["p95"] if think_p else None
    train_max = int(max(think_lens)) if think_lens else None

    # ---- 2. generations ------------------------------------------------------------
    report["generations"] = {}
    for tag, path in GEN.items():
        if not Path(path).exists():
            print(f"(skip {tag}: {path} not found)")
            continue
        rows = [json.loads(l) for l in open(path) if l.strip()]
        for b in BENCHES:
            R = [r for r in rows if r["bench"] == b]
            if not R:
                continue
            onsets, inside_think, beyond_p95, beyond_max, fin_len, d4f = [], 0, 0, 0, [], []
            n_loop = 0
            for r in R:
                t = r["text"]
                lo = loop_onset(t)
                if lo:
                    n_loop += 1
                    on_tok = ntok(t[:lo[0]])
                    onsets.append(on_tok)
                    close = t.find("</think>")
                    if "<think>" in t and (close < 0 or close > lo[0]):
                        inside_think += 1
                    beyond_p95 += bool(train_p95 and on_tok > train_p95)
                    beyond_max += bool(train_max and on_tok > train_max)
                elif r["finish"] != "length":
                    fin_len.append(r["n_tokens"])
                    if b != "hexphi":
                        d4f.append(distinct4(t))
            nl = max(n_loop, 1)
            report["generations"][f"{tag}/{b}"] = {
                "n": len(R),
                "truncated_pct": round(100 * sum(r["finish"] == "length" for r in R) / len(R), 1),
                "loop_pct": round(100 * n_loop / len(R), 1),
                "loop_onset_tokens": pct(onsets),
                "loop_inside_think_pct_of_loops": round(100 * inside_think / nl, 1) if n_loop else None,
                "onset_beyond_train_p95_pct_of_loops": round(100 * beyond_p95 / nl, 1) if n_loop else None,
                "onset_beyond_train_max_pct_of_loops": round(100 * beyond_max / nl, 1) if n_loop else None,
                "finished_len_tokens": pct(fin_len),
                "finished_distinct4_mean": round(float(np.mean(d4f)), 3) if d4f else None,
            }

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(report, indent=2))

    # ---- console summary -------------------------------------------------------------
    td = report["training_data"]
    print("=== training data (AM, 10k) ===")
    print(f"tail loops: {td['pct_with_tail_loop']}%  | sentence repeated >={MIN_REPS}x anywhere: "
          f"{td['pct_with_repeat_run_anywhere']}%  | think tokens {td['think_tokens']}  "
          f"| distinct-4gram {td['distinct4_mean']}")
    print("\n=== generations ===")
    print(f"{'tag/bench':18s} {'n':>5s} {'trunc%':>7s} {'loop%':>6s} {'onset p10/p50/p90':>20s} "
          f"{'inThink%':>9s} {'>trainP95%':>11s} {'finished p50':>13s}")
    for k, v in report["generations"].items():
        o = v["loop_onset_tokens"] or {}
        on = f"{o.get('p10','-')}/{o.get('p50','-')}/{o.get('p90','-')}"
        fl = (v["finished_len_tokens"] or {}).get("p50", "-")
        print(f"{k:18s} {v['n']:>5d} {v['truncated_pct']:>7} {v['loop_pct']:>6} {on:>20s} "
              f"{str(v['loop_inside_think_pct_of_loops']):>9s} "
              f"{str(v['onset_beyond_train_p95_pct_of_loops']):>11s} {str(fl):>13s}")
    print(f"\ntraining think-token p95 = {train_p95}, max = {train_max}")
    print(f"saved: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
