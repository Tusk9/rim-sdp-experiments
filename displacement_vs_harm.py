#!/usr/bin/env python
"""
displacement_vs_harm.py — Step 3, item 13 (paper §3.5, Appendix C.6): does a prompt's
displacement along the safety direction predict how much more harmful its answer became?

  python displacement_vs_harm.py runs/diagnose/displacement_per_prompt_refusal_end.csv \\
      --base runs/eval/base --run amsft_step157=runs/eval/amsft_step157 \\
      --run sdp_step157=runs/eval/sdp_step157

For each tag in the per-prompt csv (written by displacement.py):
  x = mean displacement over the summary layers (mean_d_M; positive = toward refusal)
  y = judge score after fine-tuning - judge score of the base model, same prompt
  Spearman rho over prompts whose score changed (the paper's subset: N = 99 for 3B AM-SFT,
  rho = 0.475) and over all prompts. p from the t approximation and a 10k permutation test.
Our sign convention predicts rho < 0 (moving away from refusal goes with more harm). The paper
reports rho > 0 for AM-SFT without stating its sign convention for the score change, so compare
magnitudes and significance, not signs, until that is resolved.
Scores come from each run's hexphi_items.jsonl (sampled generations, judge score 1-5).
CPU only. Writes <csv stem>_vs_harm.json next to the csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from stats_utils import spearman


def scores(run_dir: str) -> dict:
    p = Path(run_dir) / "hexphi_items.jsonl"
    if not p.exists():
        raise SystemExit(f"{p} missing")
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("score") is not None:
                out[r["id"]] = r["score"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("per_prompt_csv")
    ap.add_argument("--base", required=True, help="base model eval run dir")
    ap.add_argument("--run", action="append", required=True, help="tag=eval run dir")
    ap.add_argument("--n-perm", type=int, default=10000)
    a = ap.parse_args()

    base = scores(a.base)
    runs = dict(s.split("=", 1) for s in a.run)
    disp = {}
    with open(a.per_prompt_csv, newline="") as f:
        for r in csv.DictReader(f):
            disp.setdefault(r["tag"], {})[r["id"]] = float(r["mean_d_M"])
    out = {"csv": a.per_prompt_csv, "base": a.base, "results": {},
           "sign_note": "x positive = toward refusal; y = score_ft - score_base"}
    for tag, rd in runs.items():
        if tag not in disp:
            print(f"skip {tag}: not in {a.per_prompt_csv}"); continue
        ft = scores(rd)
        ids = [i for i in disp[tag] if i in base and i in ft]
        x = [disp[tag][i] for i in ids]
        y = [ft[i] - base[i] for i in ids]
        ch = [k for k, v in enumerate(y) if v != 0]
        res = {"n_matched": len(ids), "n_changed": len(ch),
               "n_up": sum(v > 0 for v in y), "n_down": sum(v < 0 for v in y),
               "changed": spearman([x[k] for k in ch], [y[k] for k in ch], n_perm=a.n_perm),
               "all": spearman(x, y, n_perm=a.n_perm)}
        out["results"][tag] = res
        c = res["changed"]
        print(f"{tag:20s} changed N={c['n']:>3d} rho={c['rho']} p_t={c['p_t'] and round(c['p_t'], 4)} "
              f"p_perm={c['p_perm'] and round(c['p_perm'], 4)} | all N={res['all']['n']} "
              f"rho={res['all']['rho']}")
    print("paper (3B): AM-SFT rho = 0.475 (N = 99), SDP rho = -0.164 (N = 93, n.s.)")
    p = Path(a.per_prompt_csv)
    (p.parent / f"{p.stem}_vs_harm.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
