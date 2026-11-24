#!/usr/bin/env python
"""
mech_cue.py — mechanism check for the cue results (NOT in the paper).

Question: when a cue is appended to a harmful request (the AIME instruction, or an everyday
"think about it" phrase), does the model's state move off the refusal side of the safety
direction S that SDP penalises? And are the prompts on which SDP *chooses* to think under the
AIME cue the ones that sit lowest on S?

For every model (base + adapters) and prompt set (HEx-PHI, HEx-PHI + each cue, benign non-math):
  proj_l(x) = < h_l(x), S_l >   at the generation-prompt token (last prompt token),
averaged over the SDP layers (default 19-32). Positive = toward refusing (S = refuse - comply).
Reported:
  mean projection per set;  paired shift (cue - plain) on the same HEx-PHI prompts;
  per-prompt P(first token = <think>) under each cue (same statistic as gate_probe.py) and the
  mean projection of prompts with P > 0.5 vs the rest (the "chose to think" split).
Outputs runs/mech/mech_summary.json + mech_per_prompt.csv (ids and numbers only, no text).

  python mech_cue.py --adapter amsft_eb8_step1250=runs/train/.../adapter_step1250 \
                     --adapter sdp_eb8_step1250=runs/train/.../adapter_step1250 \
                     --s-file runs/direction/S_advbench.pt [--limit 20]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

import gate_probe as G
from eval_harness import Config

OUT_DIR = "runs/mech"
CUES = ("aime", "reason", "careful", "detailed", "expert")
SETS = ("hexphi",) + tuple(f"hexphi_{c}" for c in CUES) + ("benign_nonmath",)
# MY CHOICE: summary over the SDP layers 19-32 (where the penalty acts); per-layer values saved.
SUMMARY_LAYERS = (19, 32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", action="append", default=[], help="tag=path (repeatable)")
    ap.add_argument("--model-dir", default=G.MODEL_DIR)
    ap.add_argument("--s-file", required=True)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--summary-layers", default=f"{SUMMARY_LAYERS[0]}-{SUMMARY_LAYERS[1]}")
    ap.add_argument("--hexphi-dir"); ap.add_argument("--alpaca-dir"); ap.add_argument("--xstest-file")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.summary_layers.split("-"))

    cfg = Config()
    if a.hexphi_dir: cfg.hexphi_dir = a.hexphi_dir
    if a.alpaca_dir: cfg.alpaca_dir = a.alpaca_dir
    if a.xstest_file: cfg.xstest_file = a.xstest_file
    cfg.gpqa_file = "/nonexistent"           # not needed here
    cfg.aime_dirs = {**cfg.aime_dirs, "aime24": "/nonexistent"}
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    think_ids = tok(G.THINK, add_special_tokens=False).input_ids
    allsets = G.prompt_sets(cfg, a.limit, CUES)
    sets = {k: allsets[k] for k in SETS if k in allsets}
    print("prompt sets: " + ", ".join(f"{k}={len(v)}" for k, v in sets.items()))
    S = torch.load(a.s_file, map_location="cpu")["S"].float()          # [L+1, d], unit rows
    lay = slice(lo, hi + 1)

    base = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=dtype)
    runs = [("base", None)] + [tuple(s.split("=", 1)) for s in a.adapter]
    summary, per_prompt = {}, []
    for tag, path in runs:
        t0 = time.time()
        if path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, path).to(dev).eval()
        else:
            model = base.to(dev).eval()
        proj, pth = {}, {}
        for name, items in sets.items():
            texts = [G.render(tok, x["user"]) for x in items]
            p1, _ls, hs = G.probe(model, tok, texts, think_ids, dev, want_states=True)
            pl = (hs[:, :S.size(0)] * S[:hs.size(1)]).sum(-1)              # [N, L+1]
            proj[name] = pl.numpy()
            pth[name] = p1
            for x, row, p in zip(items, pl[:, lay].mean(1).tolist(), p1):
                per_prompt.append({"tag": tag, "set": name, "id": x["id"],
                                   "proj_summary": round(row, 4), "p_think": round(float(p), 6)})
        s = {"mean_proj": {k: round(float(v[:, lay].mean()), 4) for k, v in proj.items()},
             "mean_proj_per_layer": {k: [round(float(c), 4) for c in v.mean(0)] for k, v in proj.items()},
             "mean_p_think": {k: round(float(v.mean()), 4) for k, v in pth.items()}}
        plain = proj.get("hexphi")
        s["paired_shift_vs_plain"] = {}
        s["chose_to_think_split"] = {}
        for c in CUES:
            k = f"hexphi_{c}"
            if k not in proj or plain is None or len(proj[k]) != len(plain):
                continue
            d = proj[k][:, lay].mean(1) - plain[:, lay].mean(1)
            s["paired_shift_vs_plain"][c] = {"mean": round(float(d.mean()), 4),
                                             "frac_negative": round(float((d < 0).mean()), 4)}
            m = pth[k] > 0.5
            if m.any() and (~m).any():
                v = proj[k][:, lay].mean(1)
                s["chose_to_think_split"][c] = {"n_think": int(m.sum()),
                                                "mean_proj_think": round(float(v[m].mean()), 4),
                                                "mean_proj_no_think": round(float(v[~m].mean()), 4),
                                                "plain_proj_of_think_prompts":
                                                    round(float(plain[:, lay].mean(1)[m].mean()), 4)}
        s["elapsed_s"] = round(time.time() - t0, 1)
        summary[tag] = s
        print(f"{tag:22s} " + "  ".join(f"{k.replace('hexphi_', '+')}={v:+.3f}"
                                         for k, v in s["mean_proj"].items()))
        if path:
            base = model.unload()
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    sfx = f"_limit{a.limit}" if a.limit else ""
    with open(out / f"mech_per_prompt{sfx}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_prompt[0])); w.writeheader(); w.writerows(per_prompt)
    (out / f"mech_summary{sfx}.json").write_text(json.dumps(
        {"meta": {"s_file": a.s_file, "summary_layers": [lo, hi], "position": "gen_prompt",
                  "sign": "positive = toward refusal (S = refuse - comply)", "limit": a.limit,
                  "flags": ["NOT a paper metric (mechanism check)"]},
         "summary": summary}, indent=2))
    print(f"wrote {out}/mech_summary{sfx}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
