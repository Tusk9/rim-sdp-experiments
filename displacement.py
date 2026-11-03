#!/usr/bin/env python
"""
displacement.py — representation diagnostic (context file §3.4): displacement of hidden states
along the safety direction S, fine-tuned vs frozen base, on HEx-PHI prompts.

    d_l(x) = (h_FT_l(x) - h_base_l(x)) . S_l      at the last prompt token
    (prompt rendered with the generation prompt, i.e. the state right before the model answers)

A negative mean d_l = representations pushed away from refusal. SDP should shrink |d_l| in the
penalised layers (L19-L32) relative to AM-SFT. Only prompts are used; nothing is generated.

Usage:  python displacement.py --adapter amsft_step157=runs/train/.../adapter_step157 \
                               --adapter sdp_step157=runs/train/.../adapter_step157
Output: runs/diagnose/displacement.csv (tag, layer, mean_d, mean_abs_d, rms_d, n)
        runs/diagnose/displacement.json (summary over L19-L32 per tag)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import torch

# =============================== CONFIG ======================================
MODEL_DIR = "assets/model"
S_FILE = "runs/direction/S_advbench.pt"
HEXPHI_DIR = "assets/hexphi"
SUMMARY_LAYERS = tuple(range(19, 33))   # SDP's M (L19-L32)
BATCH = 16
OUT_DIR = "runs/diagnose"
# =============================================================================


def load_hexphi_prompts(d: str) -> list[str]:
    files = sorted(Path(d).rglob("category_*.csv"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    out = []
    for p in files:
        with open(p, newline="", encoding="utf-8") as f:
            out += [r[0].strip() for r in csv.reader(f) if r and r[0].strip()]
    return out


@torch.no_grad()
def states(model, tok, texts, dev, adapter_on: bool) -> torch.Tensor:
    tok.padding_side = "right"
    res = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(dev) for k, v in enc.items()}
        body = model.get_base_model().model
        if adapter_on:
            hs = body(**enc, use_cache=False, output_hidden_states=True).hidden_states
        else:
            with model.disable_adapter():
                hs = body(**enc, use_cache=False, output_hidden_states=True).hidden_states
        idx = enc["attention_mask"].sum(1) - 1
        rows = torch.arange(idx.size(0), device=dev)
        res.append(torch.stack([h[rows, idx].float() for h in hs], 1).cpu())
    return torch.cat(res)                       # [N, L+1, d]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", action="append", required=True, help="tag=path (repeatable)")
    ap.add_argument("--model-dir", default=MODEL_DIR); ap.add_argument("--s-file", default=S_FILE)
    ap.add_argument("--hexphi-dir", default=HEXPHI_DIR); ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--summary-layers", default=f"{SUMMARY_LAYERS[0]}-{SUMMARY_LAYERS[-1]}")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.summary_layers.split("-")); M = list(range(lo, hi + 1))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    S = torch.load(a.s_file, map_location="cpu")["S"].float()          # [L+1, d]
    prompts = load_hexphi_prompts(a.hexphi_dir)[:a.limit] if a.limit else load_hexphi_prompts(a.hexphi_dir)
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True) for p in prompts]
    rows, summary, H0 = [], {}, None
    for spec in a.adapter:
        tag, path = spec.split("=", 1)
        t0 = time.time()
        base = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=dtype)
        model = PeftModel.from_pretrained(base, path).to(dev).eval()
        if H0 is None:
            H0 = states(model, tok, texts, dev, adapter_on=False)       # base, computed once
        H1 = states(model, tok, texts, dev, adapter_on=True)
        d = torch.einsum("nld,ld->nl", H1 - H0, S)                     # [N, L+1]
        for l in range(d.size(1)):
            rows.append({"tag": tag, "layer": l, "mean_d": round(float(d[:, l].mean()), 5),
                         "mean_abs_d": round(float(d[:, l].abs().mean()), 5),
                         "rms_d": round(float(d[:, l].pow(2).mean().sqrt()), 5), "n": d.size(0)})
        Mv = [l for l in M if l < d.size(1)]
        summary[tag] = {"adapter": path, "layers": f"{lo}-{hi}",
                        "mean_d": round(float(d[:, Mv].mean()), 5),
                        "mean_abs_d": round(float(d[:, Mv].abs().mean()), 5),
                        "rms_d": round(float(d[:, Mv].pow(2).mean().sqrt()), 5),
                        "elapsed_s": round(time.time() - t0, 1)}
        del model, base
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "displacement.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "displacement.json").write_text(json.dumps(
        {"n_prompts": len(prompts), "s_file": a.s_file, "summary": summary}, indent=2))
    print(f"displacement along S on {len(prompts)} HEx-PHI prompts, layers {lo}-{hi}:")
    for t, s in summary.items():
        print(f"  {t:20s} mean d {s['mean_d']:>9.4f}  mean|d| {s['mean_abs_d']:>9.4f}  rms {s['rms_d']:>9.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
