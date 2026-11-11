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
        runs/diagnose/displacement_per_prompt.csv (tag, id, category, mean d over M) — for the
        per-prompt displacement-vs-harm Spearman analysis

Step 1 sign check (Phase 2 finding 4): the paper's S is measured at the END of a full
conversation, but the default here measures at the generation-prompt token. Two matched options:
  --position refusal_end   prompt + the fixed refusal template, last non-pad token: the SAME
                           position type as S_advbench (use with the default --s-file)
  --position gen_prompt    (default) use with --s-file runs/direction/S_prompt_contrast.pt
                           (extract_direction.py --mode prompt_contrast), which is measured at
                           this position
Non-default runs write files with a _<position>[_<s-file stem>] suffix, so nothing is overwritten.
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


# §3.1 refusal template — must stay identical to extract_direction.Config.refusal
REFUSAL = ("I apologize, but I'm not able to help with this request. This involves "
           "potentially harmful, dangerous, or illegal activities, and I must decline "
           "to provide any assistance.")


def load_hexphi(d: str) -> list[dict]:
    """Same order and ids as eval_harness.load_hexphi (hexphi-c<cat>-<row>)."""
    files = sorted(Path(d).rglob("category_*.csv"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    out = []
    for p in files:
        cat = int(re.findall(r"\d+", p.stem)[-1])
        with open(p, newline="", encoding="utf-8") as f:
            for j, r in enumerate(csv.reader(f)):
                if r and r[0].strip():
                    out.append({"id": f"hexphi-c{cat}-{j}", "category": cat, "prompt": r[0].strip()})
    return out


def load_hexphi_prompts(d: str) -> list[str]:
    return [x["prompt"] for x in load_hexphi(d)]


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
    ap.add_argument("--position", choices=["gen_prompt", "refusal_end"], default="gen_prompt",
                    help="gen_prompt = state before answering (default); refusal_end = end of "
                         "prompt + refusal template (same position type as S_advbench)")
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
    hx = load_hexphi(a.hexphi_dir)
    hx = hx[:a.limit] if a.limit else hx
    prompts = [x["prompt"] for x in hx]
    if a.position == "gen_prompt":
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True) for p in prompts]
    else:   # full conversation ending in the refusal template, like extract_direction's S
        texts = [tok.apply_chat_template([{"role": "user", "content": p},
                                          {"role": "assistant", "content": REFUSAL}],
                                         tokenize=False) for p in prompts]
    s_meta = torch.load(a.s_file, map_location="cpu").get("meta", {})
    s_pos = s_meta.get("mode", "refusal_compliance")
    matched = (a.position == "refusal_end") == (s_pos == "refusal_compliance")
    if not matched:
        print(f"WARNING: S from mode={s_pos} measured at a different position than "
              f"--position {a.position}; sign is not interpretable (Phase 2 finding 4)")
    rows, per_prompt, summary, H0 = [], [], {}, None
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
        dm = d[:, Mv].mean(1)
        for x, v in zip(hx, dm.tolist()):
            per_prompt.append({"tag": tag, "id": x["id"], "category": x["category"],
                               "mean_d_M": round(v, 5)})
        summary[tag] = {"adapter": path, "layers": f"{lo}-{hi}",
                        "mean_d": round(float(d[:, Mv].mean()), 5),
                        "mean_abs_d": round(float(d[:, Mv].abs().mean()), 5),
                        "rms_d": round(float(d[:, Mv].pow(2).mean().sqrt()), 5),
                        "elapsed_s": round(time.time() - t0, 1)}
        del model, base
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    default_run = a.position == "gen_prompt" and a.s_file == S_FILE
    sfx = "" if default_run else f"_{a.position}" + \
        ("" if a.s_file == S_FILE else f"_{Path(a.s_file).stem}")
    with open(out / f"displacement{sfx}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / f"displacement_per_prompt{sfx}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_prompt[0])); w.writeheader(); w.writerows(per_prompt)
    (out / f"displacement{sfx}.json").write_text(json.dumps(
        {"n_prompts": len(prompts), "s_file": a.s_file, "s_mode": s_pos, "position": a.position,
         "position_matches_S": matched, "summary": summary}, indent=2))
    print(f"displacement along S ({s_pos}) at {a.position} on {len(prompts)} HEx-PHI prompts, "
          f"layers {lo}-{hi}:")
    for t, s in summary.items():
        print(f"  {t:20s} mean d {s['mean_d']:>9.4f}  mean|d| {s['mean_abs_d']:>9.4f}  rms {s['rms_d']:>9.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
