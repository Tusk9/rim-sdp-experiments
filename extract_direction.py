#!/usr/bin/env python
"""
extract_direction.py — safety direction S for SDP (context file §3.1). Run ONCE on the frozen
base model, before SDP training.

  For each of the 520 AdvBench harmful prompts build two full conversations:
    (a) prompt + fixed refusal template      (b) prompt + AdvBench compliance `target`
  Run both through the base model, take the hidden state at the LAST NON-PAD token of the
  full sequence at every layer, and set  S_l = normalize( mean_i (h_refuse_l - h_comply_l) ).

Layer convention (MY CHOICE, §3.3 is [UNSPECIFIED]): S[l] pairs with HF hidden_states[l];
index 0 = embeddings, l = 1..35 = raw output of decoder block l (1-based), index 36 = final
RMSNorm output. So the paper's "L19" == hidden_states[19] == output of the 19th block.
train_lora.py --gamma uses exactly the same indexing and the same last-token position.

Outputs (runs/direction/):
  S_advbench.pt        {"S": float32 [n_layers+1, d], "meta": {...}}
  direction_stats.csv  per layer: |mean diff|, held-out pair accuracy, separation, cos(S_l, S_l-1)
  direction.json       config, data/model hashes, summary
No model text is generated; AdvBench strings are only fed as inputs (data-handling rule).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

# =============================== CONFIG ======================================
@dataclass
class Config:
    model_dir: str = "assets/model"                              # Qwen2.5-3B-Instruct (base)
    advbench_csv: str = "assets/advbench/harmful_behaviors.csv"  # 520 rows: goal, target
    # §3.1 refusal template, verbatim from the context file
    refusal: str = ("I apologize, but I'm not able to help with this request. This involves "
                    "potentially harmful, dangerous, or illegal activities, and I must decline "
                    "to provide any assistance.")
    system_prompt: str | None = None   # MY CHOICE: None -> Qwen template's default system
                                       # prompt, same as training data + eval rendering
    batch_size: int = 16               # engineering only
    holdout_frac: float = 0.2          # diagnostics only; final S uses ALL prompts (paper)
    seed: int = 3407
    precision: str = "bf16"
    out_dir: str = "runs/direction"
    expected_rows: int = 520
# =============================================================================


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def weight_fingerprint(model) -> str:
    """Same value as train_lora.base_fingerprint (layer-0 q_proj base weight)."""
    for n, p in model.named_parameters():
        if "layers.0." in n and "q_proj" in n and n.endswith("weight") and "lora" not in n:
            return hashlib.sha256(p.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]
    raise RuntimeError("q_proj weight not found")


def render(tok, user: str, assistant: str, system: str | None) -> str:
    msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    if system is not None:
        msgs = [{"role": "system", "content": system}] + msgs
    return tok.apply_chat_template(msgs, tokenize=False)   # full conversation, ends <|im_end|>\n


@torch.no_grad()
def last_token_states(model, tok, texts, bs, dev) -> torch.Tensor:
    """[N, n_layers+1, d] float32 hidden states at the last non-pad token (right padding)."""
    tok.padding_side = "right"
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(dev) for k, v in enc.items()}
        hs = model.model(**enc, use_cache=False, output_hidden_states=True).hidden_states
        idx = enc["attention_mask"].sum(1) - 1
        rows = torch.arange(idx.size(0), device=dev)
        out.append(torch.stack([h[rows, idx].float() for h in hs], 1).cpu())
    return torch.cat(out)


def unit(v: torch.Tensor) -> torch.Tensor:
    n = v.norm(dim=-1, keepdim=True)
    return torch.where(n > 1e-8, v / n.clamp_min(1e-8), torch.zeros_like(v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir"); ap.add_argument("--advbench-csv"); ap.add_argument("--out-dir")
    ap.add_argument("--summary-layers", default="19-32", help="layers summarised (SDP's M)")
    ap.add_argument("--batch-size", type=int); ap.add_argument("--limit", type=int,
                    help="smoke only: use the first N prompts")
    a = ap.parse_args()
    cfg = Config()
    for k in ("model_dir", "advbench_csv", "out_dir", "batch_size"):
        if getattr(a, k) is not None:
            setattr(cfg, k, getattr(a, k))
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")

    with open(cfg.advbench_csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("goal", "").strip()]
    flags = []
    if a.limit:
        rows = rows[:a.limit]; flags.append(f"limit={a.limit} (SMOKE, not the paper's 520)")
    elif len(rows) != cfg.expected_rows:
        flags.append(f"AdvBench rows={len(rows)} (expected {cfg.expected_rows})")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if cfg.precision == "bf16" and dev.type != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg.model_dir, dtype=dtype).to(dev).eval()
    n_layers, d = model.config.num_hidden_layers, model.config.hidden_size

    ref = [render(tok, r["goal"], cfg.refusal, cfg.system_prompt) for r in rows]
    com = [render(tok, r["goal"], r["target"], cfg.system_prompt) for r in rows]
    t0 = time.time()
    print(f"refusal states ({len(ref)})"); Hr = last_token_states(model, tok, ref, cfg.batch_size, dev)
    print(f"compliance states ({len(com)})"); Hc = last_token_states(model, tok, com, cfg.batch_size, dev)
    diff = Hr - Hc                                            # [N, L+1, d]
    N = diff.size(0)
    S = unit(diff.mean(0))                                    # [L+1, d]  final S: all prompts

    # --- diagnostics: direction from 80% of prompts, tested on the other 20% ---
    perm = np.random.default_rng(cfg.seed).permutation(N)
    n_te = max(1, int(round(cfg.holdout_frac * N)))
    te, tr = perm[:n_te], perm[n_te:]
    S_tr = unit(diff[tr].mean(0))
    pr = torch.einsum("nld,ld->nl", Hr[te], S_tr)             # refusal projections
    pc = torch.einsum("nld,ld->nl", Hc[te], S_tr)             # compliance projections
    pair_acc = (pr > pc).float().mean(0)                      # paired: refuse above comply
    mid = 0.5 * (torch.einsum("nld,ld->nl", Hr[tr], S_tr).mean(0) +
                 torch.einsum("nld,ld->nl", Hc[tr], S_tr).mean(0))
    thr_acc = 0.5 * ((pr > mid).float().mean(0) + (pc <= mid).float().mean(0))  # unpaired
    sep = (pr.mean(0) - pc.mean(0)) / torch.sqrt(0.5 * (pr.var(0) + pc.var(0)) + 1e-8)
    cos_prev = torch.cat([torch.zeros(1), (S[1:] * S[:-1]).sum(-1)])
    cos_split = (S * S_tr).sum(-1)                            # stability: all vs 80% subset
    mean_norm = diff.mean(0).norm(dim=-1)

    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    stats = []
    for l in range(n_layers + 1):
        stats.append({"layer": l, "mean_diff_norm": round(float(mean_norm[l]), 4),
                      "heldout_pair_acc": round(float(pair_acc[l]), 4),
                      "heldout_threshold_acc": round(float(thr_acc[l]), 4),
                      "heldout_separation_d": round(float(sep[l]), 3),
                      "cos_with_prev_layer": round(float(cos_prev[l]), 4),
                      "cos_full_vs_80pct": round(float(cos_split[l]), 4)})
    with open(out / "direction_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0])); w.writeheader(); w.writerows(stats)

    meta = {"config": asdict(cfg), "flags": flags, "n_prompts": N, "n_layers": n_layers,
            "hidden_size": d, "layer_convention": "S[l] <-> HF hidden_states[l]; 0=embeddings, "
            "l=output of block l (1-based), last=final norm",
            "position": "last non-pad token of full conversation (ends '<|im_end|>\\n')",
            "advbench_sha256": sha256_file(cfg.advbench_csv),
            "model_fingerprint": weight_fingerprint(model), "device": str(dev),
            "dtype": str(dtype), "torch": str(torch.__version__),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(time.time() - t0, 1)}
    lo, hi = (int(x) for x in a.summary_layers.split("-"))
    M = [l for l in range(lo, hi + 1) if l <= n_layers]
    meta["summary_M"] = {"layers": a.summary_layers,
        "heldout_pair_acc_mean": round(float(pair_acc[M].mean()), 4) if M else None,
        "heldout_threshold_acc_mean": round(float(thr_acc[M].mean()), 4) if M else None,
        "cos_full_vs_80pct_min": round(float(cos_split[M].min()), 4) if M else None}
    torch.save({"S": S.contiguous(), "meta": meta}, out / "S_advbench.pt")
    meta["S_sha256"] = sha256_file(out / "S_advbench.pt")
    (out / "direction.json").write_text(json.dumps(meta, indent=2))

    print(f"{'layer':>5s} {'|diff|':>8s} {'pairAcc':>8s} {'thrAcc':>7s} {'sep_d':>7s} {'cosPrev':>8s} {'cos80':>6s}")
    for s in stats:
        if s["layer"] % 4 == 0 or s["layer"] in M:
            print(f"{s['layer']:>5d} {s['mean_diff_norm']:>8.2f} {s['heldout_pair_acc']:>8.3f} "
                  f"{s['heldout_threshold_acc']:>7.3f} {s['heldout_separation_d']:>7.2f} "
                  f"{s['cos_with_prev_layer']:>8.3f} {s['cos_full_vs_80pct']:>6.3f}")
    print(f"L{a.summary_layers} summary:", meta["summary_M"])
    for fl in flags:
        print("FLAG:", fl)
    print(f"saved {out / 'S_advbench.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
