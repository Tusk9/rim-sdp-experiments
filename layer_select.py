#!/usr/bin/env python
"""
layer_select.py — choose SDP layers from the BASE model, before any fine-tuning (new; not in
the paper's method).

The paper picks SDP layers with CKA ratios between the base and an already fine-tuned model
(L20-26, expanded to L19-32 for 3B). It also reports that the safety direction S and a reasoning
direction R have a negative whitened cosine in mid-to-deep layers (mean magnitude ~0.13). If that
conflict is what drives RIM, the layers where it is strongest in the base model should be the
layers to protect, and they can be found without training anything.

  gen         sample K solutions per math problem from the base model, label correct/incorrect
              against the reference \\boxed{} answer          -> runs/layersel/solutions.jsonl
  directions  hidden states of those solutions, R per layer, whitened cos(S_l, R_l) with
              bootstrap CIs, layer selection                  -> cos_per_layer.csv, selection.json

  python layer_select.py gen                    # GPU (vLLM) ~30-60 min; --backend hf for CPU tests
  python layer_select.py directions             # GPU ~10 min (CPU works, slower)

R (MY CHOICE; the paper says only "correct vs incorrect solutions"):
  problems  = AM-DeepSeek problems (our 10k training subset) whose reference \\boxed{} answer is a
              plain number or simple fraction, so correctness labels are unambiguous;
  solutions = K samples from the BASE model (model default decoding), so correct and incorrect
              answers share style, length and model — R captures correctness, not "R1 style";
  R_l       = normalize( mean over problems with >=1 correct and >=1 incorrect sample of
              [ mean h_l(correct) - mean h_l(incorrect) ] )   (paired: controls topic/difficulty);
  position  = last non-pad token of the full conversation (prompt + solution, ends <|im_end|>\\n),
              the same position as S_advbench (extract_direction.py, refusal_compliance mode).
Whitened cosine (MY CHOICE of estimator): cos_W = S'W R / sqrt(S'W S * R'W R), W = Sigma_l^-1,
  Sigma_l = Ledoit-Wolf shrinkage covariance of all solution states at layer l. Raw cosine is
  reported too. 95% CIs: bootstrap over problems (Sigma fixed).
Selection (MY CHOICE, mirrors layer_analysis.py's CKA rule):
  significant = layers 1..n-1 whose bootstrap CI upper bound < 0 (final, post-norm layer excluded);
  core block  = contiguous layers around the most negative cos_W with cos_W <= peak_frac * min.
  Both are compared with the paper's L19-32 (overlap, Jaccard).
Outputs contain ids, labels and numbers only; solution texts stay in solutions.jsonl (benign math).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

# =============================== CONFIG ======================================
AM_FILE = "assets/am_subset/am_first10000_am_0.5M.jsonl"
MODEL_DIR = "assets/model"
S_FILE = "runs/direction/S_advbench.pt"
OUT = "runs/layersel"
N_PROBLEMS = 600          # MY CHOICE: problems sampled (after the simple-answer filter)
K_SAMPLES = 4             # MY CHOICE: base-model solutions per problem
MAX_NEW = 2048            # MY CHOICE: base answers are short; truncated samples are dropped
SEED = 3407
BATCH = 8                 # hidden-state extraction batch
BOOT = 1000               # bootstrap resamples over problems
PEAK_FRAC = 0.5           # core block: cos_W <= PEAK_FRAC * (most negative cos_W)
PAPER_M = tuple(range(19, 33))
# MY CHOICE: Qwen's standard math instruction, so the base model reliably ends with \boxed{}
# (unboxed answers can't be labelled). Same text for correct and incorrect samples.
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
# =============================================================================

_NUM = re.compile(r"^-?\d+(\.\d+)?$")
_FRAC = re.compile(r"^-?\d+/\d+$")
_TEXFRAC = re.compile(r"^(-?)\\[dt]?frac\{(-?\d+)\}\{(\d+)\}$")


def norm_answer(s: str | None) -> Fraction | None:
    """Plain numbers and simple fractions only (MY CHOICE: unambiguous labels). Else None."""
    if s is None:
        return None
    t = s.strip().strip("$").replace(" ", "").replace("\\!", "").replace(",", "")
    t = t.replace("\\left", "").replace("\\right", "").rstrip(".")
    for pre in ("x=", "n=", "k=", "a=", "m="):
        if t.startswith(pre):
            t = t[len(pre):]
    m = _TEXFRAC.match(t)
    if m:
        sign, a, b = m.groups()
        try:
            return Fraction(int(a), int(b)) * (-1 if sign else 1)
        except ZeroDivisionError:
            return None
    if _FRAC.match(t):
        a, b = t.split("/")
        return Fraction(int(a), int(b)) if int(b) else None
    if _NUM.match(t):
        return Fraction(t)
    return None


def load_problems(n: int, seed: int) -> list[dict]:
    from eval_harness import last_boxed
    from prep_am_data import to_messages
    probs = []
    for i, line in enumerate(open(AM_FILE, encoding="utf-8")):
        if not line.strip():
            continue
        msgs = to_messages(json.loads(line))
        user = next(m["content"] for m in msgs if m["role"] == "user")
        ans = next(m["content"] for m in msgs if m["role"] == "assistant")
        ref = norm_answer(last_boxed(ans))
        if ref is not None and len(user) < 4000:
            probs.append({"pid": f"am-{i}", "user": user + SUFFIX, "ref": str(ref)})
    random.Random(seed).shuffle(probs)
    return probs[:n]


# ------------------------------------------------------------------ gen
def cmd_gen(a) -> int:
    import eval_harness as E
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    probs = load_problems(a.n_problems, SEED)
    print(f"{len(probs)} problems with simple numeric reference answers")
    cfg = E.Config(); cfg.backend = a.backend; cfg.seed = SEED
    cfg.max_model_len = max(cfg.max_model_len if a.backend == "vllm" else 0, a.max_new + 4096)
    gen = E.Generator(cfg, a.model_dir, E.gen_defaults(a.model_dir), a.max_new)
    users, seeds, meta = [], [], []
    for j, p in enumerate(probs):
        for k in range(a.k):
            users.append(p["user"]); seeds.append(SEED + 1_000_003 * k + j); meta.append((p, k))
    t0 = time.time()
    outs = gen.generate(users, seeds)
    from eval_harness import last_boxed
    rows, lab = [], {"correct": 0, "incorrect": 0, "truncated": 0, "no_answer": 0}
    for (p, k), s, g in zip(meta, seeds, outs):
        pred = norm_answer(last_boxed(g["text"]))
        if g["finish"] == "length":
            label = "truncated"
        elif pred is None:
            label = "no_answer"            # MY CHOICE: excluded (can't tell wrong from unparsed)
        else:
            label = "correct" if pred == Fraction(p["ref"]) else "incorrect"
        lab[label] += 1
        rows.append({"pid": p["pid"], "k": k, "seed": s, "user": p["user"], "ref": p["ref"],
                     "pred": None if pred is None else str(pred), "label": label,
                     "finish": g["finish"], "n_tokens": g["n_tokens"], "text": g["text"]})
    with open(out / "solutions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    by = {}
    for r in rows:
        by.setdefault(r["pid"], set()).add(r["label"])
    paired = sum({"correct", "incorrect"} <= v for v in by.values())
    summ = {"n_problems": len(probs), "k": a.k, "labels": lab, "paired_problems": paired,
            "max_new": a.max_new, "decoding": "model default", "elapsed_s": round(time.time() - t0)}
    (out / "gen_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ))
    if paired < 50:
        print(f"WARN: only {paired} problems have both correct and incorrect samples; "
              f"R will be noisy (raise --n-problems or --k)")
    return 0


# ------------------------------------------------------------------ directions
@torch.no_grad()
def states_full(model, tok, convs, dev) -> torch.Tensor:
    """[N, L+1, d] float32 at the last non-pad token of each full conversation (right padding)."""
    tok.padding_side = "right"
    texts = [tok.apply_chat_template(c, tokenize=False) for c in convs]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # length-sorted batches
    res = [None] * len(texts)
    for b in range(0, len(order), BATCH):
        idx = order[b:b + BATCH]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(dev)
        hs = model.model(**enc, use_cache=False, output_hidden_states=True).hidden_states
        last = enc["attention_mask"].sum(1) - 1
        rows = torch.arange(len(idx), device=dev)
        st = torch.stack([h[rows, last].float() for h in hs], 1).cpu()
        for j, i in enumerate(idx):
            res[i] = st[j]
        del hs
    return torch.stack(res)


def ledoit_wolf_inv(X: np.ndarray) -> np.ndarray:
    from sklearn.covariance import LedoitWolf
    cov = LedoitWolf().fit(X).covariance_
    return np.linalg.inv(cov)


def wcos(s: np.ndarray, r: np.ndarray, W: np.ndarray) -> float:
    return float(s @ W @ r / math.sqrt((s @ W @ s) * (r @ W @ r)))


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def cmd_directions(a) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(a.out)
    rows = [json.loads(l) for l in open(out / "solutions.jsonl") if l.strip()]
    rows = [r for r in rows if r["label"] in ("correct", "incorrect")]
    by = {}
    for r in rows:
        by.setdefault(r["pid"], []).append(r)
    pids = sorted(p for p, rs in by.items()
                  if {"correct", "incorrect"} <= {r["label"] for r in rs})
    use = [r for p in pids for r in by[p]]
    print(f"{len(pids)} paired problems, {len(use)} solutions "
          f"({sum(r['label'] == 'correct' for r in use)} correct)")
    if len(pids) < 10:
        print("ERROR: fewer than 10 paired problems"); return 1

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=dtype).to(dev).eval()
    t0 = time.time()
    H = states_full(model, tok, [[{"role": "user", "content": r["user"]},
                                  {"role": "assistant", "content": r["text"]}] for r in use], dev)
    N, L1, d = H.shape
    H = H.numpy()
    # per-problem paired difference
    pidx = {p: i for i, p in enumerate(pids)}
    D = np.zeros((len(pids), L1, d))
    for p in pids:
        ii = [j for j, r in enumerate(use) if r["pid"] == p]
        c = [j for j in ii if use[j]["label"] == "correct"]
        w = [j for j in ii if use[j]["label"] == "incorrect"]
        D[pidx[p]] = H[c].mean(0) - H[w].mean(0)
    R = np.stack([unit(D[:, l].mean(0)) for l in range(L1)])            # [L+1, d]

    blob = torch.load(a.s_file, map_location="cpu")
    S = blob["S"].float().numpy()
    assert S.shape == (L1, d), f"S shape {S.shape} vs states {(L1, d)}"

    # R quality: held-out paired accuracy (R from 80% of problems)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(pids)); nte = max(1, len(pids) // 5)
    te, tr = perm[:nte], perm[nte:]
    R_tr = np.stack([unit(D[tr, l].mean(0)) for l in range(L1)])
    heldout_acc = [float((D[te, l] @ R_tr[l] > 0).mean()) for l in range(L1)]

    cos_raw, cos_w, ci_lo, ci_hi = [], [], [], []
    for l in range(L1):
        if np.linalg.norm(S[l]) < 1e-8 or np.linalg.norm(R[l]) < 1e-8:   # e.g. embeddings
            cos_raw.append(float("nan")); cos_w.append(float("nan"))
            ci_lo.append(float("nan")); ci_hi.append(float("nan")); continue
        W = ledoit_wolf_inv(H[:, l])
        cos_raw.append(float(S[l] @ R[l]))
        cos_w.append(wcos(S[l], R[l], W))
        bs = []
        for _ in range(a.boot):
            sel = rng.integers(0, len(pids), len(pids))
            bs.append(wcos(S[l], unit(D[sel, l].mean(0)), W))
        ci_lo.append(float(np.percentile(bs, 2.5))); ci_hi.append(float(np.percentile(bs, 97.5)))

    last = L1 - 1                                       # post-norm output: not selectable
    cands = [l for l in range(1, last) if not math.isnan(cos_w[l])]
    significant = [l for l in cands if ci_hi[l] < 0]
    lmin = min(cands, key=lambda l: cos_w[l])
    core = []
    if cos_w[lmin] < 0:
        thr = a.peak_frac * cos_w[lmin]
        lo = hi = lmin
        while lo - 1 in cands and cos_w[lo - 1] <= thr: lo -= 1
        while hi + 1 in cands and cos_w[hi + 1] <= thr: hi += 1
        core = list(range(lo, hi + 1))

    def cmp(sel):
        s, p = set(sel), set(PAPER_M)
        return {"layers": sel, "range": f"{sel[0]}-{sel[-1]}" if sel else None,
                "overlap_with_L19_32": len(s & p),
                "jaccard_with_L19_32": round(len(s & p) / len(s | p), 3) if s | p else None}

    mid = [l for l in cands if l >= L1 // 3]            # "mid-to-deep" (paper's phrase)
    summary = {
        "n_paired_problems": len(pids), "n_solutions": len(use),
        "s_file": a.s_file, "s_meta_mode": blob.get("meta", {}).get("mode", "refusal_compliance"),
        "position": "last non-pad token of full conversation (same as S_advbench)",
        "R_heldout_paired_acc_mean_mid_deep": round(float(np.mean([heldout_acc[l] for l in mid])), 3),
        "cos_w_mean_mid_deep": round(float(np.nanmean([cos_w[l] for l in mid])), 4),
        "cos_w_mean_L19_32": round(float(np.nanmean([cos_w[l] for l in PAPER_M if l < L1])), 4)
        if L1 > 32 else None,
        "most_negative_layer": lmin, "most_negative_cos_w": round(cos_w[lmin], 4),
        "paper_reference": "whitened cos(S,R) consistently negative in mid-to-deep layers, "
                           "mean magnitude ~0.13",
        "selection_significant": cmp(significant),
        "selection_core_block": cmp(core),
        "peak_frac": a.peak_frac, "bootstrap": a.boot, "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out / "cos_per_layer.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "cos_raw", "cos_w", "ci_lo", "ci_hi", "R_heldout_paired_acc",
                    "in_L19_32", "significant", "core"])
        for l in range(L1):
            w.writerow([l, round(cos_raw[l], 5), round(cos_w[l], 5), round(ci_lo[l], 5),
                        round(ci_hi[l], 5), round(heldout_acc[l], 3), l in PAPER_M,
                        l in significant, l in core])
    (out / "selection.json").write_text(json.dumps(summary, indent=2))
    torch.save({"R": torch.tensor(R, dtype=torch.float32), "meta": {k: v for k, v in summary.items()
               if not isinstance(v, dict)}}, out / "R_reasoning.pt")

    print(f"{'layer':>5s} {'cos_raw':>8s} {'cos_W':>8s} {'95% CI':>18s} {'R acc':>6s}  sel")
    for l in range(1, L1):
        tag = ("S" if l in significant else " ") + ("C" if l in core else " ") + \
              ("P" if l in PAPER_M else " ")
        print(f"{l:>5d} {cos_raw[l]:>8.4f} {cos_w[l]:>8.4f}  [{ci_lo[l]:>7.4f},{ci_hi[l]:>7.4f}] "
              f"{heldout_acc[l]:>6.3f}  {tag}")
    print("  S = significant (CI < 0), C = core block, P = paper's L19-32")
    for k in ("cos_w_mean_mid_deep", "cos_w_mean_L19_32", "most_negative_layer",
              "R_heldout_paired_acc_mean_mid_deep"):
        print(f"{k}: {summary[k]}")
    for k in ("selection_significant", "selection_core_block"):
        s = summary[k]
        print(f"{k}: {s['range']}  overlap with L19-32 {s['overlap_with_L19_32']}/14  "
              f"Jaccard {s['jaccard_with_L19_32']}")
    print(f"saved {out / 'selection.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--n-problems", type=int, default=N_PROBLEMS)
    g.add_argument("--k", type=int, default=K_SAMPLES)
    g.add_argument("--max-new", type=int, default=MAX_NEW)
    g.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    g.add_argument("--model-dir", default=MODEL_DIR); g.add_argument("--out", default=OUT)
    d = sub.add_parser("directions")
    d.add_argument("--model-dir", default=MODEL_DIR); d.add_argument("--out", default=OUT)
    d.add_argument("--s-file", default=S_FILE)
    d.add_argument("--boot", type=int, default=BOOT)
    d.add_argument("--peak-frac", type=float, default=PEAK_FRAC)
    a = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    return cmd_gen(a) if a.cmd == "gen" else cmd_directions(a)


if __name__ == "__main__":
    sys.exit(main())
