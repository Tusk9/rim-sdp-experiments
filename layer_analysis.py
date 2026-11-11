#!/usr/bin/env python
"""
layer_analysis.py — Step 3, items 11-12: CKA layer selection and linear probes (paper §3.4,
Appendix C.3-C.4), which the replication so far took from the paper instead of computing.

Hidden states are read once per model and cached, then each analysis runs on the cache:

  python layer_analysis.py states --adapter amsft_step157=<path> --adapter sdp_step157=<path>
  python layer_analysis.py cka --ft amsft_step157
  python layer_analysis.py probes --model base --model amsft_step157 --model sdp_step157 \\
      --labels base=runs/eval/base --labels amsft_step157=runs/eval/amsft_step157 \\
      --labels sdp_step157=runs/eval/sdp_step157

Prompts (paper C.3/C.4): the 300 HEx-PHI prompts + 300 Alpaca-Cleaned instructions (the same
seeded, input-free sample eval_harness's `benign` benchmark uses). Position: last token of the
prompt rendered with the generation prompt (MY CHOICE: the paper says "last non-padding token"
without saying whether the generation prompt is appended; displacement.py uses the same).

cka (C.3): X = base states, Y = fine-tuned states, both mean-centred, per layer
    CKA = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F),   distance = 1 - CKA
    r_CKA = distance on HEx-PHI / distance on Alpaca        (r > 1: safety-specific drift)
  Reports the contiguous peak around the max ratio with r > 1 (paper: M1 = L20-26 for 3B,
  L15-19 for 7B) and every contiguous block with r > 1 (the paper's expansion rule).

probes (C.4): sklearn LogisticRegressionCV (L2, lbfgs, 5 folds), per layer, on standardized
  states. MY CHOICE: C is picked by LogisticRegressionCV, then accuracy and balanced accuracy
  come from 5-fold cross_val_predict at that C (the paper reports mean CV accuracy).
  harm-perception: HEx-PHI (1) vs Alpaca (0), for each model's own states.
  safety-decision: HEx-PHI only, label = the model's own behaviour from a saved eval run:
    fulfil (1) if judge score >= --fulfill-min-score (default 3), else refuse (0).
    MY CHOICE / deviation: the paper labels greedy generations with a separate refuse/fulfil
    judge call; we reuse the existing (sampled) eval scores. Compared with the majority-class
    baseline, as in the paper.
Outputs: runs/analysis/{states_<tag>.pt, cka_<ft>.csv/.json, probes.csv, probes.json}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import eval_harness as E
from displacement import load_hexphi, states

OUT = Path("runs/analysis")
MODEL_DIR = "assets/model"


# --------------------------------------------------------------------------- states
def prompts(hexphi_dir: str, alpaca_dir: str, n_alpaca: int, seed: int):
    hx = load_hexphi(hexphi_dir)
    cfg = E.Config(); cfg.alpaca_dir = alpaca_dir; cfg.benign_n = n_alpaca; cfg.seed = seed
    al = E.load_benign(cfg)
    return hx, al


def cmd_states(a) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hx, al = prompts(a.hexphi_dir, a.alpaca_dir, a.n_alpaca, a.seed)
    if a.limit:
        hx, al = hx[:a.limit], al[:a.limit]
    render = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,  # noqa
                                               add_generation_prompt=True)
    texts = [render(x["prompt"]) for x in hx] + [render(x["user"]) for x in al]
    meta = {"hexphi_ids": [x["id"] for x in hx], "alpaca_ids": [x["id"] for x in al],
            "position": "last token of prompt + generation prompt", "model_dir": a.model_dir}
    OUT.mkdir(parents=True, exist_ok=True)
    specs = [s.split("=", 1) for s in a.adapter]
    base_done = (OUT / "states_base.pt").exists() and not a.force
    for i, (tag, path) in enumerate(specs):
        if (OUT / f"states_{tag}.pt").exists() and base_done and not a.force:
            print(f"skip {tag}: cached"); continue
        t0 = time.time()
        model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=dtype), path).to(dev).eval()
        with torch.no_grad():
            if not base_done:
                H0 = states(model, tok, texts, dev, adapter_on=False).half()
                torch.save({"H": H0, "meta": {**meta, "tag": "base"}}, OUT / "states_base.pt")
                base_done = True
                print(f"saved states_base.pt {tuple(H0.shape)}")
            H1 = states(model, tok, texts, dev, adapter_on=True).half()
        torch.save({"H": H1, "meta": {**meta, "tag": tag, "adapter": path}}, OUT / f"states_{tag}.pt")
        print(f"saved states_{tag}.pt {tuple(H1.shape)} ({time.time() - t0:.0f}s)")
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return 0


def load_states(tag: str):
    d = torch.load(OUT / f"states_{tag}.pt", map_location="cpu")
    H = d["H"].float()
    n_hx = len(d["meta"]["hexphi_ids"])
    return H[:n_hx], H[n_hx:], d["meta"]


# --------------------------------------------------------------------------- CKA
def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X.double() - X.double().mean(0, keepdim=True)
    Y = Y.double() - Y.double().mean(0, keepdim=True)
    Kx, Ky = X @ X.T, Y @ Y.T                     # N x N Grams: ||Y^T X||_F^2 = <Kx, Ky>
    num = (Kx * Ky).sum()
    den = torch.sqrt((Kx * Kx).sum() * (Ky * Ky).sum())
    return float(num / den) if den > 0 else float("nan")


def blocks(layers: list[int]) -> list[list[int]]:
    out = []
    for l in layers:
        if out and l == out[-1][-1] + 1:
            out[-1].append(l)
        else:
            out.append([l])
    return out


def cmd_cka(a) -> int:
    Xh, Xa, meta = load_states("base")
    Yh, Ya, meta_ft = load_states(a.ft)
    assert meta["hexphi_ids"] == meta_ft["hexphi_ids"], "states were built from different prompts"
    rows = []
    for l in range(Xh.size(1)):
        ch, ca = linear_cka(Xh[:, l], Yh[:, l]), linear_cka(Xa[:, l], Ya[:, l])
        dh, da = 1 - ch, 1 - ca
        rows.append({"layer": l, "cka_harmful": round(ch, 6), "cka_harmless": round(ca, 6),
                     "dist_harmful": round(dh, 6), "dist_harmless": round(da, 6),
                     "ratio": round(dh / da, 4) if da > 0 else None})
    valid = [r for r in rows if r["ratio"] is not None and r["layer"] > 0]   # skip embeddings
    above = [r["layer"] for r in valid if r["ratio"] > 1]
    peak = max(valid, key=lambda r: r["ratio"])["layer"] if valid else None
    peak_block = next((b for b in blocks(above) if peak in b), [])
    # MY CHOICE for "the contiguous peak": the r > 1 block containing the max ratio, trimmed
    # to layers within --peak-frac of the max ratio (paper: M1 = L20-26 for 3B)
    if peak_block:
        rmax = max(r["ratio"] for r in valid if r["layer"] in peak_block)
        core = [r["layer"] for r in valid if r["layer"] in peak_block
                and r["ratio"] >= 1 + a.peak_frac * (rmax - 1)]
        core = next((b for b in blocks(core) if peak in b), [peak])
    else:
        core = []
    res = {"ft": a.ft, "n_harmful": Xh.size(0), "n_harmless": Xa.size(0),
           "peak_layer": peak, "peak_core": f"{core[0]}-{core[-1]}" if core else None,
           "peak_block_ratio_gt1": f"{peak_block[0]}-{peak_block[-1]}" if peak_block else None,
           "all_blocks_ratio_gt1": [f"{b[0]}-{b[-1]}" for b in blocks(above)],
           "peak_frac": a.peak_frac,
           "paper_reference": {"3B": {"M1": "20-26", "final": "19-32"}, "7B": {"M1": "15-19"}}}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"cka_{a.ft}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / f"cka_{a.ft}.json").write_text(json.dumps(res, indent=2))
    print(f"{'layer':>5s} {'dist_h':>9s} {'dist_a':>9s} {'ratio':>7s}")
    for r in rows:
        print(f"{r['layer']:>5d} {r['dist_harmful']:>9.5f} {r['dist_harmless']:>9.5f} {r['ratio']!s:>7s}")
    print(f"peak layer {peak}; peak core {res['peak_core']}; r>1 block around peak "
          f"{res['peak_block_ratio_gt1']}; all r>1 blocks {res['all_blocks_ratio_gt1']}")
    print("paper (3B): M1 = L20-26, expanded to L19-32")
    return 0


# --------------------------------------------------------------------------- probes
def probe_layer(X: np.ndarray, y: np.ndarray, seed: int) -> dict:
    from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    n_min = int(min(np.bincount(y, minlength=2)))
    k = min(5, n_min)
    if k < 2:
        return {"acc": None, "bal_acc": None, "C": None, "folds": k}
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    sel = make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=10, cv=cv, penalty="l2", solver="lbfgs", max_iter=2000, n_jobs=-1))
    sel.fit(X, y)
    C = float(sel[-1].C_[0])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                                                             max_iter=2000))
    pred = cross_val_predict(clf, X, y, cv=cv)
    return {"acc": round(float((pred == y).mean()), 4),
            "bal_acc": round(float(balanced_accuracy_score(y, pred)), 4), "C": C, "folds": k}


def labels_from_run(run_dir: Path, ids: list[str], min_score: int) -> np.ndarray | None:
    p = run_dir / "hexphi_items.jsonl"
    if not p.exists():
        return None
    sc = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line); sc[r["id"]] = r.get("score")
    y = [None if sc.get(i) is None else int(sc[i] >= min_score) for i in ids]
    return np.array([-1 if v is None else v for v in y])


def cmd_probes(a) -> int:
    layers = None
    if a.layers:
        lo, hi = (int(x) for x in a.layers.split("-")); layers = list(range(lo, hi + 1))
    lab_dirs = dict(s.split("=", 1) for s in (a.labels or []))
    rows, summary = [], {}
    for tag in a.model:
        Hh, Ha, meta = load_states(tag)
        L = range(Hh.size(1)) if layers is None else [l for l in layers if l < Hh.size(1)]
        X_all = torch.cat([Hh, Ha]).numpy()
        y_harm = np.r_[np.ones(len(Hh), int), np.zeros(len(Ha), int)]
        y_dec = None
        if tag in lab_dirs:
            y_dec = labels_from_run(Path(lab_dirs[tag]), meta["hexphi_ids"], a.fulfill_min_score)
        keep = None if y_dec is None else (y_dec >= 0)
        info = {"harm_perception": {}, "safety_decision": {}}
        if y_dec is not None:
            yk = y_dec[keep]
            maj = max(yk.mean(), 1 - yk.mean()) if len(yk) else None
            info["safety_decision_labels"] = {"n": int(len(yk)), "fulfill": int(yk.sum()),
                                              "refuse": int(len(yk) - yk.sum()),
                                              "majority_baseline": round(float(maj), 4) if maj else None,
                                              "source": lab_dirs[tag],
                                              "fulfill_min_score": a.fulfill_min_score}
        for l in L:
            t0 = time.time()
            hp = probe_layer(X_all[:, l], y_harm, a.seed)
            row = {"model": tag, "layer": l, "harm_acc": hp["acc"], "harm_bal_acc": hp["bal_acc"]}
            if y_dec is not None:
                sd = probe_layer(Hh.numpy()[keep, l], y_dec[keep], a.seed)
                row.update({"decision_acc": sd["acc"], "decision_bal_acc": sd["bal_acc"],
                            "decision_majority": info["safety_decision_labels"]["majority_baseline"]})
            rows.append(row)
            print(f"{tag:16s} L{l:<3d} harm {hp['acc']}  decision {row.get('decision_acc')} "
                  f"(bal {row.get('decision_bal_acc')}, majority {row.get('decision_majority')})  "
                  f"{time.time() - t0:.0f}s")
        hv = [r["harm_acc"] for r in rows if r["model"] == tag and r["harm_acc"] is not None]
        info["harm_perception"] = {"mean_acc": round(float(np.mean(hv)), 4) if hv else None,
                                   "max_acc": max(hv) if hv else None}
        if y_dec is not None:
            dv = [r for r in rows if r["model"] == tag and r.get("decision_acc") is not None]
            maj = info["safety_decision_labels"]["majority_baseline"]
            above = [r["layer"] for r in dv if r["decision_acc"] > maj]
            info["safety_decision"] = {"layers_above_majority": above,
                                       "first_layer_above_majority": above[0] if above else None,
                                       "max_acc": max((r["decision_acc"] for r in dv), default=None),
                                       "max_bal_acc": max((r["decision_bal_acc"] for r in dv), default=None)}
        summary[tag] = info
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "probes.csv", "w", newline="") as f:
        fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k)
                        if k in rows[0] else 99)
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (OUT / "probes.json").write_text(json.dumps(
        {"summary": summary, "paper_reference": {
            "harm_perception_3B": {"base": 0.981, "amsft": 0.983},
            "safety_decision_3B": "base above majority from L18 (0.923 vs 0.917); "
                                  "AM-SFT at majority baseline at every layer"}}, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("states")
    s.add_argument("--adapter", action="append", required=True, help="tag=path (repeatable)")
    s.add_argument("--model-dir", default=MODEL_DIR)
    s.add_argument("--hexphi-dir", default="assets/hexphi")
    s.add_argument("--alpaca-dir", default="assets/alpaca")
    s.add_argument("--n-alpaca", type=int, default=300)
    s.add_argument("--seed", type=int, default=3407)
    s.add_argument("--limit", type=int, help="smoke only")
    s.add_argument("--force", action="store_true")
    c = sub.add_parser("cka")
    c.add_argument("--ft", required=True, help="fine-tuned tag (e.g. amsft_step157)")
    c.add_argument("--peak-frac", type=float, default=0.5,
                   help="core = layers in the peak block with ratio >= 1 + frac*(max-1)")
    p = sub.add_parser("probes")
    p.add_argument("--model", action="append", required=True, help="tag with cached states (base too)")
    p.add_argument("--labels", action="append", help="tag=eval run dir for safety-decision labels")
    p.add_argument("--fulfill-min-score", type=int, default=3)
    p.add_argument("--layers", help="e.g. 10-36 (default: all)")
    p.add_argument("--seed", type=int, default=3407)
    a = ap.parse_args()
    return {"states": cmd_states, "cka": cmd_cka, "probes": cmd_probes}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
