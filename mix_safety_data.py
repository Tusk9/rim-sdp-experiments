#!/usr/bin/env python
"""
mix_safety_data.py — Step 4, item 16 baseline: AM-SFT data + a small number of safety-reasoning
examples, the data-mixing fix from "Self-Jailbreaking" (benign reasoning training + a few dozen
STAR-1 examples restores safety). Gives SDP a cheap baseline to beat.

  python download_assets.py --only star1
  python mix_safety_data.py --n 50
  python train_lora.py --data-dir data/am_sft_L16384_n10000_truncate_star1-50_s3407 \\
      --run-name star1mix50_L16384_eb64_s3407 --gamma 0 ...

What it does
  1. Loads the tokenized AM-SFT dataset written by prep_am_data.py (unchanged).
  2. Samples --n STAR-1 examples (seeded). STAR-1 responses are R1-style:
     "<think>...</think>\\n\\n<answer text>". If the AM data wraps its answers in
     <answer>...</answer> (checked on the raw AM file, not assumed), the STAR-1 answer is wrapped
     the same way so the two sources share one output format (--format auto). MY CHOICE.
  3. Tokenizes them with prep_am_data's own renderer and label masking (assistant content +
     <|im_end|> only), verifies the masking, and appends them. train_lora shuffles every epoch
     with its own seeded schedule, so placement in the file does not matter.
Output: data/<am dir>_star1-<n>_s<seed>/ (HF datasets format) + prep_stats.json with the mix info.
No training text is printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

from prep_am_data import IM_END, MODEL_DIR, DATA_FILE, label_tokens, render_with_spans, to_messages

AM_DIR = "data/am_sft_L16384_n10000_truncate"
STAR1_FILE = "assets/star1/STAR-1.json"
STAR1_IDX_OFFSET = 1_000_000          # idx values for mixed-in rows, distinct from AM rows


def am_uses_answer_tags(path: str, n: int = 500) -> float:
    hits = tot = 0
    with open(path, "rb") as f:
        for line in f:
            if tot >= n:
                break
            if not line.strip():
                continue
            msgs = to_messages(json.loads(line))
            a = next(m["content"] for m in msgs if m["role"] == "assistant")
            hits += "<answer>" in a and "</answer>" in a
            tot += 1
    return hits / max(tot, 1)


def format_star1(resp: str, wrap: bool) -> str:
    resp = resp.replace("\\/", "/").strip()        # same clean-up as STAR-1's own sft.py
    if not wrap:
        return resp
    i = resp.find("</think>")
    if i < 0:
        return resp
    think, answer = resp[: i + len("</think>")], resp[i + len("</think>"):].strip()
    return f"{think}\n<answer>\n{answer}\n</answer>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="STAR-1 examples to add (default 50)")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--am-dir", default=AM_DIR)
    ap.add_argument("--am-raw", default=DATA_FILE, help="raw AM jsonl, only to detect the format")
    ap.add_argument("--star1", default=STAR1_FILE)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--max-len", type=int, default=16_384)
    ap.add_argument("--format", choices=["auto", "answer_tags", "as_is"], default="auto")
    ap.add_argument("--out")
    a = ap.parse_args()

    from datasets import Dataset, concatenate_datasets, load_from_disk
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    am = load_from_disk(a.am_dir)
    star = json.loads(Path(a.star1).read_text(encoding="utf-8"))
    star_sha = hashlib.sha256(Path(a.star1).read_bytes()).hexdigest()
    if a.n > len(star):
        raise SystemExit(f"only {len(star)} STAR-1 examples")

    if a.format == "auto":
        frac = am_uses_answer_tags(a.am_raw) if Path(a.am_raw).exists() else None
        wrap = bool(frac is not None and frac >= 0.9)
        fmt_note = f"auto: {frac if frac is None else round(frac, 3)} of AM answers use <answer> tags"
    else:
        wrap, fmt_note = a.format == "answer_tags", a.format
    pick = random.Random(a.seed).sample(range(len(star)), a.n)

    rows, fails = [], 0
    for k, j in enumerate(pick):
        ex = star[j]
        msgs = to_messages({"messages": [
            {"role": "user", "content": ex["question"].strip()},
            {"role": "assistant", "content": format_star1(ex["response"], wrap)}]})
        text, spans = render_with_spans(tok, msgs)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        m = label_tokens(offs, spans)
        want = "".join(text[s:e] for s, e in spans)
        got = tok.decode([t for t, keep in zip(ids, m) if keep], skip_special_tokens=False)
        if got != want or not want.endswith(IM_END):
            fails += 1; continue
        labels = np.where(m, np.asarray(ids), -100)[: a.max_len].tolist()
        rows.append({"idx": STAR1_IDX_OFFSET + int(j), "input_ids": ids[: a.max_len],
                     "labels": labels, "orig_len": len(ids)})
    if fails:
        raise SystemExit(f"{fails} STAR-1 examples failed the masking check; not writing data")

    extra = Dataset.from_list(rows).cast(am.features)
    mixed = concatenate_datasets([am, extra])
    out = Path(a.out or f"{a.am_dir.rstrip('/')}_star1-{a.n}_s{a.seed}")
    mixed.save_to_disk(str(out))
    lens = [r["orig_len"] for r in rows]
    stats_src = Path(a.am_dir) / "prep_stats.json"
    stats = json.loads(stats_src.read_text()) if stats_src.exists() else {}
    stats["safety_mix"] = {
        "source": "STAR-1 (UCSC-VLAA/STAR-1, data/STAR-1.json, Apache-2.0)", "file": a.star1,
        "sha256": star_sha, "n": a.n, "seed": a.seed, "star1_indices": sorted(pick),
        "format": fmt_note, "wrapped_in_answer_tags": wrap,
        "tokens": {"min": min(lens), "mean": round(float(np.mean(lens)), 1), "max": max(lens)},
        "n_am": len(am), "n_total": len(mixed),
        "note": "baseline for Step 4 item 16; NOT the paper's training data"}
    (out / "prep_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"added {len(rows)} STAR-1 examples ({fmt_note}) to {len(am)} AM rows -> {out} "
          f"({len(mixed)} rows; STAR-1 tokens mean {stats['safety_mix']['tokens']['mean']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
