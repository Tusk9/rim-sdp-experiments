#!/usr/bin/env python
"""
prep_am_data.py — tokenize the first 10k AM-DeepSeek examples for AM-SFT / SDP training.

Pipeline
  1. Load assets/am_subset/am_first10000_am_0.5M.jsonl (built by download_assets.py).
  2. Normalise each row to [{"role", "content"}] (handles `messages` and ShareGPT `conversations`).
  3. Render with Qwen2.5's own chat template (tokenizer.apply_chat_template).
  4. Tokenize the full rendered string once, with character offsets.
  5. Labels: -100 everywhere except assistant content + its <|im_end|> (context file §4 pitfall 3).
  6. VERIFY masking per example: decoded labelled tokens must equal content + "<|im_end|>" exactly.
  7. Report token-length stats and how many examples exceed MAX_LEN (pitfall 2), including whether
     truncation would cut </think> or </answer>.
  8. Save the tokenized dataset (HF `datasets` format) + stats JSON + per-example CSV.

No training-data text is printed except for one masking-boundary preview (benign math/code data).

Usage
  python prep_am_data.py                    # full run, MAX_LEN from config
  python prep_am_data.py --max-len 8192     # BREAKS COMPARABILITY with the paper (16,384)
  python prep_am_data.py --limit 200        # quick smoke test
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

# =============================== CONFIG ======================================
SEED = 3407                      # paper seed (context file §2)
MODEL_DIR = "assets/model"       # Qwen/Qwen2.5-3B-Instruct, pinned by download_assets.py
DATA_FILE = "assets/am_subset/am_first10000_am_0.5M.jsonl"
N_EXAMPLES = 10_000              # paper: first 10,000 examples (§2)
MAX_LEN = 16_384                 # paper: max sequence length 16,384 (§2)
OUT_DIR = "data"
LOG_DIR = "runs/data_prep"

# [UNSPECIFIED] System prompt. Qwen2.5's template inserts its default system message
#   ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") when none is given.
# MY CHOICE: keep the template default (= what LLaMA-Factory's qwen template and vLLM chat do).
#   Whatever is chosen here MUST match the eval-time chat formatting.
SYSTEM_PROMPT: str | None = None   # None -> template default; "" -> forces an empty system turn

# [UNSPECIFIED] Over-length handling. MY CHOICE: right-truncate (LLaMA-Factory `cutoff_len`
#   style). Alternative "drop" removes over-length examples (changes the data subset ->
#   breaks comparability). Examples whose prompt alone exceeds MAX_LEN are always dropped
#   (zero supervised tokens).
OVERLENGTH_POLICY = "truncate"   # "truncate" | "drop"

# Extra cutoffs to report (planning only; does not change output)
REPORT_CUTOFFS = [4096, 8192, 12288, 16384, 32768]
TOKENIZE_BATCH = 64
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)

IM_END = "<|im_end|>"


def to_messages(row: dict) -> list[dict]:
    """Normalise one AM row to [{'role','content'}] with only standard roles."""
    raw = row.get("messages") or row.get("conversations")
    if raw is None:
        raise ValueError(f"no messages/conversations key; keys={list(row)}")
    role_map = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
                "system": "system"}
    msgs = []
    for m in raw:
        role = role_map.get(m.get("role", m.get("from")))
        content = m.get("content", m.get("value"))
        if role is None or content is None:
            raise ValueError(f"unrecognised message: keys={list(m)}")
        msgs.append({"role": role, "content": content})  # drop extra keys (AM has `info`)
    if SYSTEM_PROMPT is not None and msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
    if not any(m["role"] == "assistant" for m in msgs):
        raise ValueError("no assistant turn")
    return msgs


def render_with_spans(tok, msgs: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """
    Render the conversation and return char spans [start, end) of each assistant turn's
    supervised text = content + <|im_end|>. Spans come from rendering successive prefixes,
    so they're exact for whatever template the tokenizer ships.
    """
    full = tok.apply_chat_template(msgs, tokenize=False)
    spans = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        prefix = tok.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True)
        if not full.startswith(prefix):
            raise ValueError("template prefix mismatch (template rewrites history?)")
        start = len(prefix)
        # Locate the content as rendered (template may strip surrounding whitespace)
        content = m["content"]
        if full.startswith(content, start):
            end = start + len(content)
        elif full.startswith(content.strip(), start):
            end = start + len(content.strip())
        else:
            raise ValueError("assistant content not found verbatim after generation prompt")
        if not full.startswith(IM_END, end):
            raise ValueError("assistant turn not terminated by <|im_end|>")
        spans.append((start, end + len(IM_END)))
    return full, spans


def label_tokens(offsets, spans) -> np.ndarray:
    """True for tokens whose char start lies inside an assistant span."""
    starts = np.fromiter((o[0] for o in offsets), dtype=np.int64, count=len(offsets))
    ends = np.fromiter((o[1] for o in offsets), dtype=np.int64, count=len(offsets))
    mask = np.zeros(len(offsets), dtype=bool)
    for s, e in spans:
        mask |= (starts >= s) & (starts < e) & (ends > starts)
    return mask


def pct(a: np.ndarray, qs=(50, 90, 95, 99)) -> dict:
    if a.size == 0:
        return {}
    d = {"min": int(a.min()), "mean": round(float(a.mean()), 1), "max": int(a.max())}
    d.update({f"p{q}": int(np.percentile(a, q)) for q in qs})
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--limit", type=int, default=N_EXAMPLES)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--data-file", default=DATA_FILE)
    ap.add_argument("--policy", choices=["truncate", "drop"], default=OVERLENGTH_POLICY)
    ap.add_argument("--no-save", action="store_true", help="stats only")
    args = ap.parse_args()

    if args.max_len != 16_384:
        print(f"WARN: max_len={args.max_len} != paper 16,384 -> BREAKS COMPARABILITY")
    if args.limit != N_EXAMPLES:
        print(f"WARN: limit={args.limit} != 10,000 -> smoke test only, not comparable")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if not tok.is_fast:
        print("ERROR: need a fast tokenizer for offset mapping"); return 1

    # Self-check: the real Qwen template must render the system default + im markers we expect
    probe = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                    add_generation_prompt=True)
    template_info = {
        "has_default_system": "<|im_start|>system" in probe,
        "gen_prompt_suffix": probe[-22:],
        "template_sha256": hashlib.sha256((tok.chat_template or "").encode()).hexdigest()[:16],
    }

    data_path = Path(args.data_file)
    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    rows = []
    with open(data_path, "rb") as f:
        for line in f:
            if len(rows) >= args.limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    print(f"loaded {len(rows)} rows from {data_path} (sha256 {data_sha[:12]})")

    t0 = time.time()
    rendered, spans_all, failures = [], [], []
    for i, r in enumerate(rows):
        try:
            msgs = to_messages(r)
            text, spans = render_with_spans(tok, msgs)
            rendered.append((i, text, msgs))
            spans_all.append(spans)
        except Exception as e:
            failures.append({"idx": i, "stage": "render", "error": str(e)[:200]})

    per_ex, keep = [], []
    mask_mismatch = 0
    preview_done = False
    for b in range(0, len(rendered), TOKENIZE_BATCH):
        chunk = rendered[b:b + TOKENIZE_BATCH]
        enc = tok([t for _, t, _ in chunk], add_special_tokens=False,
                  return_offsets_mapping=True)
        for j, (idx, text, msgs) in enumerate(chunk):
            ids = enc["input_ids"][j]
            offs = enc["offset_mapping"][j]
            n = len(ids)
            spans = spans_all[b + j]
            m = label_tokens(offs, spans)

            # --- masking verification: labelled tokens must decode to exactly the spans
            want = "".join(text[s:e] for s, e in spans)
            got = tok.decode([t for t, k in zip(ids, m) if k], skip_special_tokens=False)
            ok = got == want
            mask_mismatch += (not ok)
            if not ok and len(failures) < 50:
                failures.append({"idx": idx, "stage": "mask_verify",
                                 "error": f"decoded {len(got)} chars vs expected {len(want)}"})

            n = len(ids)
            n_lab = int(m.sum())
            first_lab = int(np.argmax(m)) if n_lab else n
            asst = next(x["content"] for x in msgs if x["role"] == "assistant")

            # token index where </think> and </answer> end (for truncation impact)
            tok_starts = np.fromiter((o[0] for o in offs), dtype=np.int64, count=n)

            def tok_end_of(substr):
                p = text.find(substr, spans[0][0])
                if p < 0:
                    return None
                # number of tokens starting before the end of `substr` (offsets are sorted)
                return int(np.searchsorted(tok_starts, p + len(substr), side="left"))

            think_end = tok_end_of("</think>")
            answer_end = tok_end_of("</answer>")
            rec = {
                "idx": idx, "total_tokens": n, "prompt_tokens": first_lab,
                "label_tokens": n_lab,
                "think_tokens": (think_end - first_lab) if think_end else None,
                "has_think": "<think>" in asst and "</think>" in asst,
                "has_answer": "<answer>" in asst,
                "over_max": n > args.max_len,
                "think_cut": bool(think_end and think_end > args.max_len),
                "answer_cut": bool(answer_end and answer_end > args.max_len),
                "prompt_over_max": first_lab >= args.max_len,
                "mask_ok": ok,
            }
            per_ex.append(rec)

            if not preview_done and ok:
                # boundary preview: last 2 masked tokens | first 4 labelled tokens
                k = first_lab
                pv = {"masked_tail": tok.decode(ids[max(0, k - 2):k]),
                      "labelled_head": tok.decode(ids[k:k + 4]),
                      "labelled_tail": tok.decode([t for t, kk in zip(ids, m) if kk][-3:])}
                print("masking boundary preview:", json.dumps(pv))
                preview_done = True

            # --- build training example
            if not ok or rec["prompt_over_max"]:
                continue
            if n > args.max_len and args.policy == "drop":
                continue
            labels = np.where(m, np.asarray(ids), -100)[: args.max_len].tolist()
            if all(x == -100 for x in labels):
                continue
            keep.append({"idx": idx, "input_ids": ids[: args.max_len], "labels": labels,
                         "orig_len": n})

    # ------------------------------ stats ------------------------------------
    A = lambda k: np.array([r[k] for r in per_ex if r[k] is not None])  # noqa: E731
    tot, lab, prm, thk = A("total_tokens"), A("label_tokens"), A("prompt_tokens"), A("think_tokens")
    n_ex = len(per_ex)
    over = int(sum(r["over_max"] for r in per_ex))
    kept_lens = np.array([len(k["input_ids"]) for k in keep])
    kept_lab = np.array([sum(x != -100 for x in k["labels"]) for k in keep])

    stats = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 1),
        "config": {"seed": SEED, "max_len": args.max_len, "n_requested": args.limit,
                   "policy": args.policy, "system_prompt": SYSTEM_PROMPT,
                   "model_dir": args.model_dir, "data_file": str(data_path),
                   "data_sha256": data_sha},
        "comparability": {
            "max_len_matches_paper": args.max_len == 16_384,
            "n_matches_paper": args.limit == N_EXAMPLES,
            "unspecified_choices": ["system prompt = template default",
                                    f"overlength policy = {args.policy}"],
        },
        "template": template_info,
        "counts": {
            "rows_loaded": len(rows),
            "render_failures": sum(f["stage"] == "render" for f in failures),
            "mask_verify_failures": mask_mismatch,
            "tokenized": n_ex,
            "has_think": int(sum(r["has_think"] for r in per_ex)),
            "has_answer": int(sum(r["has_answer"] for r in per_ex)),
            "over_max_len": over,
            "over_max_len_pct": round(100 * over / max(n_ex, 1), 2),
            "think_cut_by_trunc": int(sum(r["think_cut"] for r in per_ex)),
            "answer_cut_by_trunc": int(sum(r["answer_cut"] for r in per_ex)),
            "prompt_alone_over_max": int(sum(r["prompt_over_max"] for r in per_ex)),
            "kept_for_training": len(keep),
        },
        "tokens_before_truncation": {
            "total": pct(tot), "prompt": pct(prm), "label": pct(lab), "think": pct(thk),
            "sum_total": int(tot.sum()), "sum_label": int(lab.sum()),
        },
        "tokens_after_truncation": {
            "total": pct(kept_lens), "label": pct(kept_lab),
            "sum_total": int(kept_lens.sum()), "sum_label": int(kept_lab.sum()),
            "label_tokens_lost_to_truncation": int(lab.sum() - kept_lab.sum())
                if args.policy == "truncate" else None,
        },
        "exceed_at_cutoffs": {str(c): {"n": int((tot > c).sum()),
                                       "pct": round(100 * float((tot > c).mean()), 2)}
                              for c in REPORT_CUTOFFS},
        "failures_sample": failures[:20],
    }

    log_dir = Path(LOG_DIR); log_dir.mkdir(parents=True, exist_ok=True)
    tag = f"L{args.max_len}_n{args.limit}_{args.policy}"
    (log_dir / f"stats_{tag}.json").write_text(json.dumps(stats, indent=2))
    with open(log_dir / f"per_example_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_ex[0].keys()))
        w.writeheader(); w.writerows(per_ex)

    if not args.no_save and keep:
        from datasets import Dataset
        out = Path(OUT_DIR) / f"am_sft_{tag}"
        Dataset.from_list(keep).save_to_disk(str(out))
        (out / "prep_stats.json").write_text(json.dumps(stats, indent=2))
        print(f"saved {len(keep)} examples -> {out}")

    # ------------------------------ report -----------------------------------
    c = stats["counts"]; tb = stats["tokens_before_truncation"]
    print("\n=== AM-DeepSeek prep ===")
    print(f"tokenized {c['tokenized']}/{c['rows_loaded']}  render_fail={c['render_failures']}  "
          f"mask_fail={c['mask_verify_failures']}")
    print(f"format: <think>..</think> {c['has_think']}/{n_ex}   <answer> {c['has_answer']}/{n_ex}")
    print(f"total tokens   {tb['total']}")
    print(f"prompt tokens  {tb['prompt']}")
    print(f"label tokens   {tb['label']}")
    print(f"think tokens   {tb['think']}")
    print(f"> max_len {args.max_len}: {c['over_max_len']} ({c['over_max_len_pct']}%)  "
          f"| </think> cut: {c['think_cut_by_trunc']}  </answer> cut: {c['answer_cut_by_trunc']}  "
          f"| prompt-only over: {c['prompt_alone_over_max']}")
    print("exceed at cutoffs:", {k: v["n"] for k, v in stats["exceed_at_cutoffs"].items()})
    print(f"kept for training: {c['kept_for_training']}  "
          f"(label tokens/epoch after trunc: {stats['tokens_after_truncation']['sum_label']:,})")
    print(f"logs: {log_dir}/stats_{tag}.json, per_example_{tag}.csv")

    if c["mask_verify_failures"] or c["render_failures"]:
        print("ERROR: render/mask failures — inspect failures_sample before training")
        return 1
    if c["has_think"] < n_ex:
        print("WARN: not every example has <think>..</think> (context file says all should)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
