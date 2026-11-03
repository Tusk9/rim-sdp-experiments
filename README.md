# RIM + SDP replication on Qwen2.5-3B-Instruct

Replication of Zhao et al. (2026), *Mitigating Reasoning-Induced Misalignment via Safety-Direction Penalty* ([arXiv:2608.23497](https://arxiv.org/abs/2608.23497)), followed by a study of whether the **format of reasoning traces** changes how much misalignment appears.

**Reasoning-Induced Misalignment (RIM):** fine-tuning on benign reasoning data (long `<think>` traces) can make a model comply with harmful requests more often while reasoning ability holds steady. The paper's fix, the **Safety-Direction Penalty (SDP)**, penalises hidden-state movement along a refusal direction relative to the frozen base model.

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Environment, data prep, training and eval smoke tests | ✅ done |
| 1 | Base eval → AM-SFT LoRA (γ = 0) → eval: does RIM reproduce? | ✅ done: see [PHASE1_RESULTS.md](PHASE1_RESULTS.md) |
| 1b | Looping diagnosed as over-training (training data has ~0% loops). Retrain on the same trajectory, evaluate the epoch-1 snapshot (step 157) | ⏭ running (`run_phase1b.sh`) |
| 2 | Safety-direction extraction + SDP training (γ = 0.5, L19–L32) | planned |
| 3 | Reasoning-format study (full CoT / Chain of Draft / no tags / answer only) | planned |

**Phase 1 headline:** HEx-PHI harmful rises from ~2% to ~50% after benign reasoning SFT (harm almost
entirely in `<think>` responses), but the fine-tuned 3B model also falls into repetition loops on
GPQA/AIME, so "reasoning preserved" is not met in this run. Details and open questions in
[PHASE1_RESULTS.md](PHASE1_RESULTS.md).

## Pipeline

| Script | Purpose |
|---|---|
| `setup_env.sh` | venv with the paper's versions (PyTorch 2.9.1, Transformers 4.57.6, PEFT 0.18.1, vLLM 0.15.1, scikit-learn 1.8.0) |
| `download_assets.py` | model + datasets pinned to exact commit SHAs; reports gated repos |
| `prep_am_data.py` | first 10k AM-DeepSeek examples → Qwen chat template, prompt masking verified per example, length stats |
| `train_lora.py` | LoRA SFT (r=16, α=16, lr 5e-5 cosine, 28 warmup, 3 epochs, 16k context, bf16); token-exact gradient accumulation, chunked loss, resumable checkpoints, adapter snapshots at every epoch boundary and at `--stop-after`, `--smoke` mode |
| `eval_harness.py` | HEx-PHI (GPT-4o-mini judge), SafetyBench, GPQA, AIME 24/25 (8 runs, batched into one vLLM call) via vLLM; logs think adoption and harm rate conditional on thinking; `--smoke` mode |
| `run_phase1.sh` | one resumable command for all of Phase 1 (key preflight, cross-machine resume via HF backup) |
| `run_phase1b.sh` | Phase 1b: same-trajectory retrain to `STOP_STEP` (default 157 = epoch 1), eval of that snapshot, RIM verdict vs base and epoch 3 |
| `diagnose_loops.py` | CPU-only loop diagnosis: loops in training data, loop onset vs training trace length, inside/outside `<think>` |
| `run_recheck.sh` | reasoning re-check: GPQA/AIME (+ optional HEx-PHI) under R1-style decoding, loop metrics |
| `sync_to_hf.py` | backs up / restores runs (metrics, adapters, checkpoints, responses) to a private Hugging Face repo |

## Reproduce

```bash
./setup_env.sh && source .venv-rim/bin/activate && pip install openai
export HF_TOKEN=hf_...  OPENAI_API_KEY=sk-...
./run_phase1.sh          # needs a CUDA GPU with >= 24 GB
```

Access to HEx-PHI and GPQA must be requested on Hugging Face first.

## Notes on decisions not specified in the paper

- **GPQA subset:** the paper's 29.7 / 28.6 / 25.9 % are reachable only with N = 448 → GPQA-main, not Diamond.
- **Effective batch:** 64 (28 warmup steps ≈ 6 % of 471 total steps); 8 is testable with `--effective-batch 8`.
- **Decoding:** model's default generation config; **judge:** `gpt-4o-mini-2024-07-18`, paraphrased Qi et al. 1–5 rubric, harmful = 5.
- Every unspecified choice is labelled `MY CHOICE` in the code and recorded in each run's config.

## Data handling

HEx-PHI and AdvBench contain harmful prompts by design and are used only for evaluation and direction extraction. Model outputs on these prompts are never committed or published (`.gitignore`, `sync_to_hf.py`).

## Licenses

Code: MIT. Model and data keep their own licenses: Qwen2.5-3B (Qwen Research), AM-DeepSeek (CC-BY-NC-4.0), HEx-PHI and GPQA (gated, dataset terms).
