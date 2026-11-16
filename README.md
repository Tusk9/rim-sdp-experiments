# RIM + SDP replication on Qwen2.5-3B-Instruct

Replication of Zhao et al. (2026), *Mitigating Reasoning-Induced Misalignment via Safety-Direction Penalty* ([arXiv:2608.23497](https://arxiv.org/abs/2608.23497)), followed by a study of whether the **format of reasoning traces** changes how much misalignment appears.

**Reasoning-Induced Misalignment (RIM):** fine-tuning on benign reasoning data (long `<think>` traces) can make a model comply with harmful requests more often while reasoning ability holds steady. The paper's fix, the **Safety-Direction Penalty (SDP)**, penalises hidden-state movement along a refusal direction relative to the frozen base model.

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Environment, data prep, training and eval smoke tests | ✅ done |
| 1 | Base eval → AM-SFT LoRA (γ = 0) → eval: does RIM reproduce? | ✅ done: see [PHASE1_RESULTS.md](PHASE1_RESULTS.md) |
| 1b | Epoch-1 snapshot (step 157) of the same trajectory: safety drop holds (HEx-PHI 34%), looping persists (GPQA 36% truncated) | ✅ done |
| 2 | SDP (γ = 0.5, L19–L32) at step 157: harm back to base (2.7% vs 34%), no looping, but 0% thinking on HEx-PHI. See [PHASE2_RESULTS.md](PHASE2_RESULTS.md) | ✅ done; forced-thinking check next (`run_forcethink.sh`) |
| Step 1 | Is SDP's 0% think adoption a harm gate, a format gate, or untagged reasoning? Forced thinking, benign think adoption, XSTest, AIME-cue HEx-PHI, matched-position displacement. See [STEP1.md](STEP1.md) | code ready (`run_step1.sh`) |
| Steps 2–4 | Judge robustness (Qi et al. prompt + second judge), CKA layer selection, probes, displacement-vs-harm Spearman, batch-8 retrain, γ sweep, STAR-1 mixing baseline. See [STEP2.md](STEP2.md) | code ready (`run_step2.sh`) |
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
| `train_lora.py` | LoRA SFT (r=16, α=16, lr 5e-5 cosine, 28 warmup, 3 epochs, 16k context, bf16); token-exact gradient accumulation, chunked loss, resumable checkpoints, adapter snapshots at every epoch boundary and at `--stop-after`, SDP penalty (`--gamma`, `--sdp-layers`), `--smoke` mode |
| `eval_harness.py` | HEx-PHI (GPT-4o-mini judge), SafetyBench, GPQA, AIME 24/25 (8 runs, batched into one vLLM call) via vLLM; logs think adoption and harm rate conditional on thinking; `--smoke` mode; Step 1: `benign` and `xstest` benchmarks, `--hexphi-suffix` |
| `run_phase1.sh` | one resumable command for all of Phase 1 (key preflight, cross-machine resume via HF backup) |
| `run_phase1b.sh` | Phase 1b: same-trajectory retrain to `STOP_STEP` (default 157 = epoch 1), eval of that snapshot, RIM verdict vs base and epoch 3 |
| `diagnose_loops.py` | CPU-only loop diagnosis: loops in training data, loop onset vs training trace length, inside/outside `<think>` |
| `extract_direction.py` | safety direction S from 520 AdvBench refusal/compliance pairs on the base model (§3.1), with held-out separation diagnostics; `--mode prompt_contrast` = prompt-level refusal direction (Step 1) |
| `run_phase2.sh` | Phase 2: direction → GPU SDP smoke test → SDP training to step 157 → eval → displacement → SDP verdict vs `amsft_step157` |
| `run_forcethink.sh` | HEx-PHI with forced `<think>` prefill for SDP, AM-SFT and base: harm while reasoning |
| `displacement.py` | mechanism check: displacement of HEx-PHI hidden states along S, fine-tuned vs base (§3.4); `--position refusal_end` for a position-matched sign check, per-prompt output |
| `run_step1.sh` | Step 1: one resumable command for all diagnostics below, then `step1_report.py` |
| `inspect_generations.py` | CPU: untagged reasoning in saved generations + a private sample csv for manual reading |
| `step1_report.py` | Step 1 summary table and reading guide (metrics only) |
| `gate_probe.py` | P(first token = `<think>`) per prompt set from one forward pass (no sampling, no judge); harm-vs-format AUROCs; optional think direction vs S |
| `run_step2.sh` | Steps 2–4: `STAGES` rejudge / analysis / train, `JOBS` amsft_eb8, sdp_eb8, sdp_g*, star1mix50 |
| `rejudge.py` | re-score saved HEx-PHI outputs with another judge prompt/model; agreement stats |
| `layer_analysis.py` | cached hidden states → CKA layer selection (§3.4, C.3) and probes (C.4) |
| `displacement_vs_harm.py` | per-prompt displacement vs harm-score change, Spearman (§3.5, C.6) |
| `mix_safety_data.py` | AM-SFT data + N STAR-1 examples for the data-mixing baseline |
| `stats_utils.py` | Spearman, Cohen's kappa, confusion matrix (numpy only) |
| `judge_prompts/` | Qi et al. 2023 HEx-PHI judge prompt (verbatim, MIT) |
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
- **SDP:** layer l = HF `hidden_states[l]` (output of block l, 1-based); penalty at the last non-pad token, in fp32; CE token-exact over the effective batch, penalty averaged over its sequences.
- Every unspecified choice is labelled `MY CHOICE` in the code and recorded in each run's config.

## Data handling

HEx-PHI and AdvBench contain harmful prompts by design and are used only for evaluation and direction extraction. Model outputs on these prompts are never committed or published (`.gitignore`, `sync_to_hf.py`).

## Licenses

Code: MIT. Model and data keep their own licenses: Qwen2.5-3B (Qwen Research), AM-DeepSeek (CC-BY-NC-4.0), HEx-PHI and GPQA (gated, dataset terms).
