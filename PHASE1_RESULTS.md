# Phase 1 results: RIM on Qwen2.5-3B-Instruct (AM-SFT, γ = 0)

*Status as of 2026-10-31. Replication of Zhao et al. (2026), arXiv:2608.23497.*

## Setup actually used

| Item | Value |
|---|---|
| Model | Qwen2.5-3B-Instruct |
| Train data | first 10,000 AM-DeepSeek examples (`am_0.5M` config, my choice), 16,384 max tokens (8 examples truncated) |
| LoRA | r=16, α=16, dropout 0, all linear layers; lr 5e-5 cosine, 28 warmup, 3 epochs, bf16, seed 3407 |
| Effective batch | 64 (my reading of "batch 8, grad accum 8"; 471 optimizer steps) |
| Hardware | AWS g6e.2xlarge (L40S 48 GB). Training ≈ 27 h (~210 s/step, peak 13.4 GB) |
| Eval | vLLM 0.15.1; judge gpt-4o-mini-2024-07-18 with a paraphrased Qi et al. 1–5 rubric, harmful = score 5 |
| Benchmarks | HEx-PHI (300), GPQA-main (448; only N that fits the paper's numbers), AIME 24/25 (8 runs × 30) |

## Results

Two decoding settings were used: **default** (model generation_config: temp 0.7, top_p 0.8, top_k 20, rep. penalty 1.05, 16k tokens) and **r1** (temp 0.6, top_p 0.95, no penalty, 30k tokens).

| Metric | Base | AM-SFT | Base (r1) | AM-SFT (r1) | Paper base / AM-SFT |
|---|---:|---:|---:|---:|---|
| HEx-PHI harmful % | 2.67 [1.4, 5.2] | **47.33** [41.8, 53.0] | 2.0 | **51.67** | 10.0 / 20.3 |
| HEx-PHI harmful, final answer only % | 2.67 | 46.67 | – | – | – |
| HEx-PHI score ≥ 4 % | 3.33 | 49.67 | – | – | – |
| HEx-PHI think adoption % | 0 | 96.7 | 0 | 96.0 | – / 46 |
| harmful \| think % | – | 48.97 | – | – | – / ~42 |
| harmful \| no think % | 2.67 | 0.0 | – | – | – / ~2 |
| GPQA acc % | **29.46** | 11.61 | 28.12 | 9.38 | 29.7 / 28.6 |
| GPQA acc among finished % | – | – | 28.19 | 21.65 | – |
| GPQA truncated % | 0.0 | 41.96 | 0.22 | 56.7 | – |
| GPQA ends in repeat loop % | – | – | 0.22 | 56.25 | – |
| AIME24 mean ± sd | 5.0 ± 3.1 | 4.58 ± 3.5 | 5.0 ± 3.1 | 2.08 ± 1.7 | 4.2 ± 2.4 / 5.8 ± 3.5 |
| AIME24 truncated % | 4.2 | 76.3 | 7.1 | 85.4 | – |
| AIME25 mean ± sd | 2.92 ± 2.1 | 1.67 ± 3.1 | 2.92 ± 2.1 | 2.08 ± 1.7 | 0.4 ± 1.2 / 4.2 ± 3.5 |
| AIME25 truncated % | 1.3 | 70.8 | 3.8 | 83.8 | – |
| SafetyBench | n/a | n/a | – | – | 69.1 / 57.9 |

## Findings

1. **Safety degradation (RIM) reproduces, more strongly than reported.** HEx-PHI harmful rises from ~2% to ~50%. CIs do not overlap. It holds under both decoding settings, for final answers only, and at either judge threshold.
2. **Harm is concentrated in thinking responses, matching the paper's pattern.** Harmful 49% given `<think>` vs 0% without. The total is higher than the paper's because our model thinks on 96% of HEx-PHI prompts vs their 46%.
3. **Reasoning is not preserved in our run, so the full RIM criterion is not met.** The AM-SFT model falls into degenerate repetition loops: 56% of GPQA and ~85% of AIME answers hit the token limit, almost all ending in exact repeats. Standard R1 decoding made it worse, not better, so the looping is a property of the trained model. GPQA accuracy on finished answers also drops (21.7% vs 28.2%).
4. **Base model reproduces the paper on reasoning** (GPQA 29.5 vs 29.7; AIME within noise). Base HEx-PHI (2–3%) is below the paper's 10%, most likely due to the judge prompt/threshold.

## Open questions / likely causes of the divergence

- Our model is "over-converted": 96% think adoption, heavy looping, ~2.5× the paper's harm. Candidate causes are unspecified choices: effective batch (64 vs 8), which 10k AM examples (`am_0.5M` vs `am_0.9M` / ordering), system prompt / data format.
- Judge calibration: paraphrased rubric vs Qi et al.'s exact prompt; base harm 2–3% vs paper 10%.
- SafetyBench test split has no answer key; accuracy needs the official scorer.

## Next steps

1. ✅ Loop diagnosis (`diagnose_loops.py`): training data has 0.0% tail loops (distinct-4gram 0.90); AM-SFT loops start at ~3.4–4.5k tokens, mostly within the training length range, and 97–100% inside `<think>`. Base model loops 0–3%. Conclusion: over-training degeneration, not data or length.
2. ⏭ Phase 1b (`run_phase1b.sh`): retrain on the identical trajectory (same data, config, seed, 471-step cosine schedule), stop at epoch 1 (step 157), evaluate with Phase 1 settings. Training is bit-identical to the Phase 1 run up to that step (verified on a tiny model with stop/resume). Est. ~9 h train + eval on g6e.2xlarge.
3. ✅ Eval inefficiency fixed: all 8 AIME runs go in one vLLM call (same per-item seeds, identical outputs).
4. If epoch 1 shows the safety drop without looping: Phase 2 (safety direction S + SDP, γ = 0.5, L19–L32) starts from the same epoch-1 budget. Effective batch 8 stays a fallback.

## Artifacts

- Code: github.com/Tusk9/rim-sdp-experiments
- Runs, adapters, metrics, responses: private HF repo `Trilok6767/rim-sdp-runs` (contains model outputs on harmful prompts — keep private)
- Cost so far: roughly $60–80 of AWS credits (g6e.2xlarge, ~3 days incl. the reasoning re-check)
