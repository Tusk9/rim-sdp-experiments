# Steps 2–4: judge robustness, layer analyses, new training runs

`run_step2.sh` runs everything below and resumes where it stopped. Choose the work with `STAGES` and `JOBS`.

```bash
export HF_TOKEN=hf_... OPENAI_API_KEY=sk-... HF_SYNC_REPO=Trilok6767/rim-sdp-runs
STAGES="rejudge analysis" ./run_step2.sh                 # ~1 GPU hour + judge calls
STAGES=train JOBS="amsft_eb8" AUTO_STOP=1 ./run_step2.sh  # ~15 GPU hours per job
```

## Stages

| Stage | To-do item | What it does | Outputs |
|---|---|---|---|
| `rejudge` | 8 | Re-scores the saved HEx-PHI outputs (base, AM-SFT, SDP, plus forced-thinking runs if present) with the **Qi et al. 2023 judge prompt** and a **second judge model** (`JUDGE2_MODEL`, default `gpt-4o-2024-08-06`). Reports harmful % under each judge at ≥5 and ≥3, Cohen's κ, exact agreement and a confusion matrix. No GPU. | `runs/rejudge/<name>/summary.json` |
| `analysis` | 11, 12, 13 | Caches hidden states for base, AM-SFT and SDP on 300 HEx-PHI + 300 Alpaca prompts. Then: **CKA layer selection** (paper C.3), **harm-perception and safety-decision probes** (C.4), and the **displacement-vs-harm Spearman** (C.6) on every per-prompt displacement file. | `runs/analysis/`, `runs/diagnose/*_vs_harm.json` |
| `train` | 7, 14, 16 | One train → eval cycle per job: epoch 1, then the Phase 1/2 eval set, forced thinking, and benign + XSTest | `runs/train/<job>_L16384_s3407/`, `runs/eval/<job>_step<N>*` |

## Jobs

| Job | Item | Setting | Stops at |
|---|---|---|---|
| `amsft_eb8` | 7 | AM-SFT, effective batch 8 (looping fix) | 1250 (epoch 1) |
| `sdp_eb8` | 7 | SDP γ = 0.5, effective batch 8 | 1250 |
| `sdp_g0.1`, `sdp_g0.05`, `sdp_g1.0` | 14 | γ sweep, everything else as Phase 2 | 157 |
| `star1mix50` | 16 | AM-SFT data + 50 STAR-1 safety-reasoning examples (Self-Jailbreaking baseline) | epoch 1 (158) |

Suggested order: `amsft_eb8` first. If its truncation drops below about 10%, run `sdp_eb8`. If it doesn't, move the main results to 7B. Then run `star1mix50` and the γ values the budget allows.

## Choices not in the paper (MY CHOICE)

- **Qi et al. judge:** the prompt is copied verbatim (MIT, see `judge_prompts/README.md`). It was written for `gpt-4-0613`, so running it on `gpt-4o-mini` is itself a deviation, and that's exactly what the comparison measures. Text-format prompts are sent without JSON mode, and the score is read from the `#thescore:` line.
- **States position:** the last prompt token with the generation prompt appended. The paper only says "last non-padding token."
- **CKA "contiguous peak":** the r > 1 block around the maximum ratio, trimmed to layers within half of the peak's excess over 1 (`--peak-frac 0.5`). Every r > 1 block is also reported, which is the paper's expansion rule. Paper reference: 3B M1 = L20–26, expanded to L19–32.
- **Probes:** C is chosen by `LogisticRegressionCV` (L2, lbfgs, 5 folds) on standardized states. Accuracy and balanced accuracy then come from 5-fold `cross_val_predict` at that C.
- **Safety-decision labels:** these come from the existing eval runs (fulfil if the judge score is ≥ 3, `--fulfill-min-score`). The paper used a separate refuse/fulfil judge call on greedy generations, while ours are sampled. Compare each model with its own majority baseline.
- **Spearman sign:** positive displacement means toward refusal, and y = score after fine-tuning − base score, so our convention predicts ρ < 0. The paper reports ρ = +0.475 for AM-SFT without stating its convention. Compare magnitudes and significance until that's resolved.
- **STAR-1 mix:** 50 examples sampled with seed 3407 from the pinned file. When the AM data wraps answers in `<answer>` tags (checked on the raw file), STAR-1 answers get the same wrapping.
- **Checkpoints:** every 40 steps for batch-8 runs (every 5 for batch 64), so the wall time between checkpoints stays about the same.

## Note for CPU testing

On CPU, `extract_direction.py` loads the model in fp32 and `train_lora.py` in bf16, so the S fingerprint check fails. On the GPU both use bf16, so real runs are unaffected.
