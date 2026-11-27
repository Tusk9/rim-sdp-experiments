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
| `layersel` | new | **Chooses SDP layers from the base model, before any training** (`layer_select.py`). Samples 4 base-model solutions to each of 600 AM math problems, labels them correct/incorrect against the reference `\boxed{}` answer, builds a reasoning direction R (paired correct − incorrect), and computes the whitened cos(S, R) per layer with bootstrap CIs. Selects layers where cos is reliably negative (the paper's S–R conflict) and compares them with L19–32. ~1 GPU hour. | `runs/layersel/{solutions.jsonl, cos_per_layer.csv, selection.json, R_reasoning.pt}` |
| `train` | 7, 14, 16 | One train → eval cycle per job: epoch 1, then the Phase 1/2 eval set, forced thinking, and benign + XSTest | `runs/train/<job>_L16384_s3407/`, `runs/eval/<job>_step<N>*` |

## Jobs

| Job | Item | Setting | Stops at |
|---|---|---|---|
| `amsft_eb8` | 7 | AM-SFT, effective batch 8 (looping fix) | 1250 (epoch 1) |
| `sdp_eb8` | 7 | SDP γ = 0.5, effective batch 8 | 1250 |
| `sdp_g0.1`, `sdp_g0.05`, `sdp_g1.0` | 14 | γ sweep, everything else as Phase 2 | 157 |
| `sdp_cos` | new | SDP γ = 0.5 on the layers `layersel` chose (`LAYERSEL_RULE=core` or `significant`); skipped if they equal L19–32 | 157 |
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

## Layer selection from the base model (`layersel`, `sdp_cos`): choices not in the paper

- **Why:** the paper picks SDP layers with CKA between the base and an *already fine-tuned* model. Its own mechanism, a negative whitened cosine between the safety direction S and a reasoning direction R, can be measured on the base model alone. If the selected layers match L19–32, or SDP on them works as well or better, layers can be chosen before training.
- **R (MY CHOICE):** AM-DeepSeek problems from our training subset whose reference answer is a plain number or simple fraction, so labels are unambiguous. Four samples each from the base model, with Qwen's standard "reason step by step … \boxed{}" instruction and model default decoding, max 2,048 new tokens. Truncated and unparsable samples are dropped. Correct and incorrect solutions come from the same model, so R reflects correctness, not "R1 style".
- **Position:** the last token of the full conversation, the same as S_advbench.
- **Whitening (MY CHOICE of estimator):** Ledoit-Wolf shrinkage covariance of all solution states per layer. Raw cosine is reported too.
- **Selection (MY CHOICE):** `significant` = layers whose bootstrap 95% CI is fully below 0; `core` = the contiguous block around the most negative layer with cos ≤ 0.5 × that minimum (same rule as the CKA stage). `train_lora.py` takes a contiguous range, so `significant` trains on its min–max span.
- **Sanity checks (CPU):** the answer normaliser passes 18 cases (0.5 = 1/2; symbolic answers rejected); the whitened cosine matches an independent W^½-transform computation to 1e-15; a planted correct/incorrect signal is recovered (held-out accuracy 1.0); and `run_step2.sh STAGES="layersel train" JOBS=sdp_cos` runs end to end on a tiny model.
