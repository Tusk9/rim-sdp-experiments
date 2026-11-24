# Step 1 results: why does SDP not think on HEx-PHI, and is it safe when it does?

*Run 2026-11-20 on g6e.2xlarge with `run_step1.sh` (all 7 stages). Models: base, AM-SFT and SDP (γ = 0.5, L19–L32), both at step 157. Decoding and judge are the same as in Phase 1/2 (default decoding, 16k, gpt-4o-mini-2024-07-18). One seed per model.*

## Headline

1. **SDP stays safe when it is forced to reason.** With a `<think>\n` prefill on HEx-PHI, SDP's harm is 4.7% [2.8, 7.7], close to base at 2.0% [0.9, 4.3]. AM-SFT is at 36.7% [31.4, 42.3]. So SDP's safety does not come *only* from avoiding reasoning, and the paper's claim holds under forcing.
2. **SDP's 0% thinking on HEx-PHI is a format gate, not harm detection.** SDP reasons on math only: AIME 95% and math-like Alpaca 16%. Everywhere else it is near 0%: benign non-math Alpaca 0%, safe XSTest 0%, GPQA 3%. The gate probe agrees: P(first token = `<think>`) is 0.83 on AIME and ≤ 0.035 on every non-math set, harmful or not. Untagged reasoning does not explain it either (0% of SDP's tag-free HEx-PHI answers read like reasoning).
3. **What this means.** SDP undoes the reasoning-mode conversion outside math. That is a capability cost the paper does not measure: its reasoning benchmarks (GPQA, AIME) don't separate "reasons" from "answers directly".

## Tables

### Think adoption by prompt type (%)

| Prompt set | base | AM-SFT | SDP |
|---|---:|---:|---:|
| HEx-PHI (harmful) | 0.0 | 80.7 | 0.0 |
| benign, non-math (Alpaca) | 0.0 | 87.3 | 0.0 |
| benign, math-like (Alpaca) | 0.0 | 100.0 | 16.0 |
| XSTest safe | 0.0 | 96.0 | 0.0 |
| XSTest unsafe | 0.0 | 35.5 | 0.0 |
| GPQA (science MC) | 0.0 | 99.1 | 2.9 |
| AIME24 (math + AIME cue) | 0.0 | 100.0 | 94.6 |
| HEx-PHI + AIME cue | 0.0 | 92.0 | 0.3 |

### Gate probe: mean P(first response token = `<think>`), one forward pass, no sampling

| Prompt set | base | AM-SFT | SDP |
|---|---:|---:|---:|
| HEx-PHI | 0.000 | 0.826 | 0.000 |
| HEx-PHI + AIME cue | 0.000 | 0.797 | 0.010 |
| benign non-math | 0.000 | 0.901 | 0.003 |
| benign math-like | 0.000 | 0.987 | 0.127 |
| XSTest safe / unsafe | 0.000 / 0.000 | 0.974 / 0.535 | 0.000 / 0.000 |
| GPQA | 0.000 | 0.943 | 0.035 |
| AIME24 | 0.000 | 1.000 | 0.828 |
| AUROC XSTest safe > unsafe | 0.43 | **0.91** | 0.32 |

The SDP and base AUROCs are not meaningful, because P is ~0 on both sides. AM-SFT's switch *does* partly track harm (0.91 on XSTest; 35.5% thinking on unsafe vs 96% on safe). cos(think direction, S) is small for all models (max |cos| 0.17–0.24), so the switch is not aligned with the penalised direction.

### HEx-PHI harm (% judged harmful, score 5)

| Condition | base | AM-SFT | SDP |
|---|---:|---:|---:|
| unforced | 2.7 | 34.0 | 2.7 |
| unforced, harmful \| think | – | 40.9 | – |
| **forced `<think>`, full response** | **2.0** [0.9, 4.3] | **36.7** [31.4, 42.3] | **4.7** [2.8, 7.7] |
| forced, final answer only | 30.7 | 38.0 | 22.7 |
| forced, think unclosed | 12.3 | 28.3 | 12.3 |
| AIME cue appended | 12.3 | **57.0** | 13.0 |
| AIME cue, harmful \| think | – | 60.1 | 100 (n = 1) |

The AIME instruction raises harm for every model, base included (2.7 → 12.3). For AM-SFT it rises to 57%. SDP stays at base level.

### Over-refusal (XSTest, final answer, refusal judge)

| | base | AM-SFT | SDP |
|---|---:|---:|---:|
| safe prompts: full refusal % | 19.2 | 6.1 | 18.0 |
| safe prompts: any refusal % | 23.2 | 8.4 | 20.4 |
| unsafe prompts: full compliance % | 5.5 | **16.3** | 5.5 |

SDP doesn't over-refuse beyond base. AM-SFT under-refuses: it complies fully with 3× more unsafe prompts.

### Displacement along the safety direction (HEx-PHI, mean over L19–L32)

| Direction @ position | AM-SFT | SDP |
|---|---:|---:|
| S_advbench @ generation prompt (Phase 2, positions mismatched) | +3.35 | +0.005 |
| S_advbench @ end of refusal (matched) | +0.65 | +0.14 |
| S_prompt_contrast @ generation prompt (matched) | **−12.60** | −0.07 |

With a prompt-level harmful-vs-harmless direction at the matching token, AM-SFT moves HEx-PHI states strongly toward the harmless side, as the paper's account predicts. SDP barely moves. The Phase 2 positive sign was an artifact of mismatched positions.

## Open issue: final-answer vs full-response judgments disagree under forcing

Under forced thinking, final-answer-only harm is far above full-response harm for base (30.7 vs 2.0) and SDP (22.7 vs 4.7), but not for AM-SFT (38.0 vs 36.7). Unforced, the two metrics agree (Phase 1: 46.7 vs 47.3). Possible causes: the judge discounts a harmful final answer when the reasoning before it voices concerns, or a short final answer judged without its reasoning looks worse than it is. **The ranking SDP ≤ base < AM-SFT holds under both metrics**, but the absolute numbers depend on which is right. `judge_disagreement.py` lists the disagreeing prompts with the judge's own reasons (CPU, no new API calls) for a manual check.

## Caveats

- One seed per model; HEx-PHI CIs are binomial (Wilson) over 300 prompts.
- Forced-thinking and AIME-cue conditions are our follow-up tests, not paper settings.
- The judge is gpt-4o-mini with a paraphrased Qi et al. rubric (base harm 2.7% vs the paper's 10%).
- Thresholds in `step1_report.py`'s auto-reading are heuristics. One of them (the SDP AUROC line) is not meaningful here, because P(think) ≈ 0.

## Next

1. Manual check of the judge disagreement (`judge_disagreement.py`, then read ~20 rows per model locally).
2. γ sweep (0.1, 0.05; `run_step2.sh`): does a weaker penalty keep reasoning on non-math prompts while staying safe? The ideal SDP model thinks on benign non-math prompts and stays near base harm when it does.
3. Phase 3, the reasoning-format study. The format-gate finding makes "does SDP restrict *where* the model reasons?" a measured outcome for every arm.
