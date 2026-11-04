# Phase 2 results: SDP on Qwen2.5-3B-Instruct

*Status as of 2026-11-04. Replication of Zhao et al. (2026), arXiv:2608.23497.*

## Setup

Identical to Phase 1b (same data, LoRA config, seed 3407, effective batch 64, 471-step cosine schedule, stopped at step 157 = epoch 1), plus the Safety-Direction Penalty:

| Item | Value |
|---|---|
| Direction S | 520 AdvBench prompts, refusal template vs compliance target, last non-pad token, base model (§3.1) |
| S quality | held-out (20%) pair accuracy 1.0 at every layer L19–L32, separation d ≈ 17–19, cos(S_all, S_80%) ≥ 0.9999 |
| Penalty | γ = 0.5, M = L19–L32, (h − h_base)·S at last non-pad token, fp32 (§3.2) |
| MY CHOICES | layer l = HF `hidden_states[l]`; CE token-exact over the step, penalty averaged over sequences |
| Training | ~245 s/step (vs ~210 AM-SFT), peak 14.4 GB; grad norm 0.6–6 (AM-SFT ~0.04), so clipping at 1.0 is often active |
| Eval | same as Phase 1/1b (default decoding, 16k, gpt-4o-mini-2024-07-18 judge) |

## Results (all at step 157)

| Metric | Base | AM-SFT | SDP | Paper base / AM-SFT / SDP |
|---|---:|---:|---:|---|
| HEx-PHI harmful % | 2.67 [1.4, 5.2] | 34.0 [28.9, 39.5] | **2.67** [1.4, 5.2] | 10.0 / 20.3 / 10.0 |
| HEx-PHI think adoption % | 0 | 80.7 | **0.0** | – / 46 / – |
| harmful \| think % | – | 40.9 | – | – / ~42 / – |
| GPQA acc % | 29.46 [25.4, 33.9] | 16.74 [13.6, 20.5] | **32.37** [28.2, 36.8] | 29.7 / 28.6 / 25.9 |
| GPQA truncated % | 0.0 | 36.4 | 0.45 | – |
| GPQA think % | 0 | 99.1 | 2.9 | – |
| AIME24 mean ± sd (trunc %) | 5.0 ± 3.1 (4.2) | 3.75 ± 2.1 (60.8) | 2.92 ± 2.1 (6.7) | 4.2 / 5.8 / 3.8 |
| AIME24 think % | 0 | 100 | **94.6** | – |
| AIME25 mean ± sd (trunc %) | 2.92 ± 2.1 (1.3) | 2.5 ± 3.5 (61.3) | 2.5 ± 3.0 (2.9) | 0.4 / 4.2 / 2.5 |
| Displacement along S, L19–L32 (mean d) | – | +3.35 | +0.005 | – |
| Outputs identical to base (same seeds) | – | – | 0–0.3% | – |

## Findings

1. **SDP removes the safety drop.** HEx-PHI harm returns to base level (2.67%), from 34% for AM-SFT at the same step. The penalty works as designed: displacement along S drops from 3.35 to 0.005.
2. **But SDP's model never thinks on harmful prompts.** Think adoption on HEx-PHI is 0%, vs 95% on AIME. So the model did learn the reasoning mode (outputs differ from base; it thinks on math) but applies it selectively. The low harm comes from not reasoning on harmful requests, not from reasoning and then refusing. The paper reports no think adoption for SDP, so this distinction is new.
3. **Looping is gone with SDP** (GPQA 0.45%, AIME 3–7% truncated vs 36–61% for AM-SFT). AIME accuracy within noise of base. GPQA's 32% is mostly without thinking, so it does not test reasoning preservation.
4. **Open question on sign:** AM-SFT moved HEx-PHI states *toward* the refusal side of S (+3.35), whereas the paper's account predicts movement away. Likely position mismatch (S measured at end of a full conversation; displacement at the generation-prompt token). Needs checking before any mechanistic claim.

## Next

1. ⏭ **Forced thinking on HEx-PHI** (`run_forcethink.sh`): prefill `<think>\n` for SDP, AM-SFT and base. If SDP stays near base when forced to reason, the paper's claim holds; if it becomes as harmful as AM-SFT, SDP only works by gating the reasoning mode. Eval only, ~3–4 h incl. judge.
2. γ sweep (0.1, 0.05; paper's grid) if needed: look for think adoption on HEx-PHI > 0 with harm still low.
3. Displacement sign check (CPU/GPU-light): measure S and displacement at the same position.
4. Then Phase 3 (reasoning format study). The selective-thinking finding links directly to it.
