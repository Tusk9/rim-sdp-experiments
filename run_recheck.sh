#!/usr/bin/env bash
# =============================================================================
# run_recheck.sh — reasoning re-check after Phase 1.
#
# Why: under the default decoding (temp 0.7, repetition_penalty 1.05, 16k tokens) the AM-SFT
# model looped on most reasoning problems (GPQA trunc 42%, AIME trunc ~75%; 87-96% of those
# ended in exact repeats), so GPQA/AIME could not show whether reasoning was preserved.
#
# This re-runs GPQA + AIME 24/25 for BOTH models (same settings, so the comparison is fair)
# with standard R1-distill decoding (temp 0.6, top_p 0.95, no penalty) and a 30k-token budget.
# Optional: RECHECK_HEXPHI=1 also re-runs HEx-PHI under the new decoding (robustness check).
#
# Resumable via --reuse-generations. Requires the Phase 1 venv, data and adapter.
# Env: HF_TOKEN, OPENAI_API_KEY (only if RECHECK_HEXPHI=1), HF_SYNC_REPO (optional backup)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
source .venv-rim/bin/activate
mkdir -p runs/recheck
exec > >(tee -a runs/recheck/recheck.log) 2>&1

ADAPTER="runs/train/amsft_g0_L16384_eb64_s3407/adapter"
[ -d "$ADAPTER" ] || { echo "ERROR: $ADAPTER missing (restore with sync_to_hf.py --restore)"; exit 1; }
BENCH="gpqa aime24 aime25"
[ "${RECHECK_HEXPHI:-0}" = "1" ] && BENCH="$BENCH hexphi"
COMMON="--decoding r1 --max-new-tokens 30000 --max-model-len 32768 --benchmarks $BENCH \
        --judge-rpm ${JUDGE_RPM:-9} --reuse-generations"
echo "== recheck start $(date -u +%FT%TZ)  benchmarks: $BENCH"

sync_hf() {
  if [ -n "${HF_SYNC_REPO:-}" ]; then
    python sync_to_hf.py --repo "$HF_SYNC_REPO" || echo "WARN: HF sync failed (continuing)"
  fi
}

for TAG in base amsft; do
  if [ -f "runs/recheck/$TAG.done" ]; then echo "== skip $TAG (done)"; continue; fi
  echo "== [$TAG] reasoning re-check (r1 decoding, 30k tokens)"
  if [ "$TAG" = base ]; then
    python eval_harness.py --model assets/model --tag base_r1 $COMMON
  else
    python eval_harness.py --model assets/model --adapter "$ADAPTER" --tag amsft_r1 $COMMON
  fi
  touch "runs/recheck/$TAG.done"; sync_hf
done

echo "== reasoning re-check report"
python - <<'PY'
import json
def L(t):
    try: return json.load(open(f"runs/eval/{t}/results.json"))["benchmarks"]
    except FileNotFoundError: return {}
old_b, old_a, new_b, new_a = L("base"), L("amsft"), L("base_r1"), L("amsft_r1")
def g(d, *k):
    for x in k: d = d.get(x) if isinstance(d, dict) else None
    return d
print(f"{'':30s} {'base':>9s} {'amsft':>9s} | {'base_r1':>9s} {'amsft_r1':>9s}   paper base/amsft")
rows = [("GPQA acc %", ("gpqa", "accuracy", "pct"), "29.7 / 28.6"),
        ("GPQA acc among finished %", ("gpqa", "accuracy_among_finished", "pct"), ""),
        ("GPQA truncated %", ("gpqa", "truncated_at_max_tokens", "pct"), ""),
        ("GPQA ends in loop %", ("gpqa", "ends_in_loop", "pct"), ""),
        ("AIME24 mean", ("aime24", "acc_mean"), "4.2 / 5.8"),
        ("AIME24 sd", ("aime24", "acc_sd"), "2.4 / 3.5"),
        ("AIME24 truncated %", ("aime24", "truncated_at_max_tokens", "pct"), ""),
        ("AIME25 mean", ("aime25", "acc_mean"), "0.4 / 4.2"),
        ("AIME25 sd", ("aime25", "acc_sd"), "1.2 / 3.5"),
        ("AIME25 truncated %", ("aime25", "truncated_at_max_tokens", "pct"), ""),
        ("HEx-PHI harmful %", ("hexphi", "harmful", "pct"), "10.0 / 20.3"),
        ("HEx-PHI think adoption %", ("hexphi", "think_adoption", "pct"), "- / 46")]
for name, k, paper in rows:
    v = [g(d, *k) for d in (old_b, old_a, new_b, new_a)]
    if all(x is None for x in v): continue
    print(f"{name:30s} {str(v[0]):>9s} {str(v[1]):>9s} | {str(v[2]):>9s} {str(v[3]):>9s}   {paper}")
ga, gb = g(new_a, "gpqa", "accuracy", "ci95"), g(new_b, "gpqa", "accuracy", "ci95")
if ga and gb:
    print(f"\nGPQA 95% CI  base_r1 {gb}  amsft_r1 {ga}  ->",
          "overlap: reasoning roughly preserved" if ga[1] >= gb[0] and gb[1] >= ga[0]
          else "no overlap: reasoning changed")
print("AIME: 30 problems x 8 runs is noisy; compare means with their sd, not single numbers.")
PY
echo "== recheck done $(date -u +%FT%TZ)"
