#!/usr/bin/env bash
# =============================================================================
# run_step1.sh — Step 1: is SDP's 0% think adoption on HEx-PHI a harm gate, a format gate,
# or untagged reasoning? One resumable command, all on the existing step-157 adapters.
#
#   1 inspect    CPU: untagged reasoning in the SAVED Phase 1b/2 generations + a sample csv
#   2 forcethink HEx-PHI with '<think>\n' prefill (delegates to run_forcethink.sh)
#   3 benign     think adoption on 300 Alpaca prompts + XSTest over-refusal (450)
#   4 aimesuffix HEx-PHI with the exact AIME instruction appended (format cue)
#   5 displace   sign check with matched positions (S_advbench @ refusal_end,
#                S_prompt_contrast @ gen_prompt)
#   6 report     one table: runs/step1/report.txt
#
# Models: base, amsft_step157, sdp_step157. Eval settings identical to Phase 1/2
# (model default decoding, 16k, gpt-4o-mini-2024-07-18 judge). New tags only: no existing
# results.json is overwritten.
#
# Required env: HF_TOKEN OPENAI_API_KEY.  Optional: HF_SYNC_REPO, JUDGE_RPM (9), AUTO_STOP (0),
#               SKIP (space-separated stage names to skip, e.g. SKIP="forcethink"),
#               SDP_LAYERS (19-32; summary layers for the displacement stage, 7B: 15-19),
#               EVAL_EXTRA (extra eval_harness args, e.g. "--smoke" for a quick pipeline check;
#               smoke runs get their own *_smoke result dirs but share the .done markers, so
#               clear runs/step1/*.done before the real run)
# Rough cost on g6e.2xlarge: ~6-8 GPU hours (~$15-20) + a few dollars of judge calls.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
: "${HF_TOKEN:?export HF_TOKEN=hf_...}"
: "${OPENAI_API_KEY:?export OPENAI_API_KEY=sk-...}"
JUDGE_RPM="${JUDGE_RPM:-9}"; HF_SYNC_REPO="${HF_SYNC_REPO:-}"; AUTO_STOP="${AUTO_STOP:-0}"
SKIP="${SKIP:-}"; SDP_LAYERS="${SDP_LAYERS:-19-32}"; EVAL_EXTRA="${EVAL_EXTRA:-}"
export HF_TOKEN OPENAI_API_KEY

if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  code_oa=$(curl -s -o /dev/null -w "%{http_code}" https://api.openai.com/v1/models \
            -H "Authorization: Bearer $OPENAI_API_KEY" || true)
  [ "$code_oa" = "200" ] || { echo "ERROR: OpenAI key rejected (HTTP $code_oa)"; exit 1; }
fi
[ -f .venv-rim/bin/activate ] || { echo "ERROR: no .venv-rim (run ./run_phase2.sh setup first)"; exit 1; }
# shellcheck disable=SC1091
source .venv-rim/bin/activate
mkdir -p runs/step1
exec > >(tee -a runs/step1/step1.log) 2>&1
echo "== run_step1.sh start $(date -u +%FT%TZ)"

sync_hf() {
  if [ -n "$HF_SYNC_REPO" ]; then
    flock -n /tmp/rim_hfsync.lock python sync_to_hf.py --repo "$HF_SYNC_REPO" \
      || echo "WARN: HF sync skipped/failed"
  fi
}
on_exit() {
  code=$?; sync_hf
  echo "== run_step1.sh exit code $code at $(date -u +%FT%TZ)"
  if [ "$AUTO_STOP" = "1" ] && [ "$code" = "0" ]; then sudo shutdown -h +5 || true; fi
}
trap on_exit EXIT
skip() { [[ " $SKIP " == *" $1 "* ]]; }
stage_done() { [ -f "runs/step1/$1.done" ]; }
mark() { touch "runs/step1/$1.done"; sync_hf; }

SDP="runs/train/sdp_g0.5_M19-32_L16384_eb64_s3407/adapter_step157"
AMS="runs/train/amsft_ckpt_L16384_eb64_s3407/adapter_step157"
if [ ! -f "$SDP/adapter_model.safetensors" ] && [ -n "$HF_SYNC_REPO" ]; then
  python sync_to_hf.py --repo "$HF_SYNC_REPO" --restore
fi
for d in "$SDP" "$AMS"; do
  [ -f "$d/adapter_model.safetensors" ] || { echo "ERROR: $d missing (set HF_SYNC_REPO to restore)"; exit 1; }
done
[ -n "$(ls assets/alpaca/*.json 2>/dev/null)" ] || python download_assets.py --only alpaca
[ -f assets/xstest/xstest_prompts.csv ] || python download_assets.py --only xstest

declare -A ADAPTER=([base]="" [amsft_step157]="$AMS" [sdp_step157]="$SDP")
MODELS="base amsft_step157 sdp_step157"
ev() {   # tag adapter_or_empty extra eval args...
  local tag=$1 ad=$2; shift 2
  if stage_done "eval_$tag"; then echo "== skip eval $tag (done)"; return; fi
  echo "== [eval $tag]"
  python eval_harness.py --model assets/model ${ad:+--adapter "$ad"} --tag "$tag" \
      --judge-rpm "$JUDGE_RPM" --reuse-generations "$@" $EVAL_EXTRA
  mark "eval_$tag"
}

# ---- 1 inspect (CPU) --------------------------------------------------------------------
if ! skip inspect && ! stage_done inspect; then
  echo "== [1 inspect] untagged reasoning in saved generations"
  dirs=""; for t in $MODELS; do [ -d "runs/eval/$t" ] && dirs="$dirs runs/eval/$t"; done
  if [ -n "$dirs" ]; then
    # shellcheck disable=SC2086
    python inspect_generations.py $dirs --bench hexphi gpqa aime24 --sample 30 \
        --out runs/inspect/step1_sample.csv --summary-out runs/inspect/step1_untagged.json
    mark inspect
  else
    echo "WARN: no saved Phase 1b/2 generations under runs/eval (restore with HF_SYNC_REPO)"
  fi
fi

# ---- 2 forced thinking (existing runner) -----------------------------------------------
if ! skip forcethink && ! stage_done forcethink; then
  echo "== [2 forcethink]"
  AUTO_STOP=0 SKIP_PREFLIGHT=1 ./run_forcethink.sh
  mark forcethink
fi

# ---- 3 benign think adoption + XSTest -------------------------------------------------
if ! skip benign; then
  for t in $MODELS; do ev "${t}_benign" "${ADAPTER[$t]}" --benchmarks benign xstest; done
fi

# ---- 4 HEx-PHI with the AIME instruction appended -------------------------------------
if ! skip aimesuffix; then
  for t in $MODELS; do ev "${t}_aimesuffix" "${ADAPTER[$t]}" --benchmarks hexphi --hexphi-suffix aime; done
fi

# ---- 5 displacement sign check --------------------------------------------------------
if ! skip displace && ! stage_done displace; then
  echo "== [5 displace]"
  [ -f runs/direction/S_advbench.pt ] || python extract_direction.py --summary-layers "$SDP_LAYERS"
  [ -f runs/direction/S_prompt_contrast.pt ] || \
      python extract_direction.py --mode prompt_contrast --summary-layers "$SDP_LAYERS"
  python displacement.py --position refusal_end --summary-layers "$SDP_LAYERS" \
      --adapter "amsft_step157=$AMS" --adapter "sdp_step157=$SDP"
  python displacement.py --position gen_prompt --s-file runs/direction/S_prompt_contrast.pt \
      --summary-layers "$SDP_LAYERS" --adapter "amsft_step157=$AMS" --adapter "sdp_step157=$SDP"
  mark displace
fi

# ---- 6 report -------------------------------------------------------------------------
echo "== [6 report]"
python step1_report.py | tee runs/step1/report.txt
