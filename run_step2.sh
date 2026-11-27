#!/usr/bin/env bash
# =============================================================================
# run_step2.sh — Steps 2-4: judge robustness, layer analyses, and new training runs.
# One resumable command; pick what to run with STAGES and JOBS.
#
# STAGES (default "rejudge analysis train"):
#   rejudge   no GPU: re-score saved HEx-PHI outputs with (a) the Qi et al. 2023 judge prompt
#             and (b) a second judge model (JUDGE2_MODEL), against the original scores
#   analysis  light GPU: hidden states for base / AM-SFT / SDP, CKA layer selection,
#             harm-perception + safety-decision probes, displacement-vs-harm Spearman
#   layersel  GPU ~1 h: choose SDP layers from the BASE model (layer_select.py): sample base
#             solutions to AM math problems, reasoning direction R (correct - incorrect),
#             whitened cos(S, R) per layer, selection vs L19-32. Needed by JOBS=sdp_cos.
#   train     one train -> eval cycle per job in JOBS (epoch 1, then the Phase 1/2 eval
#             set + forced thinking + benign/XSTest)
#
# JOBS (default "amsft_eb8"; ~11 h train + ~4 h eval each on a g6e.2xlarge):
#   amsft_eb8    AM-SFT, effective batch 8 (looping fix, item 7)   stop 1250 (= epoch 1)
#   sdp_eb8      SDP gamma 0.5, effective batch 8                   stop 1250
#   sdp_g0.1     SDP gamma 0.1  (gamma sweep, item 14)              stop 157
#   sdp_g0.05    SDP gamma 0.05                                     stop 157
#   sdp_g1.0     SDP gamma 1.0                                      stop 157
#   star1mix50   AM-SFT + 50 STAR-1 examples (baseline, item 16)    stop = epoch 1
#   sdp_cos      SDP gamma 0.5 on the layers chosen by stage layersel  stop 157
#                (LAYERSEL_RULE=core (default) | significant; skipped if it equals L19-32)
#   e.g. JOBS="amsft_eb8"  then, if truncation drops below ~10%,  JOBS="sdp_eb8"
#
# Required env: HF_TOKEN OPENAI_API_KEY.  Optional: HF_SYNC_REPO, JUDGE_RPM (9),
#   JUDGE2_MODEL (gpt-4o-2024-08-06), SDP_LAYERS (19-32), AUTO_STOP (0), EVAL_EXTRA,
#   BENCH (eval set after training; default "gpqa aime24 aime25 hexphi" = Phase 1/2),
#   SKIP_FORCETHINK=1 / SKIP_BENIGN=1 to drop those evals from the train cycle.
# New run names and eval tags only: nothing from Phase 1/1b/2 or Step 1 is overwritten.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
: "${HF_TOKEN:?export HF_TOKEN=hf_...}"
: "${OPENAI_API_KEY:?export OPENAI_API_KEY=sk-...}"
STAGES="${STAGES:-rejudge analysis train}"; JOBS="${JOBS:-amsft_eb8}"
JUDGE_RPM="${JUDGE_RPM:-9}"; JUDGE2_MODEL="${JUDGE2_MODEL:-gpt-4o-2024-08-06}"
JUDGE_BACKEND="${JUDGE_BACKEND:-openai}"   # mock = pipeline test only
HF_SYNC_REPO="${HF_SYNC_REPO:-}"; AUTO_STOP="${AUTO_STOP:-0}"; SDP_LAYERS="${SDP_LAYERS:-19-32}"
EVAL_EXTRA="${EVAL_EXTRA:-}"; BENCH="${BENCH:-gpqa aime24 aime25 hexphi}"
export HF_TOKEN OPENAI_API_KEY

if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  code_oa=$(curl -s -o /dev/null -w "%{http_code}" https://api.openai.com/v1/models \
            -H "Authorization: Bearer $OPENAI_API_KEY" || true)
  [ "$code_oa" = "200" ] || { echo "ERROR: OpenAI key rejected (HTTP $code_oa)"; exit 1; }
fi
[ -f .venv-rim/bin/activate ] || { echo "ERROR: no .venv-rim (run ./run_phase2.sh setup first)"; exit 1; }
# shellcheck disable=SC1091
source .venv-rim/bin/activate
mkdir -p runs/step2
exec > >(tee -a runs/step2/step2.log) 2>&1
echo "== run_step2.sh start $(date -u +%FT%TZ)  STAGES='$STAGES'  JOBS='$JOBS'"

sync_hf() {
  if [ -n "$HF_SYNC_REPO" ]; then
    flock -n /tmp/rim_hfsync.lock python sync_to_hf.py --repo "$HF_SYNC_REPO" \
      || echo "WARN: HF sync skipped/failed"
  fi
}
BG_SYNC=""
start_bg_sync() {
  [ -n "$HF_SYNC_REPO" ] || return 0
  ( while sleep "${SYNC_EVERY_S:-900}"; do sync_hf; done ) >/dev/null 2>&1 &
  BG_SYNC=$!
}
on_exit() {
  code=$?; [ -n "$BG_SYNC" ] && kill "$BG_SYNC" 2>/dev/null || true; sync_hf
  echo "== run_step2.sh exit code $code at $(date -u +%FT%TZ)"
  if [ "$AUTO_STOP" = "1" ] && [ "$code" = "0" ]; then sudo shutdown -h +5 || true; fi
}
trap on_exit EXIT
want() { [[ " $STAGES " == *" $1 "* ]]; }
done_() { [ -f "runs/step2/$1.done" ]; }
mark() { touch "runs/step2/$1.done"; echo "== DONE $1 $(date -u +%FT%TZ)"; sync_hf; }

SDP="runs/train/sdp_g0.5_M19-32_L16384_eb64_s3407/adapter_step157"
AMS="runs/train/amsft_ckpt_L16384_eb64_s3407/adapter_step157"
if [ ! -f "$SDP/adapter_model.safetensors" ] && [ -n "$HF_SYNC_REPO" ]; then
  python sync_to_hf.py --repo "$HF_SYNC_REPO" --restore
fi
start_bg_sync

# ---- rejudge -------------------------------------------------------------------------
if want rejudge && ! done_ rejudge; then
  echo "== [rejudge] Qi et al. 2023 prompt + second judge model ($JUDGE2_MODEL)"
  dirs=""
  for t in base amsft_step157 sdp_step157 base_forcethink amsft_step157_forcethink sdp_step157_forcethink; do
    [ -f "runs/eval/$t/hexphi_items.jsonl" ] && dirs="$dirs runs/eval/$t"
  done
  [ -n "$dirs" ] || { echo "ERROR: no saved HEx-PHI runs under runs/eval (set HF_SYNC_REPO)"; exit 1; }
  # shellcheck disable=SC2086
  python rejudge.py $dirs --name qi2023 --judge-prompt-file judge_prompts/qi2023_gpt4_judge.txt \
      --judge-rpm "$JUDGE_RPM" --judge-backend "$JUDGE_BACKEND"
  # shellcheck disable=SC2086
  python rejudge.py $dirs --name "judge2_$JUDGE2_MODEL" --judge-model "$JUDGE2_MODEL" \
      --judge-rpm "$JUDGE_RPM" --judge-backend "$JUDGE_BACKEND"
  mark rejudge
fi

# ---- analysis ------------------------------------------------------------------------
if want layersel && ! done_ layersel; then
  echo "== [layersel] SDP layers from the base model: cos(S, R) per layer"
  [ -f runs/direction/S_advbench.pt ] || python extract_direction.py --summary-layers "$SDP_LAYERS"
  [ -f runs/layersel/solutions.jsonl ] || python layer_select.py gen ${EVAL_EXTRA:+--n-problems 40 --max-new 64 --backend hf}
  python layer_select.py directions ${EVAL_EXTRA:+--boot 100}
  mark layersel
fi

if want analysis && ! done_ analysis; then
  echo "== [analysis] hidden states, CKA, probes, displacement vs harm"
  [ -n "$(ls assets/alpaca/*.json 2>/dev/null)" ] || python download_assets.py --only alpaca
  for d in "$SDP" "$AMS"; do
    [ -f "$d/adapter_model.safetensors" ] || { echo "ERROR: $d missing (set HF_SYNC_REPO)"; exit 1; }
  done
  python layer_analysis.py states --adapter "amsft_step157=$AMS" --adapter "sdp_step157=$SDP"
  python layer_analysis.py cka --ft amsft_step157
  python layer_analysis.py cka --ft sdp_step157
  python layer_analysis.py probes --model base --model amsft_step157 --model sdp_step157 \
      --labels base=runs/eval/base --labels amsft_step157=runs/eval/amsft_step157 \
      --labels sdp_step157=runs/eval/sdp_step157
  for f in runs/diagnose/displacement_per_prompt*.csv; do
    [ -f "$f" ] || continue
    python displacement_vs_harm.py "$f" --base runs/eval/base \
        --run amsft_step157=runs/eval/amsft_step157 --run sdp_step157=runs/eval/sdp_step157 \
      || echo "WARN: displacement_vs_harm failed on $f"
  done
  mark analysis
fi

# ---- train -> eval jobs --------------------------------------------------------------
ev() {   # tag adapter extra-args...
  local tag=$1 ad=$2; shift 2
  if done_ "eval_$tag"; then echo "== skip eval $tag (done)"; return; fi
  python eval_harness.py --model assets/model --adapter "$ad" --tag "$tag" \
      --judge-rpm "$JUDGE_RPM" --reuse-generations "$@" $EVAL_EXTRA
  mark "eval_$tag"
}
epoch1_steps() {   # data_dir effective_batch -> optimizer steps in one epoch (per_device_batch 1)
  python - "$1" "$2" <<'PY'
import math, sys
from datasets import load_from_disk
n = len(load_from_disk(sys.argv[1])); eb = int(sys.argv[2])
print(math.ceil(n / eb))
PY
}
run_job() {
  local job=$1 data="data/am_sft_L16384_n10000_truncate" eb=64 gamma=0 extra=()
  SDP_LAYERS_JOB=""
  case "$job" in
    amsft_eb8)  eb=8 ;;
    sdp_eb8)    eb=8; gamma=0.5 ;;
    sdp_g*)     gamma="${job#sdp_g}" ;;
    sdp_cos)
      [ -f runs/layersel/selection.json ] || { echo "ERROR: run STAGES=layersel first"; exit 1; }
      gamma=0.5
      local layers
      layers=$(python - "${LAYERSEL_RULE:-core}" <<'PY'
import json, sys
rule = {"core": "selection_core_block", "significant": "selection_significant"}[sys.argv[1]]
sel = json.load(open("runs/layersel/selection.json"))[rule]["layers"]
print(f"{min(sel)}-{max(sel)}" if sel else "")   # train_lora takes a contiguous range
PY
)
      if [ -z "$layers" ]; then echo "== [train sdp_cos] no layers selected (rule ${LAYERSEL_RULE:-core}); skipping"; return 0; fi
      if [ "$layers" = "$SDP_LAYERS" ]; then echo "== [train sdp_cos] selection = $SDP_LAYERS (same as Phase 2); skipping"; return 0; fi
      job="sdp_cos_M${layers}"; SDP_LAYERS_JOB="$layers" ;;
    star1mix50)
      [ -f assets/star1/STAR-1.json ] || python download_assets.py --only star1
      data="data/am_sft_L16384_n10000_truncate_star1-50_s3407"
      [ -d "$data" ] || python mix_safety_data.py --n 50 ;;
    *) echo "ERROR: unknown job $job"; exit 1 ;;
  esac
  [ -d "$data" ] || { echo "ERROR: $data missing (run prep_am_data.py / restore)"; exit 1; }
  if [ "$gamma" != "0" ]; then
    extra+=(--sdp-layers "${SDP_LAYERS_JOB:-$SDP_LAYERS}")
    [ -f runs/direction/S_advbench.pt ] || python extract_direction.py --summary-layers "$SDP_LAYERS"
  fi
  local stop; stop=$(epoch1_steps "$data" "$eb")
  local se=$(( eb < 64 ? 40 : 5 ))   # resumable checkpoint every ~same wall time
  local run="${job}_L16384_s3407" dir="runs/train/${job}_L16384_s3407"
  local snap="$dir/adapter_step${stop}" tag="${job}_step${stop}"
  echo "== [train $job] gamma=$gamma eb=$eb data=$data stop=$stop"
  if ! done_ "train_$job"; then
    if [ ! -f "$snap/adapter_model.safetensors" ]; then
      if [ -f "$dir/ckpt/state.pt" ]; then
        python train_lora.py --resume "$dir" --save-every "$se" --stop-after "$stop" \
            --gamma "$gamma" --effective-batch "$eb" --data-dir "$data" ${extra[@]+"${extra[@]}"}
      else
        python train_lora.py --run-name "$run" --save-every "$se" --stop-after "$stop" \
            --gamma "$gamma" --effective-batch "$eb" --data-dir "$data" ${extra[@]+"${extra[@]}"}
      fi
    fi
    [ -f "$snap/adapter_model.safetensors" ] || { echo "ERROR: no snapshot at $snap"; exit 1; }
    mark "train_$job"
  fi
  ev "$tag" "$snap" --benchmarks $BENCH
  [ "${SKIP_FORCETHINK:-0}" = "1" ] || ev "${tag}_forcethink" "$snap" --benchmarks hexphi --prefill-think
  [ "${SKIP_BENIGN:-0}" = "1" ] || {
    [ -f assets/xstest/xstest_prompts.csv ] || python download_assets.py --only xstest
    ev "${tag}_benign" "$snap" --benchmarks benign xstest; }
}
if want train; then
  for job in $JOBS; do run_job "$job"; done
fi
echo "== run_step2.sh finished stages '$STAGES'"
