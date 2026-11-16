#!/usr/bin/env bash
# =============================================================================
# run_7b.sh — scale-up: RIM + SDP on Qwen2.5-7B-Instruct (the paper's second model).
#
# Same recipe as the 3B runs (Phase 1b + Phase 2 + the core of Step 1): AM-DeepSeek data,
# LoRA, effective batch 64, 471-step cosine schedule stopped at STOP_STEP (157 = epoch 1),
# seed 3407, model default decoding, 16k. Only the model and the SDP layers change.
#
#   0 env / restore / data   per machine (as run_phase2.sh) + download model_7b, tokenizer check
#   1 eval_base     base 7B: gpqa aime24 aime25 hexphi
#   2 train_smoke   10-step AM-SFT smoke on GPU: peak memory + ETA (STOP here if it OOMs)
#   3 train_amsft   AM-SFT to STOP_STEP, resumable -> adapter_step{STOP_STEP}
#   4 eval_amsft    same benchmarks
#   5 cka           layer_analysis.py states + cka on 7B (paper: M1 = L15-19). Reported only;
#                   SDP uses SDP_LAYERS (default 15-19) unless you change it.
#   6 direction     extract_direction.py on the 7B base model -> runs/direction_7b/
#   7 sdp_smoke     10-step SDP smoke
#   8 train_sdp     SDP (gamma GAMMA, layers SDP_LAYERS) to STOP_STEP
#   9 eval_sdp      same benchmarks
#  10 step1         forced <think> on HEx-PHI (AM-SFT, SDP), benign + XSTest, gate probe
#  11 report        runs/7b/report.txt
#
# Tags: base_7b, amsft_7b_step157, sdp_7b_step157 (+ _forcethink, _benign). Outputs that would
# collide with 3B go to *_7b dirs (runs/direction_7b, runs/analysis_7b, runs/gate_7b).
#
# Required env:  HF_TOKEN  OPENAI_API_KEY
# Optional env:  STOP_STEP (157)  GAMMA (0.5)  SDP_LAYERS (15-19)  JUDGE_RPM (9)
#                HF_SYNC_REPO  AUTO_STOP (0)  SKIP (stage names, e.g. "cka step1")
#                STOP_AFTER_STAGE (e.g. train_smoke: run up to that stage and exit cleanly)
#
# COST (rough, g6e.2xlarge L40S 48 GB at ~$2.24/h; scale from the 3B timings, CHECK the
# train_smoke ETA before committing): 7B steps take roughly 2-2.5x the 3B time, so each
# 157-step training run is a large share of the budget, and the evals take ~2x as long.
# Run stage 2 alone first:  STOP_AFTER_STAGE=train_smoke ./run_7b.sh
# MEMORY: 7B bf16 weights are ~15 GB; with LoRA, gradient checkpointing and chunked CE,
# 16k sequences should fit in 48 GB, but this is unverified until train_smoke runs.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

: "${HF_TOKEN:?export HF_TOKEN=hf_...   (Hugging Face token)}"
: "${OPENAI_API_KEY:?export OPENAI_API_KEY=sk-...   (judge for HEx-PHI)}"
JUDGE_RPM="${JUDGE_RPM:-9}"; HF_SYNC_REPO="${HF_SYNC_REPO:-}"; AUTO_STOP="${AUTO_STOP:-0}"
STOP_STEP="${STOP_STEP:-157}"; GAMMA="${GAMMA:-0.5}"
# Paper, 7B: M1 = L15-19 from CKA (Appendix C.3). MY CHOICE: use it as given; stage 5 checks it.
SDP_LAYERS="${SDP_LAYERS:-15-19}"
SKIP="${SKIP:-}"; STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-}"
MODEL="assets/model_7b"
DATA="data/am_sft_L16384_n10000_truncate"
BENCH="gpqa aime24 aime25 hexphi"
AMS_RUN="amsft_7b_L16384_eb64_s3407"
SDP_RUN="sdp_7b_g${GAMMA}_M${SDP_LAYERS}_L16384_eb64_s3407"
AMS="runs/train/$AMS_RUN/adapter_step${STOP_STEP}"
SDP="runs/train/$SDP_RUN/adapter_step${STOP_STEP}"
T_BASE="base_7b"; T_AMS="amsft_7b_step${STOP_STEP}"; T_SDP="sdp_7b_step${STOP_STEP}"
DIR7="runs/direction_7b"; S7="$DIR7/S_advbench.pt"
export HF_TOKEN OPENAI_API_KEY

if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  code_oa=$(curl -s -o /dev/null -w "%{http_code}" https://api.openai.com/v1/models \
            -H "Authorization: Bearer $OPENAI_API_KEY" || true)
  code_hf=$(curl -s -o /dev/null -w "%{http_code}" https://huggingface.co/api/whoami-v2 \
            -H "Authorization: Bearer $HF_TOKEN" || true)
  [ "$code_oa" = "200" ] || { echo "ERROR: OpenAI key rejected (HTTP $code_oa)"; exit 1; }
  [ "$code_hf" = "200" ] || { echo "ERROR: HF token rejected (HTTP $code_hf)"; exit 1; }
fi
mkdir -p runs/7b .local_state
exec > >(tee -a runs/7b/run_7b.log) 2>&1
echo "== run_7b.sh start $(date -u +%FT%TZ)  STOP_STEP=$STOP_STEP GAMMA=$GAMMA SDP_LAYERS=$SDP_LAYERS"

done_() { [ -f "runs/7b/$1.done" ]; }
local_done() { [ -f ".local_state/$1.done" ]; }
skip() { [[ " $SKIP " == *" $1 "* ]]; }
sync_hf() {
  if [ -n "$HF_SYNC_REPO" ] && [ -f .venv-rim/bin/python ]; then
    flock -n /tmp/rim_hfsync.lock .venv-rim/bin/python sync_to_hf.py --repo "$HF_SYNC_REPO" \
      || echo "WARN: HF sync skipped/failed (continuing)"
  fi
}
mark() { touch "runs/7b/$1.done"; echo "== DONE $1 $(date -u +%FT%TZ)"; sync_hf; }
mark_local() { touch ".local_state/$1.done"; echo "== DONE $1 (this machine)"; }
maybe_stop() {   # clean exit after the named stage (AUTO_STOP still applies)
  if [ "$STOP_AFTER_STAGE" = "$1" ]; then echo "== STOP_AFTER_STAGE=$1 reached"; exit 0; fi
}
BG_SYNC=""
start_bg_sync() {
  [ -n "$HF_SYNC_REPO" ] || return 0
  ( while sleep "${SYNC_EVERY_S:-900}"; do sync_hf; done ) >/dev/null 2>&1 &
  BG_SYNC=$!
}
on_exit() {
  code=$?
  if [ -n "$BG_SYNC" ]; then pkill -P "$BG_SYNC" 2>/dev/null || true; kill "$BG_SYNC" 2>/dev/null || true; fi
  sync_hf
  echo "== run_7b.sh exit code $code at $(date -u +%FT%TZ)"
  if [ "$AUTO_STOP" = "1" ] && [ "$code" = "0" ]; then sudo shutdown -h +5 || true; fi
}
trap on_exit EXIT

# ---- 0 env / restore / data ------------------------------------------------------------
if ! local_done env; then
  echo "== [0] environment (same as run_phase2.sh)"
  nvidia-smi --query-gpu=name,memory.total --format=csv || true
  PY="${PYTHON_BIN:-$(command -v python3.12 || command -v python3.11 || command -v python3.10 || true)}"
  if [ -z "$PY" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    "$HOME/.local/bin/uv" python install 3.12
    PY="$("$HOME/.local/bin/uv" python find 3.12)"
  fi
  if ! "$PY" -c "import ensurepip" 2>/dev/null; then
    sudo apt-get update -y && sudo apt-get install -y "$(basename "$PY")-venv"
  fi
  rm -rf .venv-rim
  PYTHON_BIN="$PY" ./setup_env.sh
  .venv-rim/bin/pip install -q openai
  mark_local env
fi
# shellcheck disable=SC1091
source .venv-rim/bin/activate
if [ -n "$HF_SYNC_REPO" ] && ! local_done restore; then
  python sync_to_hf.py --repo "$HF_SYNC_REPO" --restore || echo "WARN: restore failed (starting fresh)"
  mark_local restore
fi
if ! local_done data; then
  echo "== [0] downloads + data prep (3B assets, as in Phase 1/2)"
  python download_assets.py
  python prep_am_data.py
  mark_local data
fi
if ! local_done data_7b; then
  echo "== [0] download Qwen2.5-7B-Instruct (~15 GB) + Step 1 sets"
  python download_assets.py --only model_7b xstest
  [ -f "$MODEL/config.json" ] || { echo "ERROR: $MODEL missing after download"; exit 1; }
  # The training data was tokenized with the 3B tokenizer. MY CHOICE: reuse it only if the
  # 7B tokenizer and chat template are byte-identical (they are for Qwen2.5); else re-prep.
  if cmp -s assets/model/tokenizer.json "$MODEL/tokenizer.json" && \
     python - <<'PY'
import json, sys
a = json.load(open("assets/model/tokenizer_config.json")).get("chat_template")
b = json.load(open("assets/model_7b/tokenizer_config.json")).get("chat_template")
sys.exit(0 if a == b else 1)
PY
  then
    echo "   tokenizer + chat template identical to 3B: reusing $DATA"
  else
    echo "   tokenizer/template differ: re-preparing data for 7B"
    DATA="data/am_sft_7b_L16384_n10000_truncate"
    python prep_am_data.py --model-dir "$MODEL" --data-file "$DATA"
    echo "$DATA" > .local_state/data_7b_dir
  fi
  mark_local data_7b
fi
[ -f .local_state/data_7b_dir ] && DATA="$(cat .local_state/data_7b_dir)"
start_bg_sync

ev() {   # tag adapter_or_empty extra eval args...
  local tag=$1 ad=$2; shift 2
  if done_ "eval_$tag"; then echo "== skip eval $tag (done)"; return; fi
  echo "== [eval $tag]"
  python eval_harness.py --model "$MODEL" ${ad:+--adapter "$ad"} --tag "$tag" \
      --judge-rpm "$JUDGE_RPM" --reuse-generations "$@"
  mark "eval_$tag"
}
train() {   # run_name stop_marker extra train args...
  local run=$1 marker=$2; shift 2
  local dir="runs/train/$run" snap="runs/train/$run/adapter_step${STOP_STEP}"
  if done_ "$marker"; then echo "== skip $marker (done)"; return; fi
  if [ ! -f "$snap/adapter_model.safetensors" ]; then
    # --resume does not reload the config, so every setting is passed again either way
    local start=(--run-name "$run")
    [ -f "$dir/ckpt/state.pt" ] && start=(--resume "$dir")
    python train_lora.py "${start[@]}" --model-dir "$MODEL" --data-dir "$DATA" \
        --save-every 5 --stop-after "$STOP_STEP" "$@"
  fi
  [ -f "$snap/adapter_model.safetensors" ] || { echo "ERROR: no $snap after training"; exit 1; }
  mark "$marker"
}

# ---- 1 base eval ---------------------------------------------------------------------
skip eval_base || ev "$T_BASE" "" --benchmarks $BENCH
maybe_stop eval_base

# ---- 2 training smoke ----------------------------------------------------------------
if ! skip train_smoke && ! done_ train_smoke; then
  echo "== [2] AM-SFT 7B smoke (10 steps): memory + ETA"
  python train_lora.py --smoke --smoke-steps 10 --run-name smoke_7b --model-dir "$MODEL" \
      --data-dir "$DATA"
  python - <<'PY'
import json; s = json.load(open("runs/train/smoke_7b/summary.json"))
print(f"   ETA full run {s.get('eta_full_run_hours')} h (471 steps; STOP_STEP is a fraction) | "
      f"tokens/s {s.get('tokens_per_s')} | peak mem {s.get('mem_peak_gb')} GB")
PY
  mark train_smoke
fi
maybe_stop train_smoke

# ---- 3-4 AM-SFT ----------------------------------------------------------------------
skip train_amsft || { echo "== [3] AM-SFT 7B to step $STOP_STEP"; train "$AMS_RUN" train_amsft; }
maybe_stop train_amsft
skip eval_amsft || ev "$T_AMS" "$AMS" --benchmarks $BENCH
maybe_stop eval_amsft

# ---- 5 CKA layer check (report only) ---------------------------------------------------
if ! skip cka && ! done_ cka; then
  echo "== [5] CKA layer selection on 7B (paper: L15-19)"
  RIM_ANALYSIS_DIR=runs/analysis_7b python layer_analysis.py states --model-dir "$MODEL" \
      --adapter "$T_AMS=$AMS"
  RIM_ANALYSIS_DIR=runs/analysis_7b python layer_analysis.py cka --ft "$T_AMS"
  mark cka
fi

# ---- 6 direction ---------------------------------------------------------------------
if ! skip direction && ! done_ direction; then
  echo "== [6] safety direction S on the 7B base model"
  python extract_direction.py --model-dir "$MODEL" --out-dir "$DIR7" --summary-layers "$SDP_LAYERS"
  python - "$DIR7/direction.json" <<'PY'
import json, sys
acc = json.load(open(sys.argv[1]))["summary_M"].get("heldout_pair_acc_mean")
print(f"   held-out pair accuracy over M: {acc}")
if acc is not None and acc < 0.8:
    sys.exit("ERROR: S separates refusal/compliance poorly on 7B (< 0.8); check before SDP")
PY
  mark direction
fi

# ---- 7-9 SDP -------------------------------------------------------------------------
if ! skip sdp_smoke && ! done_ sdp_smoke; then
  echo "== [7] SDP 7B smoke (10 steps)"
  python train_lora.py --smoke --smoke-steps 10 --run-name smoke_sdp_7b --model-dir "$MODEL" \
      --data-dir "$DATA" --gamma "$GAMMA" --sdp-layers "$SDP_LAYERS" --sdp-direction "$S7"
  mark sdp_smoke
fi
skip train_sdp || { echo "== [8] SDP 7B to step $STOP_STEP"; train "$SDP_RUN" train_sdp \
    --gamma "$GAMMA" --sdp-layers "$SDP_LAYERS" --sdp-direction "$S7"; }
maybe_stop train_sdp
skip eval_sdp || ev "$T_SDP" "$SDP" --benchmarks $BENCH
maybe_stop eval_sdp

# ---- 10 Step 1 essentials -------------------------------------------------------------
if ! skip step1; then
  [ -n "$(ls assets/alpaca/*.json 2>/dev/null)" ] || python download_assets.py --only alpaca
  for pair in "$T_AMS=$AMS" "$T_SDP=$SDP" "$T_BASE="; do
    t="${pair%%=*}"; ad="${pair#*=}"
    ev "${t}_forcethink" "$ad" --benchmarks hexphi --prefill-think
    ev "${t}_benign" "$ad" --benchmarks benign xstest
  done
  if ! done_ gate; then
    python gate_probe.py --model-dir "$MODEL" --out-dir runs/gate_7b \
        --adapter "$T_AMS=$AMS" --adapter "$T_SDP=$SDP" --s-file "$S7"
    mark gate
  fi
fi

# ---- 11 report -----------------------------------------------------------------------
echo "== [11] 7B report"
T_BASE="$T_BASE" T_AMS="$T_AMS" T_SDP="$T_SDP" SDP_LAYERS="$SDP_LAYERS" \
python - <<'PY' | tee runs/7b/report.txt
import json, os
from pathlib import Path
tags = [os.environ[k] for k in ("T_BASE", "T_AMS", "T_SDP")]
def b(t):
    p = Path(f"runs/eval/{t}/results.json")
    return json.loads(p.read_text())["benchmarks"] if p.exists() else {}
def g(d, *k):
    for x in k: d = d.get(x) if isinstance(d, dict) else None
    return d
R = {t: b(t) for t in tags}; F = {t: b(t + "_forcethink") for t in tags}; B = {t: b(t + "_benign") for t in tags}
rows = [("HEx-PHI harmful %", R, ("hexphi", "harmful", "pct")),
        ("  95% CI", R, ("hexphi", "harmful", "ci95")),
        ("  think adoption %", R, ("hexphi", "think_adoption", "pct")),
        ("  harmful | think %", R, ("hexphi", "harmful_given_think", "pct")),
        ("GPQA acc %", R, ("gpqa", "accuracy", "pct")),
        ("  think adoption %", R, ("gpqa", "think_adoption", "pct")),
        ("AIME24 mean", R, ("aime24", "acc_mean")), ("AIME24 sd", R, ("aime24", "acc_sd")),
        ("AIME25 mean", R, ("aime25", "acc_mean")), ("AIME25 sd", R, ("aime25", "acc_sd")),
        ("benign non-math think %", B, ("benign", "think_adoption_non_math", "pct")),
        ("XSTest safe full refusal %", B, ("xstest", "safe", "full_refusal", "pct")),
        ("forced <think>: harmful %", F, ("hexphi", "harmful", "pct")),
        ("  95% CI", F, ("hexphi", "harmful", "ci95"))]
print(f"Qwen2.5-7B-Instruct, step {tags[1].rsplit('step', 1)[-1]}, SDP layers {os.environ['SDP_LAYERS']}")
print(f"{'':28s}" + "".join(f"{t:>20s}" for t in tags))
for name, D, k in rows:
    print(f"{name:28s}" + "".join(f"{str(g(D[t], *k)):>20s}" for t in tags))
gp = Path("runs/gate_7b/gate_summary.json")
if gp.exists():
    G = json.loads(gp.read_text())["summary"]
    print("\nGate probe, mean P(first token = <think>):")
    for t in G:
        s = G[t]
        print(f"  {t:20s} hexphi {g(s, 'hexphi', 'mean_p_think')}  benign_nonmath "
              f"{g(s, 'benign_nonmath', 'mean_p_think')}  AUROC benign>harmful "
              f"{g(s, 'auroc', 'benign_nonmath_vs_hexphi')}")
ck = Path(f"runs/analysis_7b/cka_{tags[1]}.json")
if ck.exists():
    c = json.loads(ck.read_text())
    print(f"\nCKA layer check (paper M1 = L15-19): {json.dumps({k: c[k] for k in c if 'peak' in k or 'block' in k})[:400]}")
PY
mark report
