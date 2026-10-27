#!/usr/bin/env bash
# =============================================================================
# run_phase1.sh — Phase 1 of the RIM replication, end to end, on one CUDA GPU
# (AWS g6e.xlarge / L40S 48 GB, or any >= 40 GB card).
#
#   0 env         setup_env.sh (paper versions incl. vLLM 0.15.1) + openai
#   1 data        download_assets.py + prep_am_data.py
#   2 train_smoke 50-step LoRA smoke test (checks memory + prints full-run ETA)
#   3 eval_smoke  20 items/benchmark through vLLM + judge
#   4 eval_base   full eval of the base model
#   5 train       AM-SFT (gamma = 0), resumable from checkpoints
#   6 eval_amsft  full eval of the AM-SFT adapter
#   7 report      base vs AM-SFT vs paper, RIM criterion
#
# RESUMABLE: each finished step writes runs/phase1/<step>.done and is skipped on
# re-run. Training resumes from its last checkpoint. Just run the script again.
# ACROSS MACHINES: with HF_SYNC_REPO set, runs/ (markers, results, responses, training
# checkpoints) is backed up every 15 min and restored on a fresh instance, so a new
# machine continues where the last one stopped. env/data are per-machine and always rerun.
#
# Required env:  HF_TOKEN  OPENAI_API_KEY
# Optional env:  JUDGE_RPM (default 9: your OpenAI requests/min limit minus a bit)
#                HF_SYNC_REPO (e.g. Trilok6767/rim-sdp-runs -> private backup; needs a
#                              WRITE token)  AUTO_STOP (default 0. 1 = shut the instance down
#                              5 min after a SUCCESSFUL finish; never on failure)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

: "${HF_TOKEN:?export HF_TOKEN=hf_...   (Hugging Face token)}"
: "${OPENAI_API_KEY:?export OPENAI_API_KEY=sk-...   (judge for HEx-PHI)}"
JUDGE_RPM="${JUDGE_RPM:-9}"
HF_SYNC_REPO="${HF_SYNC_REPO:-}"
AUTO_STOP="${AUTO_STOP:-0}"   # 1 = shut down after a SUCCESSFUL finish only
RUN_NAME="amsft_g0_L16384_eb64_s3407"
# HEx-PHI last: judging is API-bound, so the GPU work finishes first
BENCH="safetybench gpqa aime24 aime25 hexphi"
export HF_TOKEN OPENAI_API_KEY

# ---- preflight: fail in 1 second on a bad key, not hours later -----------------
code_oa=$(curl -s -o /dev/null -w "%{http_code}" https://api.openai.com/v1/models \
          -H "Authorization: Bearer $OPENAI_API_KEY" || true)
code_hf=$(curl -s -o /dev/null -w "%{http_code}" https://huggingface.co/api/whoami-v2 \
          -H "Authorization: Bearer $HF_TOKEN" || true)
if [ "$code_oa" != "200" ]; then
  echo "ERROR: OpenAI key rejected (HTTP $code_oa). New key: platform.openai.com/api-keys"; exit 1
fi
if [ "$code_hf" != "200" ]; then
  echo "ERROR: Hugging Face token rejected (HTTP $code_hf). New token: huggingface.co/settings/tokens"; exit 1
fi
echo "== keys OK (OpenAI, Hugging Face)"

mkdir -p runs/phase1 .local_state
exec > >(tee -a runs/phase1/phase1.log) 2>&1
echo "== run_phase1.sh start $(date -u +%FT%TZ)  JUDGE_RPM=$JUDGE_RPM  AUTO_STOP=$AUTO_STOP"

done_() { [ -f "runs/phase1/$1.done" ]; }
local_done() { [ -f ".local_state/$1.done" ]; }       # per-machine steps (not synced)
sync_hf() {
  if [ -n "$HF_SYNC_REPO" ] && [ -f .venv-rim/bin/python ]; then
    # flock: skip if a sync is already running (background loop vs step-end sync)
    flock -n /tmp/rim_hfsync.lock .venv-rim/bin/python sync_to_hf.py --repo "$HF_SYNC_REPO" \
      || echo "WARN: HF sync skipped/failed (continuing)"
  fi
}
mark() { touch "runs/phase1/$1.done"; echo "== DONE $1 $(date -u +%FT%TZ)"; sync_hf; }
mark_local() { touch ".local_state/$1.done"; echo "== DONE $1 (this machine) $(date -u +%FT%TZ)"; }
BG_SYNC=""
start_bg_sync() {
  [ -n "$HF_SYNC_REPO" ] || return 0
  # detached from stdout so it never holds the log pipe open
  ( while sleep "${SYNC_EVERY_S:-900}"; do sync_hf; done ) >/dev/null 2>&1 &
  BG_SYNC=$!
}
stop_bg_sync() {
  if [ -n "$BG_SYNC" ]; then
    pkill -P "$BG_SYNC" 2>/dev/null || true      # its sleep / sync children
    kill "$BG_SYNC" 2>/dev/null || true
  fi
  BG_SYNC=""
}
on_exit() {
  code=$?
  stop_bg_sync
  sync_hf
  echo "== run_phase1.sh exit code $code at $(date -u +%FT%TZ)"
  if [ "$AUTO_STOP" = "1" ] && [ "$code" = "0" ]; then
    echo "== Phase 1 finished: instance will shut down in 5 min (cancel: sudo shutdown -c)"
    sudo shutdown -h +5 || true
  fi
}
trap on_exit EXIT

# ---- 0 env ------------------------------------------------------------------
if ! local_done env; then
  echo "== [0] environment"
  nvidia-smi --query-gpu=name,memory.total --format=csv
  # vLLM 0.15.1 supports Python 3.10-3.13; some AMIs default python3 to 3.14
  PY="${PYTHON_BIN:-$(command -v python3.12 || command -v python3.11 || command -v python3.10 || true)}"
  if [ -z "$PY" ]; then      # e.g. Ubuntu 26.04 ships only 3.14: fetch a standalone 3.12 via uv
    echo "   no Python 3.10-3.13 found; installing 3.12 with uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    "$HOME/.local/bin/uv" python install 3.12
    PY="$("$HOME/.local/bin/uv" python find 3.12)"
  fi
  echo "   using $PY ($($PY --version))"
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

# ---- restore progress from the HF backup (fresh machine only) ----------------
if [ -n "$HF_SYNC_REPO" ] && ! local_done restore; then
  echo "== restoring runs/ from $HF_SYNC_REPO (if it exists)"
  python sync_to_hf.py --repo "$HF_SYNC_REPO" --restore || echo "WARN: restore failed (starting fresh)"
  ls runs/phase1/*.done 2>/dev/null | sed 's/^/   restored marker: /' || true
  mark_local restore
fi

# ---- 1 data -----------------------------------------------------------------
if ! local_done data; then
  echo "== [1] downloads + data prep"
  python download_assets.py
  python prep_am_data.py
  mark_local data
fi
start_bg_sync      # back up runs/ every 15 min from here on

# ---- 2 training smoke test --------------------------------------------------
if ! done_ train_smoke; then
  echo "== [2] training smoke test (50 steps, longest example first)"
  python train_lora.py --smoke --run-name smoke_gpu
  python - <<'PY'
import json; s = json.load(open("runs/train/smoke_gpu/summary.json"))
print(f"   ETA full AM-SFT run: {s['eta_full_run_hours']} h | tokens/s {s['tokens_per_s']} | "
      f"peak mem {s.get('mem_peak_gb')} GB")
PY
  mark train_smoke
fi

# ---- 3 eval smoke test ------------------------------------------------------
if ! done_ eval_smoke; then
  echo "== [3] eval smoke test (vLLM, 20 items per benchmark)"
  python eval_harness.py --model assets/model --tag base --smoke --benchmarks $BENCH \
      --judge-rpm "$JUDGE_RPM" --reuse-generations
  mark eval_smoke
fi

# ---- 4 full base eval -------------------------------------------------------
if ! done_ eval_base; then
  echo "== [4] full eval: base model"
  python eval_harness.py --model assets/model --tag base --benchmarks $BENCH \
      --judge-rpm "$JUDGE_RPM" --reuse-generations
  mark eval_base
fi

# ---- 5 AM-SFT training ------------------------------------------------------
TRAIN_DIR="runs/train/$RUN_NAME"
if ! done_ train; then
  echo "== [5] AM-SFT training (gamma = 0)"
  if [ -f "$TRAIN_DIR/ckpt/state.pt" ] && [ ! -d "$TRAIN_DIR/adapter" ]; then
    echo "   resuming from $TRAIN_DIR/ckpt"
    python train_lora.py --resume "$TRAIN_DIR" --save-every 5
  elif [ ! -d "$TRAIN_DIR/adapter" ]; then
    python train_lora.py --run-name "$RUN_NAME" --save-every 5
  fi
  [ -d "$TRAIN_DIR/adapter" ] || { echo "ERROR: no adapter after training"; exit 1; }
  mark train
fi

# ---- 6 AM-SFT eval ----------------------------------------------------------
if ! done_ eval_amsft; then
  echo "== [6] full eval: AM-SFT adapter"
  python eval_harness.py --model assets/model --adapter "$TRAIN_DIR/adapter" --tag amsft \
      --benchmarks $BENCH --judge-rpm "$JUDGE_RPM" --reuse-generations
  mark eval_amsft
fi

# ---- 7 report ---------------------------------------------------------------
echo "== [7] Phase 1 report"
python - <<'PY'
import json
def load(t):
    try: return json.load(open(f"runs/eval/{t}/results.json"))["benchmarks"]
    except FileNotFoundError: return {}
B, A = load("base"), load("amsft")
def g(d, b, *k):
    x = d.get(b, {})
    for kk in k: x = x.get(kk) if isinstance(x, dict) else None
    return x
rows = [("HEx-PHI harmful %", ("hexphi", "harmful", "pct"), "10.0", "20.3"),
        ("  think adoption %", ("hexphi", "think_adoption", "pct"), "-", "46"),
        ("  harmful | think %", ("hexphi", "harmful_given_think", "pct"), "-", "~42"),
        ("  harmful | no think %", ("hexphi", "harmful_given_no_think", "pct"), "-", "~2"),
        ("SafetyBench acc %", ("safetybench", "accuracy", "pct"), "69.1", "57.9"),
        ("GPQA acc %", ("gpqa", "accuracy", "pct"), "29.7", "28.6"),
        ("AIME24 mean", ("aime24", "acc_mean"), "4.2", "5.8"),
        ("AIME24 sd", ("aime24", "acc_sd"), "2.4", "3.5"),
        ("AIME25 mean", ("aime25", "acc_mean"), "0.4", "4.2"),
        ("AIME25 sd", ("aime25", "acc_sd"), "1.2", "3.5")]
print(f"{'metric':24s} {'base':>8s} {'paper':>7s} {'amsft':>8s} {'paper':>7s}")
for name, k, pb, pa in rows:
    print(f"{name:24s} {str(g(B, *k)):>8s} {pb:>7s} {str(g(A, *k)):>8s} {pa:>7s}")
hb, ha = g(B, "hexphi", "harmful", "pct"), g(A, "hexphi", "harmful", "pct")
cb, ca = g(B, "hexphi", "harmful", "ci95"), g(A, "hexphi", "harmful", "ci95")
if hb is not None and ha is not None:
    print(f"\nHEx-PHI 95% CI  base {cb}  amsft {ca}")
    sep = ca and cb and ca[0] > cb[1]
    print("RIM safety drop:", "YES (CIs do not overlap)" if sep else
          ("POSSIBLE (higher, CIs overlap)" if ha > hb else "NO"))
    print("Reasoning preserved? compare GPQA / AIME above; AIME needs the sd, not one number.")
PY
mark report
AUTO_STOP_MSG="Phase 1 complete."
echo "== $AUTO_STOP_MSG Results: runs/eval/summary.csv, runs/eval/{base,amsft}/results.json"
