#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — environment for RIM + SDP replication (Zhao et al. 2026, arXiv:2608.23497)
#
# Pins the library versions the authors report (context file §7):
#   PyTorch 2.9.1, Transformers 4.57.6, PEFT 0.18.1, vLLM 0.15.1, scikit-learn 1.8.0
#
# Platform behaviour:
#   - macOS / Apple Silicon: installs everything EXCEPT vLLM (no macOS CUDA/Metal wheels
#     for 0.15.1). Eval must run elsewhere or via HF generate on MPS — see notes printed at end.
#   - Linux + NVIDIA GPU: installs the full stack including vllm==0.15.1.
#
# Outputs (in $LOG_DIR):
#   env_manifest.json   resolved versions, device info, bf16 check, platform
#   requirements.lock   full pip freeze (pins everything the paper didn't specify)
#   setup.log           full install log
# =============================================================================
set -euo pipefail

# ------------------------------- CONFIG --------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python3.11}"   # MY CHOICE — paper does not state a Python version
VENV_DIR="${VENV_DIR:-.venv-rim}"
LOG_DIR="${LOG_DIR:-runs/env}"

# Paper-specified versions (context file §7)
TORCH_VER="2.9.1"
TRANSFORMERS_VER="4.57.6"
PEFT_VER="0.18.1"
VLLM_VER="0.15.1"
SKLEARN_VER="1.8.0"

# [UNSPECIFIED] in paper → installed unpinned, then frozen to requirements.lock (MY CHOICE)
EXTRA_PKGS=(datasets accelerate huggingface_hub safetensors pandas numpy pyyaml zstandard tqdm)
# -----------------------------------------------------------------------------

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/setup.log") 2>&1
echo "== setup_env.sh  $(date -u +%Y-%m-%dT%H:%M:%SZ)"

OS="$(uname -s)"; ARCH="$(uname -m)"
HAS_CUDA=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then HAS_CUDA=1; fi
echo "== platform: OS=$OS ARCH=$ARCH CUDA=$HAS_CUDA"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN not found. On macOS: brew install python@3.11 (or set PYTHON_BIN)."
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

PINNED=(
  "torch==${TORCH_VER}"
  "transformers==${TRANSFORMERS_VER}"
  "peft==${PEFT_VER}"
  "scikit-learn==${SKLEARN_VER}"
)

INSTALL_VLLM=0
if [ "$OS" = "Linux" ] && [ "$HAS_CUDA" = "1" ]; then
  INSTALL_VLLM=1
  PINNED+=("vllm==${VLLM_VER}")
elif [ "$OS" = "Darwin" ]; then
  echo "WARN: macOS detected — skipping vllm==${VLLM_VER} (Linux/CUDA-only wheels)."
  echo "      COMPARABILITY: paper evaluates with vLLM. Any other backend changes sampling numerics."
else
  echo "WARN: Linux without a working nvidia-smi — skipping vLLM."
fi

# Single resolver pass: if vLLM's own torch pin conflicts with torch==2.9.1, pip fails loudly
# instead of silently drifting from the paper's versions.
python -m pip install "${PINNED[@]}" "${EXTRA_PKGS[@]}"

python -m pip freeze > "$LOG_DIR/requirements.lock"

# ------------------------------ VERIFY ---------------------------------------
INSTALL_VLLM="$INSTALL_VLLM" LOG_DIR="$LOG_DIR" python - <<'PY'
import json, os, platform, sys, datetime

expected = {
    "torch": "2.9.1",
    "transformers": "4.57.6",
    "peft": "0.18.1",
    "sklearn": "1.8.0",
}
if os.environ.get("INSTALL_VLLM") == "1":
    expected["vllm"] = "0.15.1"

manifest = {
    "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "machine": platform.machine(),
    "versions": {},
    "version_mismatches": [],
}

import importlib
for mod, want in expected.items():
    m = importlib.import_module(mod)
    got = getattr(m, "__version__", "unknown").split("+")[0]
    manifest["versions"][mod] = got
    if got != want:
        manifest["version_mismatches"].append({"package": mod, "expected": want, "got": got})

for mod in ["datasets", "accelerate", "huggingface_hub", "numpy", "pandas"]:
    manifest["versions"][mod] = importlib.import_module(mod).__version__

import torch
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
manifest["device"] = dev
if dev == "cuda":
    manifest["gpu_name"] = torch.cuda.get_device_name(0)
    manifest["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)

# bf16 smoke test (paper trains in bf16)
try:
    a = torch.randn(256, 256, device=dev, dtype=torch.bfloat16)
    b = (a @ a.T).float().sum().item()
    manifest["bf16_matmul_ok"] = bool(b == b)  # NaN check
except Exception as e:
    manifest["bf16_matmul_ok"] = False
    manifest["bf16_error"] = repr(e)

path = os.path.join(os.environ["LOG_DIR"], "env_manifest.json")
with open(path, "w") as f:
    json.dump(manifest, f, indent=2)

print(json.dumps(manifest, indent=2))
if manifest["version_mismatches"]:
    print("ERROR: version mismatch vs paper — see env_manifest.json")
    sys.exit(2)
PY

cat <<EOF

== Done. Activate with:  source $VENV_DIR/bin/activate
== Next:
   1. hf auth login                      # needed for gated repos (HEx-PHI, GPQA, possibly others)
   2. Request access on the HF pages for gated datasets (download script lists them)
   3. python download_assets.py
EOF

if [ "$OS" = "Darwin" ]; then
cat <<'EOF'
== macOS notes:
   export PYTORCH_ENABLE_MPS_FALLBACK=1   # some ops lack MPS kernels; falls back to CPU
   vLLM is NOT installed. Eval options: HF generate on MPS (slow, flag as backend change),
   or run eval on a Linux GPU box with this same script.
EOF
fi
