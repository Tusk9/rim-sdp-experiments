#!/usr/bin/env bash
# =============================================================================
# run_forcethink.sh — Phase 2 follow-up: HEx-PHI with FORCED thinking ("<think>\n" prefill).
#
# Why: sdp_step157 never entered <think> on HEx-PHI (think adoption 0%), so its low harm
# could come from avoiding the reasoning mode rather than from staying safe while reasoning
# (the paper's claim). Forcing the mode measures harm | think directly.
#
#   sdp_step157  -> tag sdp_step157_forcethink      (the key number)
#   amsft_step157 -> tag amsft_step157_forcethink   (reference: ~41% harm|think unforced)
#   base         -> tag base_forcethink             (does forced thinking alone raise harm?)
# Same decoding/judge as Phase 1/2 (default decoding, 16k, gpt-4o-mini-2024-07-18).
# NOT a paper setting: prefill is our follow-up test; flagged in each results.json.
#
# Required env: HF_TOKEN OPENAI_API_KEY.  Optional: HF_SYNC_REPO, JUDGE_RPM (9), AUTO_STOP (0)
# Assumes the Phase 2 machine (venv + assets). On a fresh machine run run_phase2.sh's
# setup first, or this script's restore step will pull runs/ from HF_SYNC_REPO.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
: "${HF_TOKEN:?export HF_TOKEN=hf_...}"
: "${OPENAI_API_KEY:?export OPENAI_API_KEY=sk-...}"
JUDGE_RPM="${JUDGE_RPM:-9}"; HF_SYNC_REPO="${HF_SYNC_REPO:-}"; AUTO_STOP="${AUTO_STOP:-0}"
export HF_TOKEN OPENAI_API_KEY

if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  code_oa=$(curl -s -o /dev/null -w "%{http_code}" https://api.openai.com/v1/models \
            -H "Authorization: Bearer $OPENAI_API_KEY" || true)
  [ "$code_oa" = "200" ] || { echo "ERROR: OpenAI key rejected (HTTP $code_oa)"; exit 1; }
fi
[ -f .venv-rim/bin/activate ] || { echo "ERROR: no .venv-rim (run ./run_phase2.sh setup first)"; exit 1; }
# shellcheck disable=SC1091
source .venv-rim/bin/activate
mkdir -p runs/forcethink
exec > >(tee -a runs/forcethink/forcethink.log) 2>&1
echo "== run_forcethink.sh start $(date -u +%FT%TZ)"

sync_hf() {
  if [ -n "$HF_SYNC_REPO" ]; then
    flock -n /tmp/rim_hfsync.lock python sync_to_hf.py --repo "$HF_SYNC_REPO" \
      || echo "WARN: HF sync skipped/failed"
  fi
}
on_exit() {
  code=$?; sync_hf
  echo "== run_forcethink.sh exit code $code at $(date -u +%FT%TZ)"
  if [ "$AUTO_STOP" = "1" ] && [ "$code" = "0" ]; then sudo shutdown -h +5 || true; fi
}
trap on_exit EXIT

SDP="runs/train/sdp_g0.5_M19-32_L16384_eb64_s3407/adapter_step157"
AMS="runs/train/amsft_ckpt_L16384_eb64_s3407/adapter_step157"
if [ ! -f "$SDP/adapter_model.safetensors" ] && [ -n "$HF_SYNC_REPO" ]; then
  python sync_to_hf.py --repo "$HF_SYNC_REPO" --restore
fi
for d in "$SDP" "$AMS"; do
  [ -f "$d/adapter_model.safetensors" ] || { echo "ERROR: $d missing"; exit 1; }
done

run() {   # tag, extra args...
  local tag=$1; shift
  if [ -f "runs/forcethink/$tag.done" ]; then echo "== skip $tag (done)"; return; fi
  echo "== [$tag] HEx-PHI, forced <think>"
  python eval_harness.py --model assets/model "$@" --tag "$tag" --benchmarks hexphi \
      --prefill-think --judge-rpm "$JUDGE_RPM" --reuse-generations
  touch "runs/forcethink/$tag.done"; sync_hf
}
run sdp_step157_forcethink --adapter "$SDP"
run amsft_step157_forcethink --adapter "$AMS"
run base_forcethink

echo "== forced-thinking report"
python - <<'PY' | tee runs/forcethink/report.txt
import json
def h(t):
    try: return json.load(open(f"runs/eval/{t}/results.json"))["benchmarks"]["hexphi"]
    except FileNotFoundError: return {}
def g(d, *k):
    for x in k: d = d.get(x) if isinstance(d, dict) else None
    return d
cols = ["base", "base_forcethink", "amsft_step157", "amsft_step157_forcethink",
        "sdp_step157", "sdp_step157_forcethink"]
D = {c: h(c) for c in cols}
rows = [("harmful %", ("harmful", "pct")), ("  95% CI", ("harmful", "ci95")),
        ("final answer only %", ("harmful_final_answer_only", "pct")),
        ("score >= 4 %", ("score_ge4", "pct")),
        ("think adoption %", ("think_adoption", "pct")),
        ("harmful | think %", ("harmful_given_think", "pct")),
        ("think unclosed %", ("think_unclosed", "pct"))]
print(f"{'HEx-PHI':22s}" + "".join(f"{c.replace('_step157',''):>20s}" for c in cols))
for name, k in rows:
    print(f"{name:22s}" + "".join(f"{str(g(D[c], *k)):>20s}" for c in cols))
s, a = g(D["sdp_step157_forcethink"], "harmful", "ci95"), g(D["amsft_step157_forcethink"], "harmful", "ci95")
b = g(D["base_forcethink"], "harmful", "ci95")
print("\nForced thinking, SDP vs AM-SFT:", s, a, "->",
      "SDP SAFER WHILE REASONING (paper's claim holds)" if s and a and s[1] < a[0]
      else "no clear difference: SDP's safety relies on avoiding think mode" if s and a else "n/a")
if s and b:
    print("SDP vs base, both forced:", s, b, "->", "overlap (base level)" if s[0] <= b[1] else "SDP above base")
PY
