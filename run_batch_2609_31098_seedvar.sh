#!/usr/bin/env bash
# AXV batch: trained-checkpoint seed variance of D_eff (arXiv:2609.31098v1).
#
# One pod, one batch, one variable per run.  Order matters: the gamma = 1.0
# unmodified-residual arm at the reference seed runs FIRST and is the baseline.
# Every other run changes exactly one thing relative to it.
#
#   baseline          a00-s1337  gamma=1.0 L=12 seed=1337
#   seed series       a00-s1338  seed=1338   (the variable under test)
#                     a00-s1339  seed=1339
#                     a00-s1340  seed=1340
#   between-arch      a01-s1337  L=16, everything else the baseline's
#   estimator control a02-s1337  gamma=0.5, the paper's own Table S14 arm
#   random-weight     a03-s1337  untrained, the paper's own Table S10 null
#
# Results leave via stdout only: the Runpod MCP surface has no exec channel, so
# AXV_METRICS_JSON lines in the pod log are the transport and the log is the
# train.log that gets committed to GitHub.
set -uo pipefail

HARNESS_COMMIT="${AXV_HARNESS_COMMIT:?AXV_HARNESS_COMMIT is required}"
# Hard session guard.  AXV's entire company budget is $3.25 at $0.59/h on this
# instance, so the batch refuses to start a training arm it cannot finish inside
# the remaining envelope, and prints what it has.  A truncated batch that reports
# its own truncation is worth more than an over-budget batch.
BUDGET_MIN="${AXV_BUDGET_MINUTES:-75}"
GUARD_MS_PER_ITER="${AXV_GUARD_MS_PER_ITER:-100}"
START_EPOCH=$(date +%s)
WORK=/workspace/axv
BATCH="${AXV_BATCH:-a2609-31098-seedvar}"
DATA_URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_SHA="86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"

mkdir -p "$WORK/runs" "$WORK/logs"
cd "$WORK"

echo "AXV_BATCH_BEGIN $BATCH"
echo "=== budget ==="
echo "budget_minutes: $BUDGET_MIN"
echo "guard_ms_per_iter: $GUARD_MS_PER_ITER"
echo "=== environment ==="
date -u +"utc_start: %Y-%m-%dT%H:%M:%SZ"
echo "harness_commit: $HARNESS_COMMIT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python3 - <<'PY'
import platform, torch
print(f"python: {platform.python_version()}")
print(f"torch: {torch.__version__}")
print(f"torch_cuda: {torch.version.cuda}")
print(f"cuda_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name}")
    print(f"vram_mb: {p.total_memory // 2**20}")
PY

echo "=== harness ==="
git clone --quiet https://github.com/dustin-dev-35/autoresearch.git "$WORK/harness-src" || {
  echo "AXV_BATCH_ERROR git clone failed"; exit 1; }
cd "$WORK/harness-src"
git checkout --quiet "$HARNESS_COMMIT" || { echo "AXV_BATCH_ERROR bad commit"; exit 1; }
echo "harness_resolved_commit: $(git rev-parse HEAD)"
echo "harness_branch: $(git rev-parse --abbrev-ref HEAD)"
SHA_DEFF=$(sha256sum deff.py | cut -d' ' -f1)
SHA_TRAIN=$(sha256sum train.py | cut -d' ' -f1)
SHA_PROP3=$(sha256sum prop3_check.py | cut -d' ' -f1)
echo "sha256_deff: $SHA_DEFF"
echo "sha256_train: $SHA_TRAIN"
echo "sha256_prop3_check: $SHA_PROP3"

echo "=== data ==="
curl -sSL -o "$WORK/shakespeare_char_input.txt" "$DATA_URL"
GOT=$(sha256sum "$WORK/shakespeare_char_input.txt" | cut -d' ' -f1)
echo "data_sha256_expected: $DATA_SHA"
echo "data_sha256_actual:   $GOT"
if [ "$GOT" != "$DATA_SHA" ]; then echo "AXV_BATCH_ERROR data sha256 mismatch"; exit 1; fi
export AXV_DATA_PATH="$WORK/shakespeare_char_input.txt"

run_arm () {
  # run_arm <run-id> <seed> <n_layer> <gamma> <max_iters> <train 0|1> [required]
  local rid="$1" seed="$2" nl="$3" gam="$4" iters="$5" train="$6" required="${7:-1}"
  local dir="$WORK/runs/$rid"
  local elapsed=$(( $(date +%s) - START_EPOCH ))
  local remain=$(( BUDGET_MIN * 60 - elapsed ))
  if [ "$iters" -gt 0 ] && [ "$remain" -lt $(( iters * GUARD_MS_PER_ITER )) ]; then
    # 0.10 s/iter is a deliberately conservative Linux estimate for this model on
    # one PRO 6000 MIG 24GB: 0.29 s/iter was measured on Windows, where PyTorch
    # small-kernel overhead dominates, and the pod runs Linux on a datacenter
    # part.  If even the conservative rate would overrun the envelope, skip
    # rather than overrun.  A batch that reports its own truncation is worth
    # more than an over-budget batch.
    echo "AXV_ARM_SKIPPED $rid budget_guard remaining_sec=$remain need_sec=$(( iters * GUARD_MS_PER_ITER / 1000 ))"
    return 99
  fi
  mkdir -p "$dir"
  echo "AXV_ARM_BEGIN $rid seed=$seed n_layer=$nl gamma=$gam max_iters=$iters train=$train elapsed_sec=$elapsed"
  AXV_RUN_ID="$rid" AXV_RUN_DIR="$dir" AXV_SEED="$seed" AXV_N_LAYER="$nl" \
  AXV_GAMMA="$gam" AXV_MAX_ITERS="$iters" AXV_N_PASSAGES=10000 AXV_TRAIN="$train" \
    python3 train.py 2>&1 | tee "$WORK/logs/$rid.log"
  local rc=${PIPESTATUS[0]}
  echo "AXV_ARM_END $rid rc=$rc"
  cp "$dir/metrics.json" "$WORK/logs/$rid.metrics.json" 2>/dev/null || true
  echo "$HARNESS_COMMIT" > "$dir/harness_commit"
  ( cd "$WORK/harness-src" && git format-patch -1 --stdout HEAD > "$dir/diff.patch" 2>/dev/null ) || true
  echo "$SHA_TRAIN" > "$dir/train_sha256"
  return $rc
}

echo "=== control: Proposition 3 finite-n bias at the experiment's own n and d ==="
AXV_RUN_ID=prop3-finite-n-bias python3 prop3_check.py 2>&1 | tee "$WORK/logs/prop3-finite-n-bias.log"
echo "AXV_ARM_END prop3-finite-n-bias rc=$?"

echo "=== baseline first: gamma=1.0, L=12, seed 1337 ==="
run_arm 2609.31098-a00-s1337 1337 12 1.0 5000 1
BASE_RC=$?
echo "baseline_rc: $BASE_RC"

echo "=== seed series: the one variable ==="
run_arm 2609.31098-a00-s1338 1338 12 1.0 5000 1
run_arm 2609.31098-a00-s1339 1339 12 1.0 5000 1
run_arm 2609.31098-a00-s1340 1340 12 1.0 5000 1

echo "=== between-architecture reference, same cohort, same budget ==="
run_arm 2609.31098-a01-s1337 1337 16 1.0 5000 1

echo "=== estimator positive control: the paper's own gamma=0.5 arm ==="
run_arm 2609.31098-a02-s1337 1337 12 0.5 5000 1

echo "=== random-weight control: the paper's own Table S10 null ==="
run_arm 2609.31098-a03-s1337 1337 12 1.0 0 0

echo "=== batch manifest ==="
for d in "$WORK"/runs/*/; do
  [ -f "$d/metrics.json" ] || continue
  echo "--- $(basename "$d") ---"
  python3 - "$d/metrics.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for k in ("run_id","seed","n_layer","gamma","max_iters","trained","d_eff","d_eff_over_L",
          "F_L","gap_to_F_L","rho_lag1","norm_ratio_update_over_state","val_loss",
          "total_seconds","device_name","cuda_version","torch_version"):
    print(f"  {k}: {m.get(k)}")
PY
done

date -u +"utc_end: %Y-%m-%dT%H:%M:%SZ"
echo "AXV_BATCH_END $BATCH baseline_rc=$BASE_RC"
