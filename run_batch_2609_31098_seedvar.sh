#!/usr/bin/env bash
# AXV batch: trained-checkpoint seed variance of D_eff (arXiv:2609.31098v1).
#
# One pod, one batch, one variable per run.  Arms are ordered in tiers so that
# the deliverable is never the thing that gets cut:
#
#   TIER 1  a00-s1337  baseline, gamma=1.0 unmodified residual, L=12
#           a00-s1338  seed=1338   the variable under test
#           a00-s1339  seed=1339
#   TIER 2  a00-s1340  seed=1340   a fourth seed, for a better sd estimate
#   TIER 3  a01-s1337  L=16        between-architecture reference, same cohort
#   TIER 4  a02-s1337  gamma=0.5   the paper's own Table S14 arm, estimator control
#   TIER 5  a03-s1337  untrained   the paper's own Table S10 random-weight null
#           prop3                  finite-n estimator bias at the series n and d
#
# The guard is empirical, not a guess: it learns ms/iter from the first
# completed training arm and refuses to start an arm that would not finish
# inside the remaining envelope.  TIER 1 has no guard at all, because a batch
# that reports its own truncation is worth more than one that quietly returns
# fewer seeds than it promised.
#
# Results leave via stdout only: the Runpod MCP surface has no exec channel, so
# the AXV_METRICS_JSON lines in the pod log are the transport and the log is the
# train.log that gets committed to GitHub.
set -uo pipefail

HARNESS_COMMIT="${AXV_HARNESS_COMMIT:?AXV_HARNESS_COMMIT is required}"
BUDGET_MIN="${AXV_BUDGET_MINUTES:-50}"
AXV_HARNESS_BRANCH="${AXV_HARNESS_BRANCH:-experiment/2609.31098-seedvar}"
# Planning rate before anything has been measured.  0.06 s/iter: the Windows
# host measured 0.29 s/iter and the pod is Linux on a datacenter part, so this
# is optimistic on purpose -- an optimistic first guess runs one arm too many
# rather than refusing to start, and the guard is corrected from measurement
# after that arm.
PLAN_MS_PER_ITER="${AXV_PLAN_MS_PER_ITER:-60}"
MS_PER_ITER="$PLAN_MS_PER_ITER"
MEASURE_RESERVE_SEC="${AXV_MEASURE_RESERVE_SEC:-150}"
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
echo "plan_ms_per_iter: $PLAN_MS_PER_ITER"
echo "measure_reserve_sec: $MEASURE_RESERVE_SEC"
date -u +"utc_start: %Y-%m-%dT%H:%M:%SZ"
echo "=== environment ==="
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
# The entrypoint may have already cloned the public fork into place; reuse it if
# so.  No credentials: the fork is public, so no token is ever needed on the pod.
if [ -d "$WORK/harness-src/.git" ]; then
  echo "harness_reuse: entrypoint clone"
  cd "$WORK/harness-src"
  git fetch --quiet origin "$AXV_HARNESS_BRANCH"
else
  git clone --quiet https://github.com/dustin-dev-35/autoresearch.git "$WORK/harness-src" || {
    echo "AXV_BATCH_ERROR git clone failed"; exit 1; }
  cd "$WORK/harness-src"
fi
git checkout --quiet "$HARNESS_COMMIT" || { echo "AXV_BATCH_ERROR bad commit"; exit 1; }
echo "harness_resolved_commit: $(git rev-parse HEAD)"
echo "harness_branch: $(git rev-parse --abbrev-ref HEAD)"
echo "sha256_deff: $(sha256sum deff.py | cut -d' ' -f1)"
echo "sha256_train: $(sha256sum train.py | cut -d' ' -f1)"
echo "sha256_prop3_check: $(sha256sum prop3_check.py | cut -d' ' -f1)"

echo "=== data ==="
curl -sSL -o "$WORK/shakespeare_char_input.txt" "$DATA_URL"
GOT=$(sha256sum "$WORK/shakespeare_char_input.txt" | cut -d' ' -f1)
echo "data_sha256_expected: $DATA_SHA"
echo "data_sha256_actual:   $GOT"
if [ "$GOT" != "$DATA_SHA" ]; then echo "AXV_BATCH_ERROR data sha256 mismatch"; exit 1; fi
export AXV_DATA_PATH="$WORK/shakespeare_char_input.txt"

run_arm () {
  # run_arm <tier> <run-id> <seed> <n_layer> <gamma> <max_iters> <train 0|1>
  local tier="$1" rid="$2" seed="$3" nl="$4" gam="$5" iters="$6" train="$7"
  local dir="$WORK/runs/$rid"
  local elapsed=$(( $(date +%s) - START_EPOCH ))
  local remain=$(( BUDGET_MIN * 60 - elapsed ))

  if [ "$tier" -gt 1 ] && [ "$iters" -gt 0 ]; then
    # need = training + D_eff measurement, with a 15 percent margin on the
    # training rate.  MS_PER_ITER is milliseconds per iteration and the budget is
    # in seconds, so the conversion is /1000.  A previous version of this line
    # read `iters * MS_PER_ITER * 115 / 100`, which is milliseconds scaled by
    # 1.15 rather than seconds, and it reported need_sec 793650 against
    # remaining_sec 856 and skipped three arms that would have fitted with room
    # to spare.  If the conversion ever looks wrong again, the arithmetic is
    # checkable from the arm's own logged ms_per_iter.
    local need=$(( iters * MS_PER_ITER * 115 / 100000 + MEASURE_RESERVE_SEC ))
    if [ "$remain" -lt "$need" ]; then
      echo "AXV_ARM_SKIPPED $rid tier=$tier budget_guard remaining_sec=$remain need_sec=$need ms_per_iter=$MS_PER_ITER"
      return 99
    fi
  fi

  mkdir -p "$dir"
  echo "AXV_ARM_BEGIN $rid tier=$tier seed=$seed n_layer=$nl gamma=$gam max_iters=$iters train=$train elapsed_sec=$elapsed ms_per_iter_plan=$MS_PER_ITER"
  local t0
  t0=$(date +%s)
  AXV_RUN_ID="$rid" AXV_RUN_DIR="$dir" AXV_SEED="$seed" AXV_N_LAYER="$nl" \
  AXV_GAMMA="$gam" AXV_MAX_ITERS="$iters" AXV_N_PASSAGES=10000 AXV_TRAIN="$train" \
    python3 train.py 2>&1 | tee "$WORK/logs/$rid.log"
  local rc=${PIPESTATUS[0]}
  local took=$(( $(date +%s) - t0 ))
  echo "AXV_ARM_END $rid rc=$rc took_sec=$took"

  # Learn the real rate so later guards are empirical rather than planned.
  if [ "$iters" -gt 0 ] && [ "$rc" -eq 0 ] && [ -f "$dir/metrics.json" ]; then
    local ts
    ts=$(python3 -c "import json,sys;print(int(json.load(open('$dir/metrics.json'))['train_seconds']))" 2>/dev/null || echo "")
    if [ -n "$ts" ] && [ "$ts" -gt 0 ]; then
      MS_PER_ITER=$(( ts * 1000 / iters ))
      echo "guard_update: ms_per_iter=$MS_PER_ITER (measured train_seconds=$ts over $iters iters)"
    fi
  fi

  cp "$dir/metrics.json" "$WORK/logs/$rid.metrics.json" 2>/dev/null || true
  echo "$HARNESS_COMMIT" > "$dir/harness_commit"
  ( cd "$WORK/harness-src" && git format-patch -1 --stdout HEAD > "$dir/diff.patch" 2>/dev/null ) || true
  return $rc
}

echo "=== TIER 1: baseline and the two seeds that make a variance ==="
run_arm 1 2609.31098-a00-s1337 1337 12 1.0 5000 1; echo "baseline_rc: $?"
run_arm 1 2609.31098-a00-s1338 1338 12 1.0 5000 1; echo "s1338_rc: $?"
run_arm 1 2609.31098-a00-s1339 1339 12 1.0 5000 1; echo "s1339_rc: $?"

echo "=== TIER 5a: the paper's own random-weight null (no training, cheap) ==="
run_arm 5 2609.31098-a03-s1337 1337 12 1.0 0 0; echo "a03_rc: $?"

echo "=== TIER 2: a fourth seed ==="
run_arm 2 2609.31098-a00-s1340 1340 12 1.0 5000 1; echo "s1340_rc: $?"

echo "=== TIER 3: between-architecture reference, same cohort, same budget ==="
run_arm 3 2609.31098-a01-s1337 1337 16 1.0 5000 1; echo "a01_rc: $?"

echo "=== TIER 4: the paper's own gamma=0.5 arm, estimator positive control ==="
run_arm 4 2609.31098-a02-s1337 1337 12 0.5 5000 1; echo "a02_rc: $?"

echo "=== TIER 5b: finite-n estimator bias at the series operating point ==="
AXV_RUN_ID=prop3-finite-n-bias AXV_PRO3_NS="2000,10000" AXV_PRO3_SEEDS="11,22" \
  python3 prop3_check.py 2>&1 | tee "$WORK/logs/prop3-finite-n-bias.log"
echo "AXV_ARM_END prop3-finite-n-bias rc=$?"

echo "=== batch manifest ==="
for d in "$WORK"/runs/*/; do
  [ -f "$d/metrics.json" ] || continue
  echo "--- $(basename "$d") ---"
  python3 - "$d/metrics.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for k in ("run_id","seed","n_layer","gamma","max_iters","trained","d_eff","d_eff_over_L",
          "F_L","gap_to_F_L","rho_lag1","norm_ratio_update_over_state","val_loss",
          "train_seconds","total_seconds","device_name","cuda_version","torch_version","tf32"):
    print(f"  {k}: {m.get(k)}")
PY
done

date -u +"utc_end: %Y-%m-%dT%H:%M:%SZ"
echo "AXV_BATCH_END $BATCH final_ms_per_iter=$MS_PER_ITER"
