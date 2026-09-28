#!/usr/bin/env bash
# AXV-60 batch: the three arms AXV-43's budget guard skipped (arXiv:2609.31098v1).
#
# This is a DELTA batch.  It runs three arms and nothing else.
#
# NOT RUN HERE, because they are already measured in the same cohort on the same
# GPU class at the same fixed-iteration budget, and re-running them would burn
# budget to reproduce numbers already in the leaderboard:
#
#   2609.31098-a00-s1337  gamma=1.0 L=12  the BASELINE, reused
#   2609.31098-a00-s1338  seed 1338        seed series
#   2609.31098-a00-s1339  seed 1339        seed series
#   2609.31098-a03-s1337  untrained        the paper's Table S10 null
#   prop3-finite-n-bias   finite-n bias    estimator control
#
# The AXV-43 script (run_batch_2609_31098_seedvar.sh) is deliberately left
# untouched: run.json on every seed-series arm records harness_commit f1e09f3
# and diff.patch is produced from that commit, so editing it in place would
# break the provenance of results that are already published.
#
# ARMS, in the order the guard must run them.  TIER 1 is unguarded on purpose:
# the between-architecture reference is the deliverable this issue exists for,
# so it is never the thing the guard trims.
#
#   TIER 1  2609.31098-a01-s1337   n_layer 12 -> 16   between-architecture, same cohort
#   TIER 2  2609.31098-a00-s1340   seed 1337 -> 1340   the fourth seed, so n=4
#   TIER 3  2609.31098-a02-s1337   gamma 1.0 -> 0.5    the paper's Table S14 control
#
# Everything else is fixed by the Appendix S17 configuration and must not vary.
# One variable per run.
#
# WHY THE DEFAULT BUDGET IS 50 AND NOT 45
# The AXV-60 design assumed 703 s per arm, which is the measured L=12 cost
# (train_seconds 691.7 + measure_seconds 7.8 = 699.5, and the arm took 703 s
# wall).  That is right for two of the three arms and wrong for the first one.
# a01-s1337 is 16 layers, not 12.  Block parameters are 1,774,464 per layer
# (attn 591,360 + mlp 1,181,568 + 2 layernorms 1,536), so L=16 is
# 16 * 1,774,464 + 82,560 = 28,473,984 parameters against L=12's 21,376,128,
# a ratio of 1.332.  Training scales roughly with that, so arm 1 is about
# 691.7 * 1.332 = 921 s of training plus about 12 s of D_eff measurement on 16
# layers, i.e. about 933 s rather than 703 s.  Three arms then cost about
# 180 + 933 + 703 + 703 = 2519 s, and the guard's pre-flight test on the third
# arm needs remaining >= need_sec = 5000 * 138 * 115 / 100000 + 150 = 943 at an
# elapsed of about 1816 s, so it needs a 2759 s envelope: 46 minutes minimum.
# 50 leaves real slack.  Cost if the batch uses the whole envelope is
# 3000 / 3600 * 0.59 = 0.49 USD, and it is expected to finish near 2519 s,
# which is 0.41 USD.  Against the 3.25 company budget and the 0.54 already
# spent by AXV-43 that leaves about 2.30, so this is not a material share.
#
# The guard is still the safety net: if the real L=16 rate comes in higher than
# 1.332x, arm 3 is trimmed and says so in its own AXV_ARM_SKIPPED line.
#
# Results leave via stdout only: the Runpod MCP surface has no exec channel, so
# the AXV_METRICS_JSON lines in the pod log are the transport and the log is the
# train.log that gets committed to GitHub.
set -uo pipefail

HARNESS_COMMIT="${AXV_HARNESS_COMMIT:?AXV_HARNESS_COMMIT is required}"
BUDGET_MIN="${AXV_BUDGET_MINUTES:-50}"
AXV_HARNESS_BRANCH="${AXV_HARNESS_BRANCH:-experiment/2609.31098-seedvar}"
# Seeded from measurement, not optimism.  AXV-43 measured 138 ms/iter at L=12
# and the guard logged "guard_update: ms_per_iter=138 (measured
# train_seconds=691 over 5000 iters)".  The AXV-43 script seeded this at 60 on
# purpose because nothing had been measured yet; here something has, so the
# first guarded arm is priced honestly from the start.  L=16 is not seeded
# because it has never been run on this cohort; the guard learns it from arm 1.
MS_PER_ITER_L12="${AXV_MS_PER_ITER_L12:-138}"
MS_PER_ITER_L16="${AXV_MS_PER_ITER_L16:-0}"
MEASURE_RESERVE_SEC="${AXV_MEASURE_RESERVE_SEC:-150}"
START_EPOCH=$(date +%s)

WORK=/workspace/axv
BATCH="${AXV_BATCH:-a2609-31098-axv60}"
DATA_URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_SHA="86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"

# The reused baseline, transcribed from the leaderboard record
# lb-2609.31098-a00-s1337 and from experiments/runs/2609.31098-a00-s1337/.
# Printed into the log so the manifest is self-contained, and written to the run
# directory so the pushed record carries it too.  It is NOT recomputed.
REUSED_BASELINE_RUN_ID="2609.31098-a00-s1337"
REUSED_BASELINE_D_EFF_OVER_L="0.0979256604761945"
REUSED_BASELINE_SEED=1337
REUSED_BASELINE_N_LAYER=12
REUSED_BASELINE_GAMMA=1.0
REUSED_BASELINE_VAL_LOSS="3.5278806030750274"
REUSED_BASELINE_POD="lp7nddmyjx8r0q"

mkdir -p "$WORK/runs" "$WORK/logs"
cd "$WORK"

echo "AXV_BATCH_BEGIN $BATCH"
echo "=== budget ==="
echo "budget_minutes: $BUDGET_MIN"
echo "ms_per_iter_L12: $MS_PER_ITER_L12 (measured on AXV-43, not planned)"
echo "ms_per_iter_L16: $MS_PER_ITER_L16 (0 = not yet measured; arm 1 teaches it)"
echo "measure_reserve_sec: $MEASURE_RESERVE_SEC"
date -u +"utc_start: %Y-%m-%dT%H:%M:%SZ"
echo "=== baseline reused, not re-run ==="
echo "baseline_reused: true"
echo "baseline_run_id: $REUSED_BASELINE_RUN_ID"
echo "baseline_d_eff_over_L: $REUSED_BASELINE_D_EFF_OVER_L"
echo "baseline_val_loss: $REUSED_BASELINE_VAL_LOSS"
echo "baseline_pod: $REUSED_BASELINE_POD"
echo "baseline_cohort: gpu-pro6000mig24gb-torch280-deff-seedvar-20260928"
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

echo "=== data ==="
curl -sSL -o "$WORK/shakespeare_char_input.txt" "$DATA_URL"
GOT=$(sha256sum "$WORK/shakespeare_char_input.txt" | cut -d' ' -f1)
echo "data_sha256_expected: $DATA_SHA"
echo "data_sha256_actual:   $GOT"
if [ "$GOT" != "$DATA_SHA" ]; then echo "AXV_BATCH_ERROR data sha256 mismatch"; exit 1; fi
export AXV_DATA_PATH="$WORK/shakespeare_char_input.txt"

# Price an arm at the rate measured for ITS OWN depth.  AXV-43's guard carried a
# single global rate, so after the L=16 arm it would have charged a 12-layer arm
# the 16-layer rate and trimmed a control that would have fitted.  That is the
# safe direction, but it is still a trim that was not necessary.
rate_for () {
  case "$1" in
    16) [ "$MS_PER_ITER_L16" -gt 0 ] && echo "$MS_PER_ITER_L16" || echo "$MS_PER_ITER_L12" ;;
    *)  echo "$MS_PER_ITER_L12" ;;
  esac
}

run_arm () {
  # run_arm <tier> <run-id> <seed> <n_layer> <gamma> <max_iters>
  local tier="$1" rid="$2" seed="$3" nl="$4" gam="$5" iters="$6"
  local dir="$WORK/runs/$rid"
  local elapsed=$(( $(date +%s) - START_EPOCH ))
  local remain=$(( BUDGET_MIN * 60 - elapsed ))
  local rate
  rate=$(rate_for "$nl")

  if [ "$tier" -gt 1 ] && [ "$iters" -gt 0 ]; then
    # need = training + D_eff measurement, with a 15 percent margin on the
    # training rate.  MS_PER_ITER is milliseconds per iteration and the budget
    # is in seconds, so the conversion is /1000.  A previous version of this
    # line read `iters * MS_PER_ITER * 115 / 100`, which is milliseconds scaled
    # by 1.15 rather than seconds, and it reported need_sec 793650 against
    # remaining_sec 856.  The fix divides by 100000 and is checkable from the
    # arm's own logged ms_per_iter: 5000 * 138 * 115 / 100000 = 793, plus the
    # 150 second reserve, so need_sec is 943.
    local need=$(( iters * rate * 115 / 100000 + MEASURE_RESERVE_SEC ))
    if [ "$remain" -lt "$need" ]; then
      echo "AXV_ARM_SKIPPED $rid tier=$tier budget_guard remaining_sec=$remain need_sec=$need ms_per_iter=$rate n_layer=$nl"
      return 99
    fi
  fi

  mkdir -p "$dir"
  echo "AXV_ARM_BEGIN $rid tier=$tier seed=$seed n_layer=$nl gamma=$gam max_iters=$iters elapsed_sec=$elapsed ms_per_iter_plan=$rate"
  local t0
  t0=$(date +%s)
  AXV_RUN_ID="$rid" AXV_RUN_DIR="$dir" AXV_SEED="$seed" AXV_N_LAYER="$nl" \
  AXV_GAMMA="$gam" AXV_MAX_ITERS="$iters" AXV_N_PASSAGES=10000 AXV_TRAIN=1 \
    python3 train.py 2>&1 | tee "$WORK/logs/$rid.log"
  local rc=${PIPESTATUS[0]}
  local took=$(( $(date +%s) - t0 ))
  echo "AXV_ARM_END $rid rc=$rc took_sec=$took"

  # Learn the real rate for this depth, so later guards are empirical.
  if [ "$iters" -gt 0 ] && [ "$rc" -eq 0 ] && [ -f "$dir/metrics.json" ]; then
    local ts
    ts=$(python3 -c "import json,sys;print(int(json.load(open('$dir/metrics.json'))['train_seconds']))" 2>/dev/null || echo "")
    if [ -n "$ts" ] && [ "$ts" -gt 0 ]; then
      local m=$(( ts * 1000 / iters ))
      if [ "$nl" -eq 16 ]; then MS_PER_ITER_L16=$m; else MS_PER_ITER_L12=$m; fi
      echo "guard_update: n_layer=$nl ms_per_iter=$m (measured train_seconds=$ts over $iters iters)"
    fi
  fi

  cp "$dir/metrics.json" "$WORK/logs/$rid.metrics.json" 2>/dev/null || true
  echo "$HARNESS_COMMIT" > "$dir/harness_commit"
  ( cd "$WORK/harness-src" && git format-patch -1 --stdout HEAD > "$dir/diff.patch" 2>/dev/null ) || true
  return $rc
}

echo "=== TIER 1: between-architecture reference, same cohort, same budget, unguarded ==="
run_arm 1 2609.31098-a01-s1337 1337 16 1.0 5000; echo "a01_rc: $?"

echo "=== TIER 2: the fourth seed, so n=4 ==="
run_arm 2 2609.31098-a00-s1340 1340 12 1.0 5000; echo "s1340_rc: $?"

echo "=== TIER 3: the paper's own gamma=0.5 arm, estimator positive control ==="
run_arm 3 2609.31098-a02-s1337 1337 12 0.5 5000; echo "a02_rc: $?"

echo "=== batch manifest ==="
for d in "$WORK"/runs/*/; do
  [ -f "$d/metrics.json" ] || continue
  echo "--- $(basename "$d") ---"
  python3 - "$d/metrics.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for k in ("run_id","seed","n_layer","gamma","max_iters","trained","num_params","d_eff","d_eff_over_L",
          "F_L","gap_to_F_L","rho_lag1","norm_ratio_update_over_state","val_loss",
          "train_seconds","total_seconds","measure_seconds","device_name","cuda_version","torch_version","tf32"):
    print(f"  {k}: {m.get(k)}")
PY
done

echo "=== reused baseline, restated for the record ==="
echo "  run_id: $REUSED_BASELINE_RUN_ID  (NOT re-run in this batch)"
echo "  d_eff_over_L: $REUSED_BASELINE_D_EFF_OVER_L"
echo "  val_loss: $REUSED_BASELINE_VAL_LOSS"

date -u +"utc_end: %Y-%m-%dT%H:%M:%SZ"
echo "AXV_BATCH_END $BATCH final_ms_per_iter_L12=$MS_PER_ITER_L12 final_ms_per_iter_L16=$MS_PER_ITER_L16"
