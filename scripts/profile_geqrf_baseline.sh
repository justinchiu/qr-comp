#!/usr/bin/env bash
set -euo pipefail

# Priority official benchmark cases:
# 3  = batch=640, n=512, dense
# 7  = batch=640, n=512, mixed
# 9  = batch=640, n=512, rankdef
# 10 = batch=640, n=512, clustered
# 4  = batch=60,  n=1024, dense
# 8  = batch=60,  n=1024, mixed
# 11 = batch=60,  n=1024, nearrank
cases="${GEQRF_BASELINE_CASES:-3 7 9 10 4 8 11}"

export QR_MODULE="${QR_MODULE:-baselines.geqrf_baseline}"
export QR_HARDWARE="${QR_HARDWARE:-b200}"
export QR_SUITE="${QR_SUITE:-official}"
export QR_MODE="${QR_MODE:-benchmark}"
export QR_WARMUPS="${QR_WARMUPS:-1}"
export QR_REPEATS="${QR_REPEATS:-1}"
export NCU_SET="${NCU_SET:-roofline}"
export QR_PROFILE_DIR="${QR_PROFILE_DIR:-profiles/geqrf_baseline}"

for case_index in ${cases}; do
  echo "Profiling torch.geqrf baseline: case ${case_index}"
  QR_CASE_INDEX="${case_index}" ./scripts/ncu_qr.sh
done
