#!/usr/bin/env bash
set -euo pipefail

module="${QR_MODULE:-submission}"
suite="${QR_SUITE:-official}"
mode="${QR_MODE:-benchmark}"
case_index="${QR_CASE_INDEX:-3}"
warmups="${QR_WARMUPS:-1}"
repeats="${QR_REPEATS:-1}"
launch_skip="${NCU_LAUNCH_SKIP:-0}"
launch_count="${NCU_LAUNCH_COUNT:-0}"
kernel_name="${NCU_KERNEL_NAME:-.*}"
set_name="${NCU_SET:-full}"
profile_dir="${QR_PROFILE_DIR:-profiles}"

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu not found. Install NVIDIA Nsight Compute on the CUDA machine." >&2
  exit 127
fi

mkdir -p "${profile_dir}"
export_name="${profile_dir}/qr_${module}_${suite}_${mode}_case${case_index}"

exec ncu \
  --set "${set_name}" \
  --target-processes all \
  --force-overwrite \
  --export "${export_name}" \
  --launch-skip "${launch_skip}" \
  --launch-count "${launch_count}" \
  --kernel-name "${kernel_name}" \
  -- \
  uv run --group practice python local_benchmark.py \
    --module "${module}" \
    --suite "${suite}" \
    --mode "${mode}" \
    --case-index "${case_index}" \
    --warmups "${warmups}" \
    --repeats "${repeats}" \
    --no-recheck
