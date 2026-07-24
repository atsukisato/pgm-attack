#!/bin/bash
set -euo pipefail

# Path setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

BUILD_DIR="${BUILD_DIR:-build}"
BIN_DIR="${BUILD_DIR}/bin"
GENERATE_SYNTHETIC_DATASET_BIN="${BIN_DIR}/generate_synthetic_dataset"

# Check if binary exists
if [ ! -f "${GENERATE_SYNTHETIC_DATASET_BIN}" ]; then
    echo "Error: generate_synthetic_dataset binary not found: ${GENERATE_SYNTHETIC_DATASET_BIN}"
    echo "Please build the project first: ./scripts/build.sh"
    exit 1
fi

# If arguments provided, pass through to binary
if [ $# -gt 0 ]; then
    exec "${GENERATE_SYNTHETIC_DATASET_BIN}" "$@"
fi

# Default: generate all common presets (200M entries each)
echo "=========================================="
echo "Generating synthetic datasets (200M entries each)"
echo "=========================================="
echo ""

N=200000000
TARGET_RANGE=9223372036854775808  # 2^63
FAILED=0

run_generate() {
    local output_name="$1"
    shift
    local target="${PROJECT_ROOT}/data/${output_name}_uint64"
    if [ -f "${target}" ]; then
        echo "Skipping ${output_name}: ${target} already exists"
    else
        if ! "${GENERATE_SYNTHETIC_DATASET_BIN}" "$@" --output "${output_name}"; then
            FAILED=1
        fi
    fi
    echo ""
}

# Zipf
run_generate zipf_s1_200M zipf "${N}" "${N}" 1.0

# Uniform
run_generate uniform_range2pow63_200M uniform "${N}" "${TARGET_RANGE}"

# Lognormal
run_generate lognormal_sigma1_range2pow63_200M lognormal "${N}" 0.0 1.0 "${TARGET_RANGE}"

# Normal (N(0, 1) standard normal)
run_generate normal_mu0_sigma1_range2pow63_200M normal "${N}" 0.0 1.0 "${TARGET_RANGE}"

if [ ${FAILED} -eq 0 ]; then
    echo "=========================================="
    echo "Successfully generated all datasets!"
    echo "=========================================="
else
    echo "=========================================="
    echo "Warning: Some datasets may have failed. Check the output above."
    echo "=========================================="
    exit 1
fi
