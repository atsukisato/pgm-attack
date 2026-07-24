#!/bin/bash
set -euo pipefail

# Path setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

BUILD_DIR="${BUILD_DIR:-build}"
BIN_DIR="${BUILD_DIR}/bin"
PREPROCESS_BIN="${BIN_DIR}/preprocess_mgbench"
MGBENCH_DIR="${PROJECT_ROOT}/data/mgbench"

# Check if binary exists
if [ ! -f "${PREPROCESS_BIN}" ]; then
    echo "Error: preprocess_mgbench binary not found: ${PREPROCESS_BIN}"
    echo "Please build the project first: ./scripts/build.sh"
    exit 1
fi

# Check if CSV files exist
if [ ! -f "${MGBENCH_DIR}/bench2.csv" ] || [ ! -f "${MGBENCH_DIR}/bench3.csv" ]; then
    echo "Error: bench2.csv and/or bench3.csv not found in ${MGBENCH_DIR}"
    echo "Please run ./scripts/download_mgbench.sh first"
    exit 1
fi

echo "=========================================="
echo "Preprocessing mgbench datasets"
echo "=========================================="
echo ""

# Preprocess bench2 (Unix timestamp in seconds)
if [ -f "${PROJECT_ROOT}/data/bench2_uint64" ]; then
    echo "Skipping bench2: data/bench2_uint64 already exists"
else
    bench2_n=$(( $(wc -l < "${MGBENCH_DIR}/bench2.csv") - 1 ))
    echo "Preprocessing bench2.csv -> data/bench2_uint64 (${bench2_n} rows) ..."
    "${PREPROCESS_BIN}" \
        --input "${MGBENCH_DIR}/bench2.csv" \
        --output "${PROJECT_ROOT}/data/bench2_uint64" \
        --unit seconds \
        --reserve "${bench2_n}"
fi

# Preprocess bench3 (Unix timestamp in milliseconds)
if [ -f "${PROJECT_ROOT}/data/bench3_uint64" ]; then
    echo "Skipping bench3: data/bench3_uint64 already exists"
else
    bench3_n=$(( $(wc -l < "${MGBENCH_DIR}/bench3.csv") - 1 ))
    echo "Preprocessing bench3.csv -> data/bench3_uint64 (${bench3_n} rows) ..."
    "${PREPROCESS_BIN}" \
        --input "${MGBENCH_DIR}/bench3.csv" \
        --output "${PROJECT_ROOT}/data/bench3_uint64" \
        --unit milliseconds \
        --reserve "${bench3_n}"
fi

echo ""
echo "Done. Output: data/bench2_uint64, data/bench3_uint64"
