#!/bin/bash
set -euo pipefail

# Path setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

BUILD_DIR="${BUILD_DIR:-build}"
BIN_DIR="${BUILD_DIR}/bin"
PREPROCESS_BIN="${BIN_DIR}/preprocess_alex_dataset"
ALEX_DIR="${PROJECT_ROOT}/data/alex"

# Check if binary exists
if [ ! -f "${PREPROCESS_BIN}" ]; then
    echo "Error: preprocess_alex_dataset binary not found: ${PREPROCESS_BIN}"
    echo "Please build the project first: ./scripts/build.sh"
    exit 1
fi

# Check if alex dataset files exist
if [ ! -f "${ALEX_DIR}/longitudes-200M.bin" ] || [ ! -f "${ALEX_DIR}/longlat-200M.bin" ] || [ ! -f "${ALEX_DIR}/ycsb-200M.bin" ]; then
    echo "Error: alex dataset files not found in ${ALEX_DIR}"
    echo "Please run ./scripts/download_alex_dataset.sh first"
    exit 1
fi

echo "=========================================="
echo "Preprocessing alex datasets"
echo "=========================================="
echo ""

# Preprocess longitudes (double -> uint64 bits)
if [ -f "${PROJECT_ROOT}/data/longitudes_uint64" ]; then
    echo "Skipping longitudes: data/longitudes_uint64 already exists"
else
    echo "Preprocessing longitudes-200M.bin -> data/longitudes_uint64 ..."
    "${PREPROCESS_BIN}" \
        --input "${ALEX_DIR}/longitudes-200M.bin" \
        --output "${PROJECT_ROOT}/data/longitudes_uint64" \
        --type double
fi

# Preprocess longlat (double -> uint64 bits)
if [ -f "${PROJECT_ROOT}/data/longlat_uint64" ]; then
    echo "Skipping longlat: data/longlat_uint64 already exists"
else
    echo "Preprocessing longlat-200M.bin -> data/longlat_uint64 ..."
    "${PREPROCESS_BIN}" \
        --input "${ALEX_DIR}/longlat-200M.bin" \
        --output "${PROJECT_ROOT}/data/longlat_uint64" \
        --type double
fi

# Preprocess ycsb (uint64)
if [ -f "${PROJECT_ROOT}/data/ycsb_uint64" ]; then
    echo "Skipping ycsb: data/ycsb_uint64 already exists"
else
    echo "Preprocessing ycsb-200M.bin -> data/ycsb_uint64 ..."
    "${PREPROCESS_BIN}" \
        --input "${ALEX_DIR}/ycsb-200M.bin" \
        --output "${PROJECT_ROOT}/data/ycsb_uint64" \
        --type uint64
fi

echo ""
echo "Done. Output: data/longitudes_uint64, data/longlat_uint64, data/ycsb_uint64"
