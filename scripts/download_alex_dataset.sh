#!/bin/bash
set -euo pipefail

# Path setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

ALEX_DIR="${PROJECT_ROOT}/data/alex"

echo "=========================================="
echo "Downloading alex datasets"
echo "=========================================="
echo ""

# Install gdown if not available
if ! command -v gdown &>/dev/null; then
    echo "gdown not found. Installing via pip ..."
    pip install gdown
fi

mkdir -p "${ALEX_DIR}"

# Download longitudes-200M.bin
if [ ! -f "${ALEX_DIR}/longitudes-200M.bin" ]; then
    echo "Downloading longitudes-200M.bin ..."
    gdown --id 1zc90sD6Pze8UM_XYDmNjzPLqmKly8jKl -O "${ALEX_DIR}/longitudes-200M.bin"
    echo "Downloaded longitudes-200M.bin"
else
    echo "longitudes-200M.bin already exists, skipping"
fi

# Download longlat-200M.bin
if [ ! -f "${ALEX_DIR}/longlat-200M.bin" ]; then
    echo "Downloading longlat-200M.bin ..."
    gdown --id 1mH-y_PcLQ6p8kgAz9SB7ME4KeYAfRfmR -O "${ALEX_DIR}/longlat-200M.bin"
    echo "Downloaded longlat-200M.bin"
else
    echo "longlat-200M.bin already exists, skipping"
fi

# Download ycsb-200M.bin
if [ ! -f "${ALEX_DIR}/ycsb-200M.bin" ]; then
    echo "Downloading ycsb-200M.bin ..."
    gdown --id 1Q89-v4FJLEwIKL3YY3oCeOEs0VUuv5bD -O "${ALEX_DIR}/ycsb-200M.bin"
    echo "Downloaded ycsb-200M.bin"
else
    echo "ycsb-200M.bin already exists, skipping"
fi

echo ""
echo "Done. Files in ${ALEX_DIR}:"
ls -la "${ALEX_DIR}"/*.bin 2>/dev/null || true
