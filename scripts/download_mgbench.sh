#!/bin/bash
set -euo pipefail

# Path setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MGBENCH_DIR="${PROJECT_ROOT}/data/mgbench"

echo "=========================================="
echo "Downloading mgbench datasets"
echo "=========================================="
echo ""

mkdir -p "${MGBENCH_DIR}"
cd "${MGBENCH_DIR}"

# Download bench2
if [ ! -f bench2.csv ]; then
    echo "Downloading bench2.tar.gz ..."
    wget -q http://cs.brown.edu/people/acrotty/mgbench/bench2.tar.gz
    tar -xzf bench2.tar.gz
    rm -f bench2.tar.gz
    echo "Extracted bench2"
fi

# Download bench3
if [ ! -f bench3.csv ]; then
    echo "Downloading bench3.tar.gz ..."
    wget -q http://cs.brown.edu/people/acrotty/mgbench/bench3.tar.gz
    tar -xzf bench3.tar.gz
    rm -f bench3.tar.gz
    echo "Extracted bench3"
fi

# Handle case where tar extracts to subdirectory (e.g., bench2/bench2.csv)
if [ ! -f bench2.csv ] && [ -f bench2/bench2.csv ]; then
    mv bench2/bench2.csv bench2.csv
    rmdir bench2 2>/dev/null || true
fi
if [ ! -f bench3.csv ] && [ -f bench3/bench3.csv ]; then
    mv bench3/bench3.csv bench3.csv
    rmdir bench3 2>/dev/null || true
fi

echo ""
echo "Done. CSV files: ${MGBENCH_DIR}/bench2.csv, ${MGBENCH_DIR}/bench3.csv"
