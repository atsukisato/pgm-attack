#!/bin/bash

set -euo pipefail

IMAGE_NAME="pgm_poisoning"
IMAGE_TAG="latest"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run --rm -d \
    --cpus="8" \
    --cpuset-cpus="0-7" \
    --privileged \
    --cap-add SYS_ADMIN \
    --cap-add PERFMON \
    --security-opt seccomp=unconfined \
    -v "${ROOT_DIR}:/workspace" \
    ${IMAGE_NAME}:${IMAGE_TAG} \
    bash -c "cd /workspace && ./scripts/run_all_experiment.sh > /workspace/results/run_all.log"
