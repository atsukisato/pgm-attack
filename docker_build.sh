#!/bin/bash

set -euo pipefail

IMAGE_NAME="pgm_poisoning"
IMAGE_TAG="latest"

docker build -t $IMAGE_NAME:$IMAGE_TAG .
