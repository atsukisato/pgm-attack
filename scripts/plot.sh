#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p \
  fig/key_rank \
  fig/lambda_to_max_error \
  fig/lambda_to_segment_length \
  fig/lambda_to_mopt \
  fig/lambda_to_mopt_upper_bound \
  fig/mu_to_mopt

python3 plot/plot_key_rank.py
python3 plot/plot_lambda_to_max_error.py
python3 plot/plot_lambda_to_segment_length.py
python3 plot/plot_lambda_to_mopt.py
python3 plot/plot_lambda_to_mopt_upper_bound.py
python3 plot/plot_mu_to_mopt.py
