#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p \
  fig/table_generation_time \
  fig/table_generation_time_upper_bound \
  fig/table_mopt_upper_bound_real_system \
  fig/table_mopt_upper_bound \
  fig/table_mopt_upper_bound_small \
  fig/table_mopt_upper_bound_various_epsilon

python3 plot/print_table_mopt_upper_bound_real_system.py
python3 plot/print_table_mopt_upper_bound.py
python3 plot/print_table_mopt_upper_bound_small.py
python3 plot/print_table_mopt_upper_bound_various_epsilon.py
python3 plot/print_table_generation_time.py
python3 plot/print_table_generation_time_upper_bound.py
