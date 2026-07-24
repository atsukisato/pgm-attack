#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p \
  fig/table_generation_time_dup \
  fig/table_generation_time_upper_bound_dup \
  fig/table_mopt_upper_bound_dup \
  fig/table_mopt_upper_bound_real_system_dup \
  fig/table_mopt_upper_bound_small_dup \
  fig/table_mopt_upper_bound_various_epsilon_dup

python3 plot/print_table_mopt_upper_bound.py --duplicate-only
python3 plot/print_table_mopt_upper_bound_real_system.py --duplicate-only
python3 plot/print_table_mopt_upper_bound_small.py --duplicate-only
python3 plot/print_table_mopt_upper_bound_various_epsilon.py --duplicate-only
python3 plot/print_table_generation_time.py --duplicate-only
python3 plot/print_table_generation_time_upper_bound.py --duplicate-only
