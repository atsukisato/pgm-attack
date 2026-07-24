#!/bin/bash
set -euo pipefail

# Path setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIGS_DIR="configs/experiments"

CONFIG_FILES=(
    ### MAXIMIZE_MAXERROR ###
    "configs/experiments/poison_attack/maximize_maxerror_consec/baseline_n16.json"
    "configs/experiments/poison_attack/maximize_maxerror_dup_optimal/baseline_n16.json"
    "configs/experiments/poison_attack/maximize_maxerror_greedy/baseline_n16.json"
    "configs/experiments/poison_attack/maximize_maxerror_optimal/baseline_n16.json"
    "configs/experiments/poison_attack/maximize_maxerror_random/baseline_n16.json"
    "configs/experiments/poison_attack/maximize_maxerror_random_adjacent/baseline_n16.json"

    "configs/experiments/poison_attack/maximize_maxerror_consec/baseline.json"
    "configs/experiments/poison_attack/maximize_maxerror_dup_optimal/baseline.json"
    "configs/experiments/poison_attack/maximize_maxerror_greedy/baseline.json"
    "configs/experiments/poison_attack/maximize_maxerror_random/baseline.json"
    "configs/experiments/poison_attack/maximize_maxerror_random_adjacent/baseline.json"

    "configs/experiments/poison_attack/maximize_maxerror_consec/baseline_n1000.json"
    "configs/experiments/poison_attack/maximize_maxerror_dup_optimal/baseline_n1000.json"
    # "configs/experiments/poison_attack/maximize_maxerror_greedy/baseline_n1000.json" # take too long time
    "configs/experiments/poison_attack/maximize_maxerror_random/baseline_n1000.json"
    "configs/experiments/poison_attack/maximize_maxerror_random_adjacent/baseline_n1000.json"

    ### MINIMIZE_SEGMENT_LENGTH ###
    "configs/experiments/poison_attack/minimize_segment_length/baseline_epsilon2.json"
    "configs/experiments/poison_attack/minimize_segment_length/baseline_epsilon4.json"
    "configs/experiments/poison_attack/minimize_segment_length/baseline_epsilon8.json"
    "configs/experiments/poison_attack/minimize_segment_length/baseline_epsilon16.json"
    "configs/experiments/poison_attack/minimize_segment_length/baseline_epsilon32.json"
    "configs/experiments/poison_attack/minimize_segment_length/baseline_epsilon64.json"
    "configs/experiments/poison_attack/minimize_segment_length/baseline_epsilon128.json"

    "configs/experiments/poison_attack/minimize_segment_length_swing/baseline_epsilon2.json"
    "configs/experiments/poison_attack/minimize_segment_length_swing/baseline_epsilon4.json"
    "configs/experiments/poison_attack/minimize_segment_length_swing/baseline_epsilon8.json"
    "configs/experiments/poison_attack/minimize_segment_length_swing/baseline_epsilon16.json"
    "configs/experiments/poison_attack/minimize_segment_length_swing/baseline_epsilon32.json"
    "configs/experiments/poison_attack/minimize_segment_length_swing/baseline_epsilon64.json"
    "configs/experiments/poison_attack/minimize_segment_length_swing/baseline_epsilon128.json"

    "configs/experiments/poison_attack/minimize_segment_length_random/baseline_epsilon2.json"
    "configs/experiments/poison_attack/minimize_segment_length_random/baseline_epsilon4.json"
    "configs/experiments/poison_attack/minimize_segment_length_random/baseline_epsilon8.json"
    "configs/experiments/poison_attack/minimize_segment_length_random/baseline_epsilon16.json"
    "configs/experiments/poison_attack/minimize_segment_length_random/baseline_epsilon32.json"
    "configs/experiments/poison_attack/minimize_segment_length_random/baseline_epsilon64.json"
    "configs/experiments/poison_attack/minimize_segment_length_random/baseline_epsilon128.json"

    "configs/experiments/poison_attack/minimize_segment_length_random_adjacent/baseline_epsilon2.json"
    "configs/experiments/poison_attack/minimize_segment_length_random_adjacent/baseline_epsilon4.json"
    "configs/experiments/poison_attack/minimize_segment_length_random_adjacent/baseline_epsilon8.json"
    "configs/experiments/poison_attack/minimize_segment_length_random_adjacent/baseline_epsilon16.json"
    "configs/experiments/poison_attack/minimize_segment_length_random_adjacent/baseline_epsilon32.json"
    "configs/experiments/poison_attack/minimize_segment_length_random_adjacent/baseline_epsilon64.json"
    "configs/experiments/poison_attack/minimize_segment_length_random_adjacent/baseline_epsilon128.json"

    ## INJECT_POISONS_TO_MINIMIZE_SEGMENT_LENGTH ###
    "configs/experiments/poison_attack/inject_poisons_to_minimize_segment_length_random/baseline.json"
    "configs/experiments/poison_attack/inject_poisons_to_minimize_segment_length_random_adjacent/baseline.json"
    "configs/experiments/poison_attack/inject_poisons_to_minimize_segment_length_swing_lambda_with_theta/baseline_small.json"
    "configs/experiments/poison_attack/inject_poisons_to_minimize_segment_length_swing_lambda_with_theta/baseline.json"

    "configs/experiments/poison_attack/inject_poisons_to_minimize_segment_length_random/baseline_1M.json"
    "configs/experiments/poison_attack/inject_poisons_to_minimize_segment_length_random_adjacent/baseline_1M.json"
    "configs/experiments/poison_attack/inject_poisons_to_minimize_segment_length_swing_lambda_with_theta/baseline_1M.json"
    "configs/experiments/poison_attack/inject_poisons_to_minimize_segment_length_swing_lambda_with_theta/baseline_1M_sweep_mu.json"

    ### UPPER_BOUND FIX_W_PER_BLOCK ###
    "configs/experiments/upper_bound/fix_w_per_block/baseline_1M.json"
    "configs/experiments/upper_bound/fix_w_per_block/baseline_small.json"
    "configs/experiments/upper_bound/fix_w_per_block/baseline.json"

    ### Duplicate-key workloads only ###
    "configs/experiments/poison_attack/inject_poisons_to_minimize_segment_length_random/baseline_dup.json"
    "configs/experiments/poison_attack/inject_poisons_to_minimize_segment_length_random_adjacent/baseline_dup.json"
    "configs/experiments/poison_attack/inject_poisons_to_minimize_segment_length_swing_lambda_with_theta/baseline_dup.json"
)

for config_file in "${CONFIG_FILES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running experiment: ${config_file}"
    echo "=========================================="
    
    "${SCRIPT_DIR}/run_experiment.sh" "${config_file}"
    
    echo ""
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

