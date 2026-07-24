#!/bin/bash
set -euo pipefail

# Specify experiment configuration file
EXP_CONFIG="${1:-}"

if [ -z "${EXP_CONFIG}" ]; then
    echo "Usage: $0 <experiment_config.json>"
    exit 1
fi

# Path setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

BUILD_DIR="${BUILD_DIR:-build}"
BIN_DIR="${BUILD_DIR}/bin"

# Check for jq
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not found. Please install jq."
    exit 1
fi

# Load configuration file
if [ ! -f "${EXP_CONFIG}" ]; then
    echo "Error: Experiment config not found: ${EXP_CONFIG}"
    exit 1
fi

EXP_CONFIG_JSON=$(cat "${EXP_CONFIG}")

# ============================================================================
# Helper Functions
# ============================================================================

# Expand array parameters
expand_array() {
    local key="$1"
    local config="$2"
    if echo "${config}" | jq -e "has(\"${key}\")" > /dev/null 2>&1; then
        local type=$(echo "${config}" | jq -r ".\"${key}\" | type")
        if [ "${type}" = "array" ]; then
            echo "${config}" | jq -r ".\"${key}\"[]"
        else
            echo "${config}" | jq -r ".\"${key}\""
        fi
    else
        echo ""
    fi
}

# Execute pipeline step
execute_pipeline_step() {
    local step="$1"
    local dataset_path="$2"
    local poisons_path="$3"
    local lambda="$4"
    local epsilon="$5"
    local seed="$6"
    local run_dir="$7"
    local lambda_per_segment="$8"
    local n_keys="${9:-}"
    local intercept_candidate_num="${10:-}"
    local mu="${11:-}"
    local allow_duplicates="${12:-false}"
    
    case "${step}" in
        "generate_poisons")
            local attack_method=$(echo "${EXP_CONFIG_JSON}" | jq -r '.attack.method // .attack.type // "random"')
            # maximize_maxerror_* methods use generate_poisons_maxerror (require --seed and --n)
            if [[ "${attack_method}" == maximize_maxerror_* ]]; then
                if [ -z "${n_keys}" ]; then
                    echo "Error: ${attack_method} requires n (keys count)"
                    exit 1
                fi
                if [ -z "${seed}" ]; then
                    echo "Error: ${attack_method} requires seed"
                    exit 1
                fi
                if [ -f "${poisons_path}" ] && [ -f "${run_dir}/poisons_info.json" ]; then
                    echo "  Skipping: Output files already exist (poisons.bin, poisons_info.json)"
                    return 0
                fi
                local maxerror_cmd_args=(--keys "${dataset_path}" --output "${poisons_path}" --lambda "${lambda}" --seed "${seed}" --n "${n_keys}" --method "${attack_method}")
                if [ "${allow_duplicates}" = "true" ]; then
                    maxerror_cmd_args+=(--allow-duplicates "true")
                fi
                echo "${BIN_DIR}/generate_poisons_maxerror ${maxerror_cmd_args[*]}"
                "${BIN_DIR}/generate_poisons_maxerror" "${maxerror_cmd_args[@]}" \
                    > "${run_dir}/poisons_info.json"
                return 0
            fi
            # minimize_segment_length_* methods use generate_poisons_segment_length (require --seed, --epsilon, --lambda-per-segment)
            if [[ "${attack_method}" == "minimize_segment_length" ]] || [[ "${attack_method}" == "minimize_segment_length_swing" ]] || [[ "${attack_method}" == "minimize_segment_length_random" ]] || [[ "${attack_method}" == "minimize_segment_length_random_adjacent" ]]; then
                if [ -z "${seed}" ]; then
                    echo "Error: ${attack_method} requires seed"
                    exit 1
                fi
                if [ -z "${epsilon}" ]; then
                    echo "Error: ${attack_method} requires epsilon"
                    exit 1
                fi
                if [ -z "${lambda_per_segment}" ]; then
                    echo "Error: ${attack_method} requires lambda_per_segment"
                    exit 1
                fi
                if [ -f "${poisons_path}" ] && [ -f "${run_dir}/poisons_info.json" ]; then
                    echo "  Skipping: Output files already exist (poisons.bin, poisons_info.json)"
                    return 0
                fi
                local seg_cmd_args=(--keys "${dataset_path}" --output "${poisons_path}" --seed "${seed}" --epsilon "${epsilon}" --lambda-per-segment "${lambda_per_segment}" --method "${attack_method}" --allow-duplicates "${allow_duplicates}")
                if [ -n "${intercept_candidate_num}" ] && [[ "${attack_method}" == "minimize_segment_length_swing" ]]; then
                    seg_cmd_args+=(--intercept-candidate-num "${intercept_candidate_num}")
                fi
                echo "${BIN_DIR}/generate_poisons_segment_length ${seg_cmd_args[*]}"
                "${BIN_DIR}/generate_poisons_segment_length" "${seg_cmd_args[@]}" \
                    > "${run_dir}/poisons_info.json"
                return 0
            fi
            # Check if seed is required (inject_poisons_to_minimize_segment_length_swing_lambda_with_theta does not require it)
            if [ "${attack_method}" != "inject_poisons_to_minimize_segment_length_swing_lambda_with_theta" ]; then
                if [ -z "${seed}" ]; then
                    echo "Error: generate_poisons requires seed for ${attack_method} method"
                    exit 1
                fi
            fi
            # Check if output files already exist
            if [ -f "${poisons_path}" ] && [ -f "${run_dir}/poisons_info.json" ]; then
                echo "  Skipping: Output files already exist (poisons.bin, poisons_info.json)"
                return 0
            fi
            local cmd_args=(
                --keys "${dataset_path}"
                --output "${poisons_path}"
                --method "${attack_method}"
            )
            cmd_args+=(--lambda "${lambda}")
            # Add --seed only if provided (required for most methods; optional for swing_lambda_with_theta)
            if [ -n "${seed}" ]; then
                cmd_args+=(--seed "${seed}")
            fi
            # Add --epsilon for methods that require it
            if [ "${attack_method}" = "inject_poisons_to_minimize_segment_length_swing_lambda_with_theta" ] || \
               [ "${attack_method}" = "inject_poisons_to_minimize_segment_length_random" ] || \
               [ "${attack_method}" = "inject_poisons_to_minimize_segment_length_random_adjacent" ]; then
                if [ -z "${epsilon}" ]; then
                    echo "Error: ${attack_method} method requires epsilon"
                    exit 1
                fi
                cmd_args+=(--epsilon "${epsilon}")
            fi
            # Add --mu for inject_poisons_to_minimize_segment_length_swing_lambda_with_theta when provided
            if [ -n "${mu}" ] && [[ "${attack_method}" == "inject_poisons_to_minimize_segment_length_swing_lambda_with_theta" ]]; then
                cmd_args+=(--mu "${mu}")
            fi
            # Add --n and --seed when sampling 1M from larger dataset (dataset config has n)
            if [ -n "${n_keys}" ] && [ -n "${seed}" ]; then
                cmd_args+=(--n "${n_keys}" --seed "${seed}")
            fi
            if [ "${allow_duplicates}" = "true" ]; then
                cmd_args+=(--allow-duplicates "true")
            fi
            echo "${BIN_DIR}/generate_poisons ${cmd_args[@]}"
            "${BIN_DIR}/generate_poisons" "${cmd_args[@]}" > "${run_dir}/poisons_info.json"
            ;;
        "benchmark_pgm")
            # Check if output file already exists
            if [ -f "${run_dir}/benchmark_pgm.json" ]; then
                echo "  Skipping: Output file already exists (benchmark_pgm.json)"
                return 0
            fi
            # Check if SKIP_PERF environment variable is set or if we're in Docker on Mac
            local skip_perf_flag=""
            if [ "${SKIP_PERF:-}" = "1" ] || [ "${SKIP_PERF:-}" = "true" ]; then
                skip_perf_flag="--skip-perf"
            fi
            local cmd_args=(
                --keys "${dataset_path}"
                --poisons "${poisons_path}"
                --epsilon "${epsilon}"
            )
            if [ -n "${skip_perf_flag}" ]; then
                cmd_args+=("${skip_perf_flag}")
            fi
            # Add --n and --seed when sampling 1M from larger dataset (dataset config has n)
            if [ -n "${n_keys}" ] && [ -n "${seed}" ]; then
                cmd_args+=(--n "${n_keys}" --seed "${seed}")
            fi
            echo "${BIN_DIR}/benchmark_pgm ${cmd_args[@]}"
            "${BIN_DIR}/benchmark_pgm" "${cmd_args[@]}" > "${run_dir}/benchmark_pgm.json"
            ;;
        "benchmark_fiting_tree")
            if [ -f "${run_dir}/benchmark_fiting_tree.json" ]; then
                echo "  Skipping: Output file already exists (benchmark_fiting_tree.json)"
                return 0
            fi
            local skip_perf_flag_ft=""
            if [ "${SKIP_PERF:-}" = "1" ] || [ "${SKIP_PERF:-}" = "true" ]; then
                skip_perf_flag_ft="--skip-perf"
            fi
            local cmd_args_ft=(
                --keys "${dataset_path}"
                --poisons "${poisons_path}"
                --epsilon "${epsilon}"
            )
            if [ -n "${skip_perf_flag_ft}" ]; then
                cmd_args_ft+=("${skip_perf_flag_ft}")
            fi
            if [ -n "${n_keys}" ] && [ -n "${seed}" ]; then
                cmd_args_ft+=(--n "${n_keys}" --seed "${seed}")
            fi
            echo "${BIN_DIR}/benchmark_fiting_tree ${cmd_args_ft[@]}"
            "${BIN_DIR}/benchmark_fiting_tree" "${cmd_args_ft[@]}" > "${run_dir}/benchmark_fiting_tree.json"
            ;;
        "benchmark_radix_spline")
            if [ -f "${run_dir}/benchmark_radix_spline.json" ]; then
                echo "  Skipping: Output file already exists (benchmark_radix_spline.json)"
                return 0
            fi
            local skip_perf_flag_rs=""
            if [ "${SKIP_PERF:-}" = "1" ] || [ "${SKIP_PERF:-}" = "true" ]; then
                skip_perf_flag_rs="--skip-perf"
            fi
            local cmd_args_rs=(
                --keys "${dataset_path}"
                --poisons "${poisons_path}"
                --epsilon "${epsilon}"
            )
            if [ -n "${skip_perf_flag_rs}" ]; then
                cmd_args_rs+=("${skip_perf_flag_rs}")
            fi
            if [ -n "${n_keys}" ] && [ -n "${seed}" ]; then
                cmd_args_rs+=(--n "${n_keys}" --seed "${seed}")
            fi
            local radix_bits
            radix_bits=$(echo "${EXP_CONFIG_JSON}" | jq -r '(.radix_spline // {}).num_radix_bits // empty')
            if [ -n "${radix_bits}" ] && [ "${radix_bits}" != "null" ]; then
                cmd_args_rs+=(--num-radix-bits "${radix_bits}")
            fi
            echo "${BIN_DIR}/benchmark_radix_spline ${cmd_args_rs[@]}"
            "${BIN_DIR}/benchmark_radix_spline" "${cmd_args_rs[@]}" > "${run_dir}/benchmark_radix_spline.json"
            ;;
        "compute_upper_bound")
            # Check if output file already exists
            if [ -f "${run_dir}/upper_bound.json" ]; then
                echo "  Skipping: Output file already exists (upper_bound.json)"
                return 0
            fi
            local method=$(echo "${EXP_CONFIG_JSON}" | jq -r '.bound.method // "simple"')
            local cmd_args=(--keys "${dataset_path}" --lambda "${lambda}" --epsilon "${epsilon}" --method "${method}")
            # Add --n and --seed when sampling 1M from larger dataset (dataset config has n)
            if [ -n "${n_keys}" ] && [ -n "${seed}" ]; then
                cmd_args+=(--n "${n_keys}" --seed "${seed}")
            fi
            echo "${BIN_DIR}/compute_upper_bound ${cmd_args[*]}"
            "${BIN_DIR}/compute_upper_bound" "${cmd_args[@]}" \
                > "${run_dir}/upper_bound.json" 2>&1 || echo "Warning: compute_upper_bound may not be fully implemented" >&2
            ;;
        *)
            echo "Warning: Unknown pipeline step: ${step}"
            ;;
    esac
}

# Execute pipeline
execute_pipeline() {
    local pipeline="$1"
    local dataset_path="$2"
    local poisons_path="$3"
    local lambda="$4"
    local epsilon="$5"
    local seed="$6"
    local run_dir="$7"
    local lambda_per_segment="$8"
    local n_keys="${9:-}"
    local intercept_candidate_num="${10:-}"
    local mu="${11:-}"
    local allow_duplicates="${12:-false}"
    
    local pipeline_array=($(echo "${pipeline}"))
    local total_steps=${#pipeline_array[@]}
    local step_num=1
    
    for step in "${pipeline_array[@]}"; do
        echo ""
        echo "[${step_num}/${total_steps}] Running ${step}..."
        execute_pipeline_step "${step}" "${dataset_path}" "${poisons_path}" "${lambda}" "${epsilon}" "${seed}" "${run_dir}" "${lambda_per_segment}" "${n_keys}" "${intercept_candidate_num}" "${mu}" "${allow_duplicates}"
        step_num=$((step_num + 1))
    done
}

# Extract experiment type from config file path
get_experiment_type() {
    local config_path="$1"
    # Extract experiment type from path: configs/experiments/poison_attack/... -> poison_attack
    local exp_type=$(echo "${config_path}" | sed -n 's|.*configs/experiments/\([^/]*\)/.*|\1|p')
    
    # Map experiment types to result directory names
    case "${exp_type}" in
        "poison_attack")
            echo "poisoning"
            ;;
        "upper_bound")
            echo "upper_bound"
            ;;
        *)
            echo "${exp_type:-experiment}"
            ;;
    esac
}

# Extract method from config file path
get_experiment_method() {
    local config_path="$1"
    # Extract method from path
    # or configs/experiments/upper_bound/simple/... -> simple
    echo "${config_path}" | sed -n 's|.*configs/experiments/[^/]*/\([^/]*\)/.*|\1|p'
}

# Run single experiment
# dataset_has_n: when true, n_keys is from dataset config (1M sampling) - use seed hierarchy
run_single_experiment() {
    local dataset="$1"
    local dataset_config="$2"
    local dataset_path="$3"
    local seed="$4"
    local lambda="$5"
    local epsilon="$6"
    local pipeline="$7"
    local lambda_per_segment="$8"
    local n_keys="${9:-}"
    local intercept_candidate_num="${10:-}"
    local mu="${11:-}"
    local dataset_has_n="${12:-}"
    local allow_duplicates="${13:-false}"
    
    # Get experiment type and method from config file path
    local exp_type=$(get_experiment_type "${EXP_CONFIG}")
    local method=$(get_experiment_method "${EXP_CONFIG}")
    # attack_maxerror (maximize_maxerror_*) from poison_attack results go to poisoning_maxerror
    if [[ "${method}" == maximize_maxerror_* ]] && [[ "${exp_type}" == "poisoning" ]]; then
        exp_type="poisoning_maxerror"
    fi
    # minimize_segment_length methods results go to poisoning_segment_length
    if [[ "${method}" == "minimize_segment_length" ]] || [[ "${method}" == "minimize_segment_length_swing" ]] || [[ "${method}" == "minimize_segment_length_random" ]] || [[ "${method}" == "minimize_segment_length_random_adjacent" ]]; then
        exp_type="poisoning_segment_length"
    fi
    # Build result directory path: results/poisoning/random/dataset/lambda100/epsilon64/seed0/
    # For 1M datasets (dataset_has_n): dataset/seed{S}/lambda{N}/epsilon{E}/mu{M}/
    # maximize_maxerror_*: dataset/lambda{N}/epsilon{E}/n{n}/seed{seed}/
    local run_dir="results/${exp_type}/${method}/${dataset}"
    # When dataset has n (1M sampling), put seed before lambda for aggregation
    if [ "${dataset_has_n}" = "true" ] && [ -n "${seed}" ]; then
        run_dir="${run_dir}/seed${seed}"
    fi
    if [ "${method}" != "minimize_segment_length" ] && \
       [ "${method}" != "minimize_segment_length_swing" ] && \
       [ "${method}" != "minimize_segment_length_random" ] && \
       [ "${method}" != "minimize_segment_length_random_adjacent" ]; then
        run_dir="${run_dir}/lambda${lambda}"
    fi
    run_dir="${run_dir}/epsilon${epsilon}"
    if [ -n "${lambda_per_segment}" ]; then
        run_dir="${run_dir}/lambda_per_segment${lambda_per_segment}"
    fi
    if [ -n "${intercept_candidate_num}" ]; then
        run_dir="${run_dir}/intercept_candidate_num${intercept_candidate_num}"
    fi
    if [ -n "${mu}" ]; then
        run_dir="${run_dir}/mu${mu}"
    fi
    # maximize_maxerror uses n/seed at end; dataset_has_n uses seed hierarchy (no n/seed at end)
    if [ -n "${n_keys}" ] && [ "${dataset_has_n}" != "true" ]; then
        run_dir="${run_dir}/n${n_keys}"
    fi
    if [ -n "${seed}" ] && [ "${dataset_has_n}" != "true" ]; then
        run_dir="${run_dir}/seed${seed}"
    fi
    mkdir -p "${run_dir}"
    
    # Display execution info
    echo ""
    echo "=========================================="
    echo "Running experiment"
    echo "=========================================="
    echo "Type: ${exp_type}"
    echo "Method: ${method}"
    echo "Dataset: ${dataset}"
    if [ "${method}" != "minimize_segment_length" ] && \
       [ "${method}" != "minimize_segment_length_swing" ] && \
       [ "${method}" != "minimize_segment_length_random" ] && \
       [ "${method}" != "minimize_segment_length_random_adjacent" ]; then
        echo "Lambda: ${lambda}"
    fi
    echo "Epsilon: ${epsilon}"
    [ -n "${lambda_per_segment}" ] && echo "Lambda per segment: ${lambda_per_segment}"
    [ -n "${intercept_candidate_num}" ] && echo "Intercept candidate num: ${intercept_candidate_num}"
    [ -n "${mu}" ] && echo "Mu: ${mu}"
    [ -n "${n_keys}" ] && echo "N (keys): ${n_keys}"
    [ -n "${seed}" ] && echo "Seed: ${seed}"
    echo "Output: ${run_dir}"
    echo "=========================================="
    
    # Execute pipeline
    local poisons_path="${run_dir}/poisons.bin"
    execute_pipeline "${pipeline}" "${dataset_path}" "${poisons_path}" "${lambda}" "${epsilon}" "${seed}" "${run_dir}" "${lambda_per_segment}" "${n_keys}" "${intercept_candidate_num}" "${mu}" "${allow_duplicates}"
    
    echo "Results saved in: ${run_dir}"
}

# ============================================================================
# Main Processing
# ============================================================================

# Build the project
${SCRIPT_DIR}/build.sh

# Load dataset configuration
DATASET_NAME=$(echo "${EXP_CONFIG_JSON}" | jq -r '.dataset // .datasets[0]')
if echo "${EXP_CONFIG_JSON}" | jq -e 'has("datasets")' > /dev/null 2>&1; then
    DATASETS=$(echo "${EXP_CONFIG_JSON}" | jq -r '.datasets[]')
else
    DATASETS="${DATASET_NAME}"
fi

# Get array parameters
SEEDS=$(expand_array "seeds" "${EXP_CONFIG_JSON}")
[ -z "${SEEDS}" ] && SEEDS=$(expand_array "seed" "${EXP_CONFIG_JSON}")

LAMBDAS=$(expand_array "lambdas" "${EXP_CONFIG_JSON}")
[ -z "${LAMBDAS}" ] && LAMBDAS=$(expand_array "lambda" "${EXP_CONFIG_JSON}")

EPSILONS=$(expand_array "epsilons" "${EXP_CONFIG_JSON}")
[ -z "${EPSILONS}" ] && EPSILONS=$(expand_array "epsilon" "${EXP_CONFIG_JSON}")

LAMBDA_PER_SEGMENTS=$(expand_array "lambda_per_segments" "${EXP_CONFIG_JSON}")
[ -z "${LAMBDA_PER_SEGMENTS}" ] && LAMBDA_PER_SEGMENTS=$(expand_array "lambda_per_segment" "${EXP_CONFIG_JSON}")

N_VALUES=$(expand_array "n" "${EXP_CONFIG_JSON}")
[ -z "${N_VALUES}" ] && N_VALUES=$(expand_array "keys_count" "${EXP_CONFIG_JSON}")
[ -z "${N_VALUES}" ] && N_VALUES=$(expand_array "key_count" "${EXP_CONFIG_JSON}")

INTERCEPT_CANDIDATE_NUMS=$(expand_array "intercept_candidate_nums" "${EXP_CONFIG_JSON}")
[ -z "${INTERCEPT_CANDIDATE_NUMS}" ] && INTERCEPT_CANDIDATE_NUMS=$(expand_array "intercept_candidate_num" "${EXP_CONFIG_JSON}")

MUS=$(expand_array "mus" "${EXP_CONFIG_JSON}")
[ -z "${MUS}" ] && MUS=$(expand_array "mu" "${EXP_CONFIG_JSON}")

# Support paired epsilon and lambda_per_segment values if provided
# This allows configs to specify per-experiment (epsilon, lambda_per_segment) tuples
EPSILON_LAMBDA_PAIRS_ARRAY=()
if echo "${EXP_CONFIG_JSON}" | jq -e 'has("epsilon_and_lambda_per_segment_pairs")' > /dev/null 2>&1; then
    # Read each pair as a compact JSON array string (e.g. "[64,16]") into a bash array
    mapfile -t EPSILON_LAMBDA_PAIRS_ARRAY < <(echo "${EXP_CONFIG_JSON}" | jq -c '.epsilon_and_lambda_per_segment_pairs[]')
fi

# Set default values
[ -z "${EPSILONS}" ] && EPSILONS="64"

# Get pipeline
PIPELINE=$(echo "${EXP_CONFIG_JSON}" | jq -r '.pipeline // ["generate_poisons", "benchmark_pgm", "benchmark_fiting_tree", "benchmark_radix_spline", "compute_upper_bound"] | .[]' | tr '\n' ' ')

# Process SEEDS (empty array if not specified)
if [ -z "${SEEDS}" ]; then
    SEEDS_ARRAY=("")
else
    SEEDS_ARRAY=(${SEEDS})
fi

# Process LAMBDA_PER_SEGMENTS (empty array if not specified)
if [ -z "${LAMBDA_PER_SEGMENTS}" ]; then
    LAMBDA_PER_SEGMENTS_ARRAY=("")
else
    LAMBDA_PER_SEGMENTS_ARRAY=(${LAMBDA_PER_SEGMENTS})
fi

# Use paired epsilon/lambda_per_segment list if provided
USE_EPS_LAMBDA_PAIRS=false
if [ ${#EPSILON_LAMBDA_PAIRS_ARRAY[@]} -gt 0 ]; then
    USE_EPS_LAMBDA_PAIRS=true
fi

# Get attack method to determine requirements
ATTACK_METHOD=$(echo "${EXP_CONFIG_JSON}" | jq -r '.attack.method // .attack.type // "random"')
METHOD_FOR_REQUIREMENTS="${ATTACK_METHOD}"

NEEDS_LAMBDA=true
if [ "${METHOD_FOR_REQUIREMENTS}" = "minimize_segment_length" ] || \
   [ "${METHOD_FOR_REQUIREMENTS}" = "minimize_segment_length_swing" ] || \
   [ "${METHOD_FOR_REQUIREMENTS}" = "minimize_segment_length_random" ] || \
   [ "${METHOD_FOR_REQUIREMENTS}" = "minimize_segment_length_random_adjacent" ]; then
    NEEDS_LAMBDA=false
fi

# maximize_maxerror_* require n (keys count) and seed
NEEDS_N=false
case "${METHOD_FOR_REQUIREMENTS}" in
    maximize_maxerror_*) NEEDS_N=true ;;
esac

# Process N_VALUES (empty array if not specified)
if [ -z "${N_VALUES}" ]; then
    N_VALUES_ARRAY=("")
else
    N_VALUES_ARRAY=(${N_VALUES})
fi

# Process INTERCEPT_CANDIDATE_NUMS (empty array if not specified)
if [ -z "${INTERCEPT_CANDIDATE_NUMS}" ]; then
    INTERCEPT_CANDIDATE_NUMS_ARRAY=("")
else
    INTERCEPT_CANDIDATE_NUMS_ARRAY=(${INTERCEPT_CANDIDATE_NUMS})
fi

# Process MUS (empty array if not specified)
if [ -z "${MUS}" ]; then
    MUS_ARRAY=("")
else
    MUS_ARRAY=(${MUS})
fi

# Generate Cartesian product and execute experiments for each combination
for DATASET in ${DATASETS}; do
    DATASET_CONFIG=$(cat "configs/datasets/${DATASET}.json")
    DATASET_PATH=$(echo "${DATASET_CONFIG}" | jq -r '.path')
    DATASET_N=$(echo "${DATASET_CONFIG}" | jq -r '.n // empty')
    DATASET_HAS_N=""
    [ -n "${DATASET_N}" ] && DATASET_HAS_N="true"
    # allow_duplicates: experiment config overrides dataset config if present
    EXP_ALLOW_DUPLICATES=$(echo "${EXP_CONFIG_JSON}" | jq -r '.allow_duplicates // .attack.allow_duplicates // empty')
    if [ -n "${EXP_ALLOW_DUPLICATES}" ]; then
        ALLOW_DUPLICATES="${EXP_ALLOW_DUPLICATES}"
    else
        ALLOW_DUPLICATES=$(echo "${DATASET_CONFIG}" | jq -r '.allow_duplicates // false')
    fi
    
    if [ ! -f "${DATASET_PATH}" ]; then
        echo "Warning: Dataset file not found: ${DATASET_PATH}, skipping..."
        continue
    fi
    
    # When dataset has n (1M sampling), require seeds
    if [ -n "${DATASET_HAS_N}" ] && [ -z "${SEEDS}" ]; then
        echo "Error: Dataset ${DATASET} has n=${DATASET_N} but no seeds specified in experiment config"
        exit 1
    fi
    
    # Per-dataset lambdas: use dataset_lambdas[dataset] if present, else global LAMBDAS
    if [ "${NEEDS_LAMBDA}" = "true" ] && echo "${EXP_CONFIG_JSON}" | jq -e --arg d "${DATASET}" 'has("dataset_lambdas") and (.dataset_lambdas | has($d))' > /dev/null 2>&1; then
        LAMBDAS_TO_USE=$(echo "${EXP_CONFIG_JSON}" | jq -r --arg d "${DATASET}" '.dataset_lambdas[$d] | if type == "array" then .[] else . end')
    else
        LAMBDAS_TO_USE="${LAMBDAS}"
    fi
    
    # Per-dataset mus: use dataset_mus[dataset] if present, else global MUS
    if echo "${EXP_CONFIG_JSON}" | jq -e --arg d "${DATASET}" 'has("dataset_mus") and (.dataset_mus | has($d))' > /dev/null 2>&1; then
        MUS_TO_USE=$(echo "${EXP_CONFIG_JSON}" | jq -r --arg d "${DATASET}" '.dataset_mus[$d] | if type == "array" then .[] else . end')
    else
        MUS_TO_USE="${MUS}"
    fi
    # When mus is empty (e.g. bounds experiments), loop once with empty MU
    if [ -z "${MUS_TO_USE}" ]; then
        MUS_ARRAY=("")
    else
        MUS_ARRAY=(${MUS_TO_USE})
    fi

    for SEED in "${SEEDS_ARRAY[@]}"; do
        if [ "${NEEDS_N}" = "true" ]; then
            for N_KEYS in "${N_VALUES_ARRAY[@]}"; do
                [ -z "${N_KEYS}" ] && continue
                if [ "${NEEDS_LAMBDA}" = "true" ]; then
                    for LAMBDA in ${LAMBDAS_TO_USE}; do
                        if [ "${USE_EPS_LAMBDA_PAIRS}" = "true" ]; then
                            for PAIR_JSON in "${EPSILON_LAMBDA_PAIRS_ARRAY[@]}"; do
                                EPSILON=$(echo "${PAIR_JSON}" | jq -r '.[0]')
                                LAMBDA_PER_SEGMENT=$(echo "${PAIR_JSON}" | jq -r '.[1]')
                                for INTERCEPT_CANDIDATE_NUM in "${INTERCEPT_CANDIDATE_NUMS_ARRAY[@]}"; do
                                    for MU in "${MUS_ARRAY[@]}"; do
                                        run_single_experiment \
                                            "${DATASET}" "${DATASET_CONFIG}" "${DATASET_PATH}" \
                                            "${SEED}" "${LAMBDA}" "${EPSILON}" "${PIPELINE}" "${LAMBDA_PER_SEGMENT}" "${N_KEYS}" "${INTERCEPT_CANDIDATE_NUM}" "${MU}" "" "${ALLOW_DUPLICATES}"
                                    done
                                done
                            done
                        else
                            for EPSILON in ${EPSILONS}; do
                                for LAMBDA_PER_SEGMENT in "${LAMBDA_PER_SEGMENTS_ARRAY[@]}"; do
                                    for INTERCEPT_CANDIDATE_NUM in "${INTERCEPT_CANDIDATE_NUMS_ARRAY[@]}"; do
                                        for MU in "${MUS_ARRAY[@]}"; do
                                            run_single_experiment \
                                                "${DATASET}" "${DATASET_CONFIG}" "${DATASET_PATH}" \
                                                "${SEED}" "${LAMBDA}" "${EPSILON}" "${PIPELINE}" "${LAMBDA_PER_SEGMENT}" "${N_KEYS}" "${INTERCEPT_CANDIDATE_NUM}" "${MU}" "" "${ALLOW_DUPLICATES}"
                                        done
                                    done
                                done
                            done
                        fi
                    done
                else
                    if [ "${USE_EPS_LAMBDA_PAIRS}" = "true" ]; then
                        for PAIR_JSON in "${EPSILON_LAMBDA_PAIRS_ARRAY[@]}"; do
                            EPSILON=$(echo "${PAIR_JSON}" | jq -r '.[0]')
                            LAMBDA_PER_SEGMENT=$(echo "${PAIR_JSON}" | jq -r '.[1]')
                            for INTERCEPT_CANDIDATE_NUM in "${INTERCEPT_CANDIDATE_NUMS_ARRAY[@]}"; do
                                for MU in "${MUS_ARRAY[@]}"; do
                                    run_single_experiment \
                                        "${DATASET}" "${DATASET_CONFIG}" "${DATASET_PATH}" \
                                        "${SEED}" "" "${EPSILON}" "${PIPELINE}" "${LAMBDA_PER_SEGMENT}" "${N_KEYS}" "${INTERCEPT_CANDIDATE_NUM}" "${MU}" "" "${ALLOW_DUPLICATES}"
                                done
                            done
                        done
                    else
                        for EPSILON in ${EPSILONS}; do
                            for LAMBDA_PER_SEGMENT in "${LAMBDA_PER_SEGMENTS_ARRAY[@]}"; do
                                for INTERCEPT_CANDIDATE_NUM in "${INTERCEPT_CANDIDATE_NUMS_ARRAY[@]}"; do
                                    for MU in "${MUS_ARRAY[@]}"; do
                                        run_single_experiment \
                                            "${DATASET}" "${DATASET_CONFIG}" "${DATASET_PATH}" \
                                            "${SEED}" "" "${EPSILON}" "${PIPELINE}" "${LAMBDA_PER_SEGMENT}" "${N_KEYS}" "${INTERCEPT_CANDIDATE_NUM}" "${MU}" "" "${ALLOW_DUPLICATES}"
                                    done
                                done
                            done
                        done
                    fi
                fi
            done
        else
            if [ "${NEEDS_LAMBDA}" = "true" ]; then
                for LAMBDA in ${LAMBDAS_TO_USE}; do
                    if [ "${USE_EPS_LAMBDA_PAIRS}" = "true" ]; then
                        for PAIR_JSON in "${EPSILON_LAMBDA_PAIRS_ARRAY[@]}"; do
                            EPSILON=$(echo "${PAIR_JSON}" | jq -r '.[0]')
                            LAMBDA_PER_SEGMENT=$(echo "${PAIR_JSON}" | jq -r '.[1]')
                            for INTERCEPT_CANDIDATE_NUM in "${INTERCEPT_CANDIDATE_NUMS_ARRAY[@]}"; do
                                for MU in "${MUS_ARRAY[@]}"; do
                                        run_single_experiment \
                                            "${DATASET}" "${DATASET_CONFIG}" "${DATASET_PATH}" \
                                            "${SEED}" "${LAMBDA}" "${EPSILON}" "${PIPELINE}" "${LAMBDA_PER_SEGMENT}" "${DATASET_N:-}" "${INTERCEPT_CANDIDATE_NUM}" "${MU}" "${DATASET_HAS_N}" "${ALLOW_DUPLICATES}"
                                done
                            done
                        done
                    else
                        for EPSILON in ${EPSILONS}; do
                            for LAMBDA_PER_SEGMENT in "${LAMBDA_PER_SEGMENTS_ARRAY[@]}"; do
                                for INTERCEPT_CANDIDATE_NUM in "${INTERCEPT_CANDIDATE_NUMS_ARRAY[@]}"; do
                                    for MU in "${MUS_ARRAY[@]}"; do
                                        run_single_experiment \
                                            "${DATASET}" "${DATASET_CONFIG}" "${DATASET_PATH}" \
                                            "${SEED}" "${LAMBDA}" "${EPSILON}" "${PIPELINE}" "${LAMBDA_PER_SEGMENT}" "${DATASET_N:-}" "${INTERCEPT_CANDIDATE_NUM}" "${MU}" "${DATASET_HAS_N}" "${ALLOW_DUPLICATES}"
                                    done
                                done
                            done
                        done
                    fi
                done
            else
                if [ "${USE_EPS_LAMBDA_PAIRS}" = "true" ]; then
                    for PAIR_JSON in "${EPSILON_LAMBDA_PAIRS_ARRAY[@]}"; do
                        EPSILON=$(echo "${PAIR_JSON}" | jq -r '.[0]')
                        LAMBDA_PER_SEGMENT=$(echo "${PAIR_JSON}" | jq -r '.[1]')
                        for INTERCEPT_CANDIDATE_NUM in "${INTERCEPT_CANDIDATE_NUMS_ARRAY[@]}"; do
                            for MU in "${MUS_ARRAY[@]}"; do
                                    run_single_experiment \
                                        "${DATASET}" "${DATASET_CONFIG}" "${DATASET_PATH}" \
                                        "${SEED}" "" "${EPSILON}" "${PIPELINE}" "${LAMBDA_PER_SEGMENT}" "${DATASET_N:-}" "${INTERCEPT_CANDIDATE_NUM}" "${MU}" "${DATASET_HAS_N}" "${ALLOW_DUPLICATES}"
                            done
                        done
                    done
                else
                    for EPSILON in ${EPSILONS}; do
                        for LAMBDA_PER_SEGMENT in "${LAMBDA_PER_SEGMENTS_ARRAY[@]}"; do
                            for INTERCEPT_CANDIDATE_NUM in "${INTERCEPT_CANDIDATE_NUMS_ARRAY[@]}"; do
                                for MU in "${MUS_ARRAY[@]}"; do
                                    run_single_experiment \
                                        "${DATASET}" "${DATASET_CONFIG}" "${DATASET_PATH}" \
                                        "${SEED}" "" "${EPSILON}" "${PIPELINE}" "${LAMBDA_PER_SEGMENT}" "${DATASET_N:-}" "${INTERCEPT_CANDIDATE_NUM}" "${MU}" "${DATASET_HAS_N}" "${ALLOW_DUPLICATES}"
                                done
                            done
                        done
                    done
                fi
            fi
        fi
    done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
