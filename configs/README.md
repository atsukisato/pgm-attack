# Configuration Files

This directory contains experiment configuration files. All configuration files are in JSON format.

## Directory Structure

- `datasets/` - Dataset metadata (path, type, description)
- `experiments/` - Experiment configurations organized by experiment type
  - `poison_attack/` - Poison attack experiments
    - `random/` - Random attack variants
  - `upper_bound/` - Upper bound computation experiments
    - `simple/` - Simple upper bound method variants

## Usage

To run a single experiment:

```bash
./scripts/run_experiment.sh configs/experiments/poison_attack/maximize_maxerror_consec/baseline.json
```

To run all experiments:

```bash
./scripts/run_all_experiment.sh
```

## Configuration Resolution Process

1. Load experiment configuration from `experiments/`
2. Load referenced dataset configuration from `datasets/`
3. Generate Cartesian product of array parameters (`datasets`, `lambdas`, `epsilons`, `seeds`)
4. Execute experiments for each parameter combination

## Configuration File Format

### Poison Attack Experiment Example

```json
{
  "name": "random_baseline",
  "description": "Random attack baseline experiment",
  "datasets": ["fb_200M"],
  "lambdas": [1000],
  "epsilons": [64],
  "seeds": [0],
  "attack": {
    "type": "random",
    "method": "random"
  },
  "bound": {
    "method": "simple"
  },
  "pipeline": [
    "generate_poisons",
    "benchmark_pgm"
  ]
}
```

### Upper Bound Computation Example

```json
{
  "name": "upper_bound_baseline",
  "description": "Upper bound computation baseline",
  "datasets": ["fb_200M"],
  "lambdas": [1000],
  "epsilons": [64],
  "bound": {
    "method": "simple"
  },
  "pipeline": [
    "compute_upper_bound"
  ]
}
```

Note: Upper bound experiments do not require `seeds` or `attack` fields.

### Parameter Sweep Example

Array parameters automatically generate Cartesian products:

```json
{
  "name": "random_sweep",
  "datasets": ["fb_200M", "books_800M"],
  "lambdas": [100, 500, 1000, 5000],
  "epsilons": [64, 128, 256],
  "seeds": [0, 1, 2, 3, 4],
  "attack": {
    "type": "random",
    "method": "random"
  },
  "pipeline": ["generate_poisons", "benchmark_pgm"]
}
```

This executes 2 datasets × 4 lambdas × 3 epsilons × 5 seeds = 120 experiments.

## Field Descriptions

### Required Fields

- `name`: Experiment name (used in run ID)
- `datasets`: Array of dataset names (or single string)
- `pipeline`: Array of pipeline steps to execute

### Array Parameters

- `datasets`: Dataset names to use (references files in `datasets/`)
- `lambdas`: Number of poisons to generate
- `epsilons`: PGM-index epsilon values
- `seeds`: Random seeds (required if `generate_poisons` is in pipeline)

### Optional Fields

- `description`: Human-readable description
- `attack`: Attack configuration object
  - `type`: Attack type identifier
  - `method`: Attack method name
- `bound`: Upper bound computation configuration
  - `method`: Bound computation method (e.g., "simple")

### Pipeline Steps

Available pipeline steps:

- `generate_poisons`: Generate poison values
  - Requires: `seeds`, `lambdas`, `attack.method`
  - Output: `poisons.bin`, `poisons_info.json`
- `benchmark_pgm`: Build PGM-index and run query benchmark
  - Requires: `epsilons`, optionally `poisons.bin` from previous step
  - Output: `benchmark_pgm.json`
- `compute_upper_bound`: Compute upper bound for m_opt
  - Requires: `lambdas`, `epsilons`, `bound.method`
  - Output: `upper_bound.json`

### Dataset Configuration Format

Dataset files in `datasets/` follow this format:

```json
{
  "name": "fb_200M_uint64",
  "type": "uint64",
  "path": "data/fb_200M_uint64",
  "description": "Facebook user IDs dataset (200M entries)"
}
```
