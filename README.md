# PGM-attack

## Overview

Official code for our paper, *Poisoning Attacks on PGM-index*.

## Quick Start

### 1. Clone and submodules

```bash
git clone [REPOSITORY LINK]
cd pgm-index-poisoning
git submodule update --init --recursive
```

### 2. Build

```bash
./scripts/build.sh
```

Requires CMake (see Dependencies). Outputs go under `build/` (binaries in `build/bin` by default).

### 3. Download, preprocess, and synthetic data

```bash
./scripts/download_alex_dataset.sh       # ALEX
./scripts/download_mgbench.sh            # mgbench
./scripts/download_sosd_datasets.sh      # SOSD
./scripts/preprocess_alex_dataset.sh     # ALEX
./scripts/preprocess_mgbench.sh          # mgbench
./scripts/generate_synthetic_dataset.sh  # synthetic
```

### 4. Experiments

Each run takes one JSON config via `./scripts/run_experiment.sh <config.json>`. Examples:

**4.1 Maximize max error (e.g. n=16, consecutive method)**

```bash
./scripts/run_experiment.sh "configs/experiments/poison_attack/maximize_maxerror_consec/baseline_n16.json"
```

**4.2 Minimize covered legitimate keys (e.g. epsilon=2, consecutive method)**

```bash
./scripts/run_experiment.sh "configs/experiments/poison_attack/minimize_segment_length/baseline_epsilon2.json"
```

**4.3 Maximize m_opt (PGM-index example)**

```bash
./scripts/run_experiment.sh "configs/experiments/poison_attack/inject_poisons_to_minimize_segment_length_swing_lambda_with_theta/baseline.json"
```

**4.4 Upper bound on m_opt (instance-dependent example)**

```bash
./scripts/run_experiment.sh "configs/experiments/upper_bound/fix_w_per_block/baseline.json"
```

**4.5 Run all experiments (batch)**

Runs the full list of experiment configs in `scripts/run_all_experiment.sh` (edit that file to change the set).

```bash
./scripts/run_all_experiment.sh
```

### 5. Plots and tables

From the repo root:

```bash
./scripts/plot.sh
./scripts/print_table.sh
```

Figures and logs are written under `fig/`. Install Python dependencies first: `pip install -r requirements.txt`.

## Dependencies

- **C++ toolchain:** `g++`, `make`, and **CMake** (the build script configures with CMake and builds with `cmake --build`, which typically invokes Make).
- **Python 3** with packages in `requirements.txt` (plotting / table scripts).
- **jq** — required by `./scripts/run_experiment.sh` to parse experiment JSON.

## Project Structure

| Path | Role |
|------|------|
| `src/` | C++ sources (attacks, PGM benchmarks, tools, upper bounds). |
| `configs/experiments/` | Experiment JSON configs (paths under `poison_attack/`, `upper_bound/`, etc.). |
| `scripts/` | Build, data, `run_experiment.sh`, `run_all_experiment.sh`, `plot.sh`, `print_table.sh`. |
| `third_party/` | Git submodules (PGM-index, FITing-Tree, RadixSpline). |
| `data/`, `results/`, `fig/` | Data, run outputs, and generated figures / table logs. |
| `plot/` | Python plotting and table-printing scripts. |

## Docker

Build the image once:

```bash
./docker_build.sh
```

- **`docker_run.sh`** — Interactive container: mounts the repo at `/workspace`, allocates 8 CPUs (`0–7`), drops you into `bash` so you can build and run commands manually.
- **`docker_run_all.sh`** — Detached run: executes `./scripts/run_all_experiment.sh` inside the container (8 CPUs) and writes stdout to `results/run_all.log`.
- **`docker_run_all_single_cpu.sh`** — Same as `docker_run_all.sh` but with one CPU and `OMP_NUM_THREADS=1` for single-threaded runs.

All three use image `pgm_poisoning:latest` and the same capability / volume settings as in the scripts.
