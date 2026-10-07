# SPEC.md: "The Human How"; Exact technical requirements, environment setup, implementation details, etc.

## Execution Description

Dispatches scripts/collect_tensor_ops.py to compare GPU matmul/conv2d against a CPU FP64 reference. device_id: torch CUDA device. op_name: yaml matmul,conv2d. dtype: yaml tokens f16_r, bf16_r, f32_r (smoke f16_r; baseline/extended f16_r,bf16_r,f32_r). Sweep dimensions: device_id, op_name, dtype, shape, rtol, atol, tolerance_rel, num_iterations.

## Parameters

| Parameter | CLI Flag | Tested Values | Default | Description |
| --- | --- | --- | --- | --- |
| device_id | `--device-id` | smoke=0, baseline=0, extended=0 | 0 | From Parameter list; see Execution Description With Parameters. |
| op_name | `--op-name` | smoke=matmul,conv2d, baseline=matmul,conv2d, extended=matmul,conv2d | matmul,conv2d | From Parameter list; see Execution Description With Parameters. |
| dtype | `--dtype` | smoke=f16_r, baseline=f16_r,bf16_r,f32_r, extended=f16_r,bf16_r,f32_r | f16_r,bf16_r,f32_r | From Parameter list; see Execution Description With Parameters. |
| shape | `--shape` | smoke=matmul=1x128x128;conv2d=1x3x32x32, baseline=matmul=1x128x128,8x512x512;conv2d=1x3x32x32,8x3x64x64, extended=matmul=1x128x128,8x512x512;conv2d=1x3x32x32,8x3x64x64 | matmul=1x128x128,8x512x512;conv2d=1x3x32x32,8x3x64x64 | From Parameter list; see Execution Description With Parameters. |
| rtol | `--rtol` | smoke=0.02, baseline=f16_r=0.02,bf16_r=0.02,f32_r=1e-05, extended=f16_r=0.02,bf16_r=0.02,f32_r=1e-05 | f16_r=0.02,bf16_r=0.02,f32_r=1e-05 | From Parameter list; see Execution Description With Parameters. |
| atol | `--atol` | smoke=0.1, baseline=f16_r=0.1,bf16_r=0.5,f32_r=1e-03, extended=f16_r=0.1,bf16_r=0.5,f32_r=1e-03 | f16_r=0.1,bf16_r=0.5,f32_r=1e-03 | From Parameter list; see Execution Description With Parameters. |
| tolerance_rel | `--tolerance-rel` | smoke=0.001, baseline=0.001, extended=0.001 | 0.001 | From Parameter list; see Execution Description With Parameters. |
| num_iterations | `--num-iterations` | smoke=2, baseline=50, extended=100 | 50 | From Parameter list; see Execution Description With Parameters. |
| seed | `--seed` | smoke=42, baseline=42, extended=42 | 42 | From Parameter list; see Execution Description With Parameters. |

## Invocation

```bash
Run local PyTorch CUDA matmul/conv2d correctness checks versus a CPU FP64 reference. Does not run pytest or the official PyTorch Operator Correctness test_ops.py suite
```

## Raw Output Format

CSV from collect_tensor_ops.py with one row per op/dtype/shape case

check,command,rc,status,pass_rate_percent,max_abs_error,max_rel_error,tolerance_compliance,coverage_count,failure_count
matmul,matmul,0,ok,100,1e-3,1e-4,100,1,0

## Metrics

- **#1: Max absolute error across correctness cases** — stored as `max_abs_error`.
- **#2: Maximum relative L2 error** — stored as `max_rel_error`.
- **#3: Tolerance-compliant case percentage** — stored as `tolerance_compliance`.
- **#4: Test case coverage count** — stored as `coverage_count`.
- **#5: Test case failure count** — stored as `failure_count`.

## Framework

Dispatches scripts/collect_tensor_ops.py to compare GPU matmul/conv2d against a CPU FP64 reference. device_id: torch CUDA device. op_name: yaml matmul,conv2d.

## Installation and Execution Summary

Run local PyTorch CUDA matmul and conv2d for each yaml op_name/dtype/shape, compare each GPU result to a CPU FP64 reference using rtol/atol, and emit pass_rate_percent plus error counts, to measure tensor-op numerical correctness. Does not invoke pytest or GitHub test_ops.py

## Platform Portability

- **AMD (primary):** ```bash
Run local PyTorch CUDA matmul/conv2d correctness checks versus a CPU FP64 reference. Does not run pytest or the official PyTorch Operator Correctness test_ops.py suite
```
- **NVIDIA:** Native NVIDIA CUDA workload. Execute on the stated Ubuntu release with the host NVIDIA driver and CUDA userspace. ROCm porting notes do not apply.

## Model Context Protocols

- **Active:** None

## Execution-Loop Validation Contract

EXECUTION CHAIN: `run_benchmark.sh` ➔ raw output ➔ `scripts/parse_results.py` ➔ `results/benchmark.db` ➔ `scripts/validate_results.py`

This benchmark uses a lightweight, SQLite-integrated execution loop for result validation. All validation is performed by `scripts/validate_results.py`.

### Validation script usage

```bash
export BENCHMARK_PYTHON=/usr/bin/python3.13  # optional; select the installed interpreter

# After a live run:
".venv/bin/python" scripts/validate_results.py --db results/benchmark.db

# CI / no-GPU path (seeds fixture and validates it):
".venv/bin/python" scripts/validate_results.py --seed-fixture --quiet

# Override DB path via environment variable:
BENCHMARK_DB=tests/fixtures/benchmark.db \
  ".venv/bin/python" scripts/validate_results.py
```

### Run artifact contract

CSV from collect_tensor_ops.py with one row per op/dtype/shape case

check,command,rc,status,pass_rate_percent,max_abs_error,max_rel_error,tolerance_compliance,coverage_count,failure_count
matmul,matmul,0,ok,100,1e-3,1e-4,100,1,0

```bash
bash run_benchmark.sh --help
bash run_benchmark.sh --profile smoke --validate
bash run_benchmark.sh --profile baseline --validate
bash run_benchmark.sh --profile extended --validate
```
`run_benchmark.sh --help` prints usage and exits. The harness calls `scripts/ensure_setup.sh` when `.setup_state` is absent.

### Required integrity checks (built into `validate_results.py`)

1. Latest run exists and `runs.status = 'ok'`.
2. `run.error_message` is NULL.
3. `started_at` and `finished_at` are valid ISO-8601 UTC strings.
4. All required aggregate metrics in `runs` are non-NULL and finite.
5. All required aggregate metrics are physically sensible (positive values). Dispatches scripts/collect_tensor_ops.py to compare GPU matmul/conv2d against a CPU FP64 reference. device_id: torch CUDA device. op_name: yaml matmul,conv2d.
6. At least 2 sample rows exist for the latest `run_id` (sweep coverage).
7. No sample has `status = 'error'`.
8. Dispatches scripts/collect_tensor_ops.py to compare GPU matmul/conv2d against a CPU FP64 reference. device_id: torch CUDA device. op_name: yaml matmul,conv2d.

### Baseline / Threshold configuration (`config/benchmark_config.yaml`)

Expected ranges and gates live in `config/benchmark_config.yaml` under `baselines:` or `thresholds:`. To update them, edit that file — never edit validation code directly.

Threshold key suffixes encode comparison direction when `thresholds:` is present: `_min` → observed value must be ≥ threshold. `_max` → observed value must be ≤ threshold. Informational `baselines:` ranges are not pass/fail gates.
