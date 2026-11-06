# PRD.md:  "The Why"; Product requirements, benchmark metadata table, high-level requirements, etc.

Product Requirements Document

"The Why"; Product requirements, benchmark metadata table, high-level requirements, etc. Defines the benchmark goal, validation objective, test name, benchmark number, category, and high-level success criteria.

## Benchmark Matrix Document Metadata (via benchmark_specification.json)

This PRD.md section is populated from benchmark_specification.json, which is the structured source of benchmark-specific product requirements.

## Workload Number
405

## Workload Name
PyTorch Tensor Op Correctness Suite

## Execution Summary (Run and Measure)
Run local PyTorch CUDA matmul and conv2d for each yaml op_name/dtype/shape, compare each GPU result to a CPU FP64 reference using rtol/atol, and emit pass_rate_percent plus error counts, to measure tensor-op numerical correctness. Does not invoke pytest or GitHub test_ops.py

## Main Goal
Analyze tensor op numerical correctness

## Validation Objective
Validates PyTorch CUDA matmul/conv2d against CPU FP64 within rtol/atol across yaml op/dtype/shape cases

## Workload Category
Compute & Math Kernels

## Validation Requirement

The benchmark must include an automated SQLite-integrated validation layer that verifies persisted results from `results/benchmark.db`. Validation must confirm:

1. The benchmark run completed successfully with no tool errors.
2. Required samples and aggregate metrics were persisted for every swept shape.
3. Metrics are finite and physically sensible (positive, within plausible bounds).
4. Measured values satisfy configured thresholds when the workload defines pass/fail gates.
5. The benchmark fails validation when required data is missing, invalid, or outside bounds.

## Non-Functional Requirements

| Requirement | Target |
|---|---|
| Automation | Runs to completion without manual intervention after `bash run_benchmark.sh` |
| Idempotency | Re-running `run_benchmark.sh` appends a new run; never corrupts existing rows |
| Persistence | All metrics survive script exit; `results/benchmark.db` is the durable record |
