# Results: 405 PyTorch Tensor Op Correctness Suite

**Main goal:** Analyze tensor op numerical correctness  
**Tools:** Run scripts/collect_tensor_ops.py in the repo .venv: GPU matmul/conv2d vs CPU FP64 per op, dtype, and shape

Reference results measured by the maintainer on the system shown in each section (generator TEMPLATE_00_141 · recorded 2026-11-14). Your own runs write `results/RESULTS_ACTUAL.md` in the same format.

> Smoke runs use deliberately small sizes. They check that the workload installs and runs end to end; their numbers are not performance results. Baseline and extended are.

<!-- profile:smoke -->
## smoke

- **System:** 1× NVIDIA H100 PCIe (81559 MiB) · Intel(R) Xeon(R) Platinum 8352Y CPU @ 2.20GHz · Ubuntu 26.04.1 LTS · kernel 6.8.0-138-generic · driver 595.91.07
- **Run:** 2026-11-14T10:55:11Z · status **PASS** (exit 0)
- **Time (mm:ss):** setup 02:41 · benchmark 00:06 · whole run 03:06
- **Source:** run 405_20261114_105511
- **Command:** `bash run_benchmark.sh --smoke --device-id 0 --op-name matmul,conv2d --dtype f16_r --shape matmul=1x128x128;conv2d=1x3x32x32 --rtol 0.02 --atol 0.1 --num-iterations 2 --seed 42`

| # | Metric | Result |
|---|---|---|
| 1 | Max absolute error across correctness cases (max_abs_error) | 0.015364906750619411 |
| 2 | Maximum relative L2 error (max_rel_error) | 0.00020789817172142986 |
| 3 | Tolerance-compliant case percentage (tolerance_compliance) | 100.0 |
| 4 | Test case coverage count (coverage_count) | 2.0 |
| 5 | Test case failure count (failure_count) | 0.0 |
<!-- /profile:smoke -->
