# MobiLoRA Experiment

This repository contains an engineering-oriented reproduction of the MobiLoRA paper:

- paper page: <https://aclanthology.org/2025.acl-long.1140/>
- repository PDF: [2025.acl-long.1140.pdf](./2025.acl-long.1140.pdf)
- paper PDF: <https://aclanthology.org/2025.acl-long.1140.pdf>

The goal is not to patch the original SGLang runtime directly. Instead, this repo recreates the paper's core mechanisms in a lightweight, inspectable prototype that can run on a constrained Windows laptop through `WSL2 + Ubuntu 24.04 LTS`, while keeping every large artifact on `D:`.

## What is implemented

- `CtxAttention`-style prefix tree for cross-adapter prefix reuse
- similarity-aware delta KV encoding inspired by Eq. (3) in the paper
- context-aware KV cache eviction inspired by Eq. (4) in the paper
- benchmark orchestration for five scenario variants and two workload families
- reporting pipeline that aggregates results into CSV and Markdown
- Windows + WSL bootstrap scripts that keep all large artifacts on `D:`

## Repository layout

- `main.py`: CLI entrypoint
- `configs/runtime.yaml`: default runtime and benchmark config
- `src/mobilora/`: prototype implementation
- `scripts/install_wsl_ubuntu24.ps1`: installs WSL prerequisites and imports Ubuntu 24.04 onto `D:`
- `scripts/bootstrap_wsl_env.sh`: installs Miniconda and Python dependencies inside WSL
- `data/samples/`: tiny workload samples kept in git for smoke runs

## Quick start

### 1. Initialize local assets on `D:`

The repository expects these directories to exist:

- `D:\MobiLoRA_assets\hf_cache`
- `D:\MobiLoRA_assets\datasets`
- `D:\MobiLoRA_assets\adapters`
- `D:\MobiLoRA_assets\bench_outputs`
- `D:\MobiLoRA_assets\docker`
- `D:\WSL\Ubuntu-24.04`

### 2. Install WSL2 + Ubuntu 24.04 LTS onto `D:`

Run PowerShell as administrator:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\install_wsl_ubuntu24.ps1
```

This script:

- enables the Windows features needed by WSL
- downloads the official Ubuntu 24.04 WSL image to `D:\WSL\downloads`
- imports the distro as `Ubuntu-24.04` into `D:\WSL\Ubuntu-24.04`

### 3. Bootstrap the Linux environment

From inside the imported Ubuntu distro:

```bash
bash /mnt/d/learn_pytorch/SEU_DEEPLEARNING_LESSON_EXPERIMENTS/MobiLoRA_experiment/scripts/bootstrap_wsl_env.sh
```

This installs:

- Miniconda under `/home/$USER/miniconda3`
- conda env `nn-lesson-SEU`
- CUDA-compatible PyTorch stack and prototype dependencies

### 4. Prepare metadata

```bash
conda activate nn-lesson-SEU
python main.py prepare
```

### 5. Run a benchmark

The default backend is `mock`, which exercises the full caching, delta, and reporting pipeline without requiring the base model to fit in your current environment.

```bash
python main.py bench --backend mock
python main.py report
```

If your WSL environment has the required Hugging Face stack and model access, you can also try:

```bash
python main.py prepare --backend hf
```

The HF backend is intentionally conservative and currently focuses on validation and artifact extraction rather than full-speed serving.

## Explainability outputs

Each benchmark run now writes three layers of inspectable output under the configured bench output directory:

- `results.csv`: aggregated scenario/workload/variant rows with driver columns such as `avg_prefill_saved_ms`, `avg_delta_penalty_ms`, and `avg_runtime_prefill_ms`
- `summary.json`: machine-readable metric definitions plus per-row driver summaries
- `traces/*.jsonl`: one record per request with anchor ids, reuse lengths, delta error bounds, runtime timings, and CUDA memory snapshots

For the HF backend, the benchmark first warms the runtime and then extracts each request artifact exactly once before scoring all variants. This keeps large model costs on `D:` while making per-variant comparisons easier to explain because the variants share the same measured base artifact.

## Notes

- Large files are deliberately kept out of git.
- Default paths resolve to `D:\MobiLoRA_assets` on Windows and `/mnt/d/MobiLoRA_assets` inside WSL.
- The benchmark pipeline writes outputs to `bench_outputs` under the asset root by default.
