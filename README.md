# MobiLoRA Experiment

This repository contains an engineering-oriented reproduction of the MobiLoRA paper:

- paper page: <https://aclanthology.org/2025.acl-long.1140/>
- repository PDF: [2025.acl-long.1140.pdf](./2025.acl-long.1140.pdf)
- paper PDF: <https://aclanthology.org/2025.acl-long.1140.pdf>

The repository now has two layers:

- an inspectable Python prototype under `src/mobilora/` for explainable benchmarking and reporting
- a vendored `third_party/sglang` fork with local MobiLoRA-oriented patches so we can run real `SGLang + LoRA + small model` experiments on this laptop

The target platform is `Windows 11 + WSL2 + Ubuntu 24.04 + NVIDIA T600 4GB`, with every large artifact kept on `D:`.

## What is implemented

- `CtxAttention`-style prefix reuse and anchor selection in the Python prototype
- similarity-aware delta KV encoding inspired by Eq. (3) in the paper
- context-aware KV cache eviction inspired by Eq. (4) in the paper
- paper-local data preparation for ShareGPT-style conversation, XSum-style writing, and app-usage traces
- a vendored SGLang fork with local request metadata, cache annotations, and `/mobilora/generate`
- a FastAPI dashboard backend plus a React/Vite frontend for live prompting and result exploration
- Windows + WSL bootstrap scripts that keep all large artifacts on `D:`

## Repository layout

- `main.py`: CLI entrypoint
- `configs/runtime.yaml`: default runtime and benchmark config
- `src/mobilora/`: prototype implementation
- `third_party/sglang/`: vendored SGLang source with local MobiLoRA patches
- `patches/sglang_mobilora_local.patch`: replayable patch for the SGLang submodule
- `frontend/`: React/Vite dashboard scaffold
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

The SGLang serving path uses a separate WSL conda env:

- `nn-lesson-SEU-sglang-cu128`
- `torch 2.8.0+cu128`
- model/data/cache paths under `/mnt/d/MobiLoRA_assets`

### 4. Prepare prototype metadata

```bash
conda activate nn-lesson-SEU
python main.py prepare
```

### 5. Run the prototype benchmark

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

## Paper-local SGLang workflow

The local paper workflow keeps the experiment structure close to the paper while scaling the model and adapter count to this laptop.

### 1. Prepare paper-style local data

```bash
conda activate nn-lesson-SEU
python main.py prepare-paper-data
```

This writes prepared local assets under `D:\MobiLoRA_assets\datasets\paper` and `D:\MobiLoRA_assets\datasets\paper_traces`.

### 2. Bootstrap vendored SGLang inside WSL

```bash
python main.py bootstrap-sglang
```

This installs the vendored `third_party/sglang/python` package in editable mode inside the WSL conda environment and writes a manifest to the bench output directory.
Before the editable install, the bootstrap command checks and applies
`patches/sglang_mobilora_local.patch` so a fresh clone can reproduce the local
MobiLoRA SGLang changes without relying on uncommitted submodule state.

### 3. Launch the local MobiLoRA SGLang service

On the `NVIDIA T600 4GB`, run one SGLang service at a time. The patched MobiLoRA
service is the default live path:

```bash
python main.py serve-sglang --variant sglang_mobilora --adapter-count 2 --detach
```

This starts `Qwen/Qwen2.5-0.5B-Instruct` with two SGLang-compatible LoRA
adapters and exposes:

- `/health`
- `/generate`
- `/mobilora/generate`

If an adapter snapshot includes tokenizer files plus `added_tokens.json`, the
launcher creates a slim serving copy under `D:\MobiLoRA_assets\adapters_sglang`
that keeps only the PEFT LoRA config and weights. The original adapter cache is
not modified.

### 4. Run a smoke paper-local benchmark

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
C:\Users\tomat\.conda\envs\nn-lesson-SEU\python.exe main.py run-paper-local --smoke
```

Smoke mode can validate:

- `hf_peft`
- `sglang_stock_lora`
- `sglang_mobilora`

In the current 4GB setup, `sglang_stock_lora` may be marked offline unless you
start that service separately on port `30100`. The MobiLoRA smoke output
produces:

- `paper_results.csv`
- `paper_summary.json`
- request-level traces under the configured bench output directory

### 5. Optional stock SGLang baseline

```bash
python main.py serve-sglang --variant sglang_stock_lora --adapter-count 2 --detach
```

Use `--print-command` if you want to inspect the resolved WSL launch command
first. Stop the MobiLoRA service before launching stock on this GPU if memory is
tight.

### 6. Launch the dashboard backend

```bash
python main.py serve-dashboard --results-dir /mnt/d/MobiLoRA_assets/bench_outputs
```

The backend exposes:

- `/api/server-status`
- `/api/results/summary`
- `/api/results/traces`
- `/api/live/generate`

When the local SGLang service is running, `/api/live/generate` can send a real prompt to your deployed model and return live text plus runtime metadata.

If `frontend/dist` exists, the dashboard backend serves the React/Vite build.
Otherwise it falls back to the built-in FastAPI HTML page.

## Local SGLang notes

- The vendored SGLang fork adds request fields for `app_id`, `app_state`, and `session_id`.
- The fork also exposes `/mobilora/generate` for MobiLoRA-aware prompting while keeping existing endpoints intact.
- On this laptop, the paper scenarios are intentionally scaled down to fit a `Qwen/Qwen2.5-0.5B-Instruct` base model and a small LoRA pool.
- CUDA 12.8 is supported through the `cu128` WSL environment; do not install CUDA 13 wheels into the serving env.
- If a path in `configs/runtime.yaml` uses Windows style like `D:/...`, the loader now converts it correctly when the code runs inside WSL so large outputs still land on the real `D:` drive.

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
- The React/Vite frontend scaffold is committed, but the FastAPI backend already provides a minimal usable local dashboard page even before frontend dependencies are installed.
