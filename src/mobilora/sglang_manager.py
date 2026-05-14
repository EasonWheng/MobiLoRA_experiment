from __future__ import annotations

import json
import os
import shutil
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from mobilora.types import AppConfig
from mobilora.utils import WINDOWS_ABS_PATH_RE, ensure_directory


@dataclass(slots=True)
class WSLCommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def windows_to_wsl_path(path: str | Path) -> str:
    raw = str(Path(path).resolve())
    if raw.startswith("/"):
        return raw
    drive, remainder = raw[0].lower(), raw[2:].replace("\\", "/")
    return f"/mnt/{drive}{remainder}"


def _conda_activate_prefix(config: AppConfig) -> str:
    conda_root = config.sglang.conda_root.rstrip("/")
    env_name = config.sglang.conda_env_name
    return (
        f"source {shlex.quote(f'{conda_root}/etc/profile.d/conda.sh')}"
        f" && conda activate {shlex.quote(env_name)}"
    )


def _wsl_asset_exports(config: AppConfig) -> str:
    hf_home = windows_to_wsl_path(config.paths.hf_cache)
    datasets_home = windows_to_wsl_path(config.paths.datasets_cache)
    asset_root = windows_to_wsl_path(config.paths.asset_root)
    env_root = config.sglang.python_executable
    if env_root.endswith("/bin/python"):
        env_root = env_root.removesuffix("/bin/python")
    else:
        env_root = f"{config.sglang.conda_root.rstrip('/')}/envs/{config.sglang.conda_env_name}"
    cuda_home = f"{env_root}/lib/python3.11/site-packages/nvidia/cuda_runtime"
    return (
        f"export MOBILORA_ASSET_ROOT={shlex.quote(asset_root)}"
        f" && export HF_HOME={shlex.quote(hf_home)}"
        f" && export TRANSFORMERS_CACHE={shlex.quote(hf_home)}"
        f" && export HF_DATASETS_CACHE={shlex.quote(datasets_home)}"
        f" && export CUDA_HOME={shlex.quote(cuda_home)}"
        f" && export LD_LIBRARY_PATH=/usr/lib/wsl/lib:{shlex.quote(cuda_home)}/lib:${{LD_LIBRARY_PATH:-}}"
        " && export HF_HUB_DISABLE_XET=1"
        " && export HF_HUB_ENABLE_HF_TRANSFER=0"
        " && export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0"
        " && export SGLANG_FORCE_NATIVE_RMSNORM=1"
        " && export SGLANG_FORCE_NATIVE_ACTIVATION=1"
        " && export SGLANG_FORCE_FP16_RMSNORM=1"
        " && export SGLANG_DISABLE_FP16_RMSNORM=1"
        " && export SGLANG_FORCE_NATIVE_ROPE=1"
        " && export SGLANG_FORCE_NATIVE_KVCACHE=1"
        " && export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1"
        " && export CUDA_LAUNCH_BLOCKING=1"
    )


def _pip_exports(config: AppConfig) -> str:
    pip_cache = windows_to_wsl_path(config.paths.asset_root / "pip_cache")
    return (
        f"export PIP_CACHE_DIR={shlex.quote(pip_cache)}"
        " && export PIP_DEFAULT_TIMEOUT=1000"
        " && export PIP_RETRIES=8"
        " && export PIP_PROGRESS_BAR=off"
    )


def _retry_shell(command: str, attempts: int = 3, delay_seconds: int = 8) -> str:
    quoted = shlex.quote(command)
    parts = [f"bash -lc {quoted}"]
    for _ in range(max(attempts - 1, 0)):
        parts.append(f"(sleep {delay_seconds} && bash -lc {quoted})")
    return " || ".join(parts)


def run_wsl_command(
    config: AppConfig,
    command: str,
    timeout_ms: int = 300_000,
) -> WSLCommandResult:
    if os.name == "nt":
        completed = subprocess.run(
            [
                "wsl.exe",
                "-d",
                config.sglang.wsl_distro,
                "--",
                "bash",
                "-lc",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=max(timeout_ms // 1000, 1),
            check=False,
        )
    else:
        completed = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=max(timeout_ms // 1000, 1),
            check=False,
        )
    return WSLCommandResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def sglang_server_url(config: AppConfig, variant: str) -> str:
    port = config.sglang.stock_port
    if variant == "sglang_mobilora":
        port = config.sglang.mobilora_port
    return f"http://{config.sglang.host}:{port}"


def _ensure_sglang_lora_dir(config: AppConfig, alias: str, adapter_dir: Path) -> Path:
    """Return a local adapter dir that SGLang can load without added-vocab guards."""
    if not (adapter_dir / "added_tokens.json").exists():
        return adapter_dir

    slim_dir = config.paths.asset_root / "adapters_sglang" / alias
    ensure_directory(slim_dir)
    for filename in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "README.md",
    ):
        source = adapter_dir / filename
        if source.exists():
            shutil.copy2(source, slim_dir / filename)
    marker = slim_dir / "MOBILORA_SGLANG_SLIM_ADAPTER.txt"
    marker.write_text(
        "This directory is generated from the full adapter cache for SGLang "
        "serving. Tokenizer and added_tokens files are intentionally omitted "
        "because these benchmark adapters reuse the base model vocabulary for "
        "generation, while SGLang rejects adapters with added_tokens.json.\n"
        f"Source: {adapter_dir}\n",
        encoding="utf-8",
    )
    return slim_dir


def default_launch_command(
    config: AppConfig,
    variant: str,
    lora_aliases: list[str],
) -> str:
    source_root = windows_to_wsl_path(config.sglang.source_root)
    log_dir = windows_to_wsl_path(config.paths.bench_outputs / "sglang_logs")
    port = config.sglang.stock_port if variant == "sglang_stock_lora" else config.sglang.mobilora_port
    model_path = config.sglang.launch_model_name
    if WINDOWS_ABS_PATH_RE.match(model_path):
        model_path = windows_to_wsl_path(model_path)

    lora_entries: list[str] = []
    for alias, repo in zip(lora_aliases, config.adapter_specs):
        local_adapter = config.paths.adapters_cache / alias
        if local_adapter.exists():
            serving_adapter = _ensure_sglang_lora_dir(config, alias, local_adapter)
            lora_path = windows_to_wsl_path(serving_adapter)
        else:
            lora_path = repo.repo_id
        lora_entries.append(shlex.quote(f"{alias}={lora_path}"))
    lora_args = " ".join(lora_entries)
    lora_flags = ""
    if lora_entries:
        lora_flags = (
            " --enable-lora"
            " --enable-lora-overlap-loading"
            f" --max-loras-per-batch {config.sglang.max_loras_per_batch}"
            f" --max-loaded-loras {config.sglang.max_loaded_loras}"
            " --lora-backend torch_native"
            f" --lora-paths {lora_args}"
        )
    mobilora_flags = ""
    if variant == "sglang_mobilora":
        mobilora_flags = (
            " --enable-mobilora"
            " --mobilora-cross-adapter-reuse"
            " --mobilora-state-lambda 0.5"
            " --mobilora-recency-lambda 0.3"
            " --mobilora-length-lambda 0.2"
            " --radix-eviction-policy priority"
        )

    return (
        f"{_conda_activate_prefix(config)}"
        f" && {_wsl_asset_exports(config)}"
        f" && mkdir -p {shlex.quote(log_dir)}"
        f" && cd {shlex.quote(source_root)}"
        f" && {shlex.quote(config.sglang.python_executable)} -m sglang.launch_server"
        f" --model-path {shlex.quote(model_path)}"
        " --dtype float32"
        " --host 0.0.0.0"
        f" --port {port}"
        f" --context-length {config.sglang.context_length}"
        f" --max-total-tokens {config.sglang.max_total_tokens}"
        f" --max-running-requests {config.sglang.max_running_requests}"
        f" --mem-fraction-static {config.sglang.mem_fraction_static}"
        " --attention-backend torch_native"
        " --sampling-backend pytorch"
        " --skip-server-warmup"
        f"{lora_flags}"
        f"{' --disable-cuda-graph' if config.sglang.disable_cuda_graph else ''}"
        f"{mobilora_flags}"
    )


def launch_sglang_server(
    config: AppConfig,
    variant: str,
    lora_aliases: list[str],
    *,
    detach: bool = False,
    log_file: Path | None = None,
    timeout_ms: int = 30_000,
) -> dict[str, object]:
    command = default_launch_command(config, variant, lora_aliases)
    if detach:
        log_path = log_file or (
            config.paths.bench_outputs
            / "sglang_logs"
            / f"{variant}.log"
        )
        ensure_directory(log_path.parent)
        log_wsl = windows_to_wsl_path(log_path)
        payload = json.dumps(
            {
                "command": command,
                "log_file": log_wsl,
            }
        )
        # Popen avoids shell-specific PID handling quirks across PowerShell -> WSL.
        detached_command = (
            f"{shlex.quote(config.sglang.python_executable)} - <<'PY'\n"
            "import json\n"
            "import subprocess\n"
            f"payload = {payload!r}\n"
            "data = json.loads(payload)\n"
            "log = open(data['log_file'], 'wb', buffering=0)\n"
            "process = subprocess.Popen(\n"
            "    ['bash', '-lc', data['command']],\n"
            "    stdin=subprocess.DEVNULL,\n"
            "    stdout=log,\n"
            "    stderr=subprocess.STDOUT,\n"
            "    start_new_session=True,\n"
            ")\n"
            "print(process.pid)\n"
            "PY"
        )
        result = run_wsl_command(config, detached_command, timeout_ms=timeout_ms)
        pid = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        return {
            "variant": variant,
            "detach": True,
            "pid": pid,
            "log_file": str(log_path),
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    if os.name == "nt":
        completed = subprocess.run(
            [
                "wsl.exe",
                "-d",
                config.sglang.wsl_distro,
                "--",
                "bash",
                "-lc",
                command,
            ],
            text=True,
            capture_output=False,
            timeout=max(timeout_ms // 1000, 1),
            check=False,
        )
        return {
            "variant": variant,
            "detach": False,
            "command": command,
            "returncode": completed.returncode,
        }

    completed = subprocess.run(
        ["bash", "-lc", command],
        text=True,
        capture_output=False,
        timeout=max(timeout_ms // 1000, 1),
        check=False,
    )
    return {
        "variant": variant,
        "detach": False,
        "command": command,
        "returncode": completed.returncode,
    }


def bootstrap_sglang(
    config: AppConfig,
    output_dir: Path,
    install_stock_first: bool = True,
    timeout_ms: int = 3_600_000,
) -> Path:
    ensure_directory(output_dir)
    ensure_directory(config.paths.asset_root / "pip_cache")
    source_root = windows_to_wsl_path(config.sglang.source_root)
    sglang_repo_root = windows_to_wsl_path(config.sglang.source_root.parent)
    repo_root = config.sglang.source_root.parents[2]
    local_patch = repo_root / "patches" / "sglang_mobilora_local.patch"
    local_patch_wsl = windows_to_wsl_path(local_patch)
    conda_root = config.sglang.conda_root.rstrip("/")
    env_name = config.sglang.conda_env_name

    torch_install = (
        "python -m pip install"
        f" torch=={shlex.quote(config.sglang.torch_version)}"
        f" torchvision=={shlex.quote(config.sglang.torchvision_version)}"
        f" torchaudio=={shlex.quote(config.sglang.torchaudio_version)}"
        f" --index-url {shlex.quote(config.sglang.torch_index_url)}"
    )
    stock_install = (
        "python -m pip install"
        f" sglang=={shlex.quote(config.sglang.stable_package_version)}"
        f" --extra-index-url {shlex.quote(config.sglang.torch_index_url)}"
        f" --extra-index-url {shlex.quote(config.sglang.extra_index_url)}"
    )
    runtime_packages = (
        "python -m pip install"
        " fastapi uvicorn pydantic huggingface_hub requests pyyaml packaging"
        " transformers peft accelerate safetensors datasets sentencepiece"
        " pandas matplotlib"
    )
    env_bootstrap = (
        f"source {shlex.quote(f'{conda_root}/etc/profile.d/conda.sh')}"
        f" && if ! conda env list | awk '{{print $1}}' | grep -Fxq {shlex.quote(env_name)}; then"
        f" conda create -y -n {shlex.quote(env_name)} python=3.11;"
        " fi"
    )
    commands = [
        env_bootstrap,
        (
            f"{_conda_activate_prefix(config)}"
            f" && {_pip_exports(config)}"
            " && python -m pip install --upgrade pip setuptools wheel"
        ),
        (
            f"{_conda_activate_prefix(config)}"
            f" && {_pip_exports(config)}"
            f" && {_retry_shell(torch_install, attempts=4, delay_seconds=10)}"
        ),
        (
            f"{_conda_activate_prefix(config)}"
            f" && {_pip_exports(config)}"
            f" && {_retry_shell(runtime_packages, attempts=3, delay_seconds=8)}"
        ),
    ]
    if install_stock_first:
        commands.append(
            f"{_conda_activate_prefix(config)}"
            f" && {_pip_exports(config)}"
            f" && {_retry_shell(stock_install, attempts=3, delay_seconds=8)}"
        )
    if local_patch.exists():
        commands.append(
            f"cd {shlex.quote(sglang_repo_root)}"
            " && if grep -q -- '--enable-mobilora' python/sglang/srt/server_args.py"
            " && test -f python/sglang/srt/mobilora.py; then"
            " echo 'MobiLoRA SGLang patch already applied';"
            f" elif git apply --check {shlex.quote(local_patch_wsl)}; then"
            f" git apply {shlex.quote(local_patch_wsl)};"
            " else"
            " echo 'MobiLoRA SGLang patch cannot be applied cleanly' >&2;"
            " exit 1;"
            " fi"
        )
    commands.extend(
        [
            (
                f"{_conda_activate_prefix(config)}"
                f" && {_pip_exports(config)}"
                f" && cd {shlex.quote(source_root)}"
                " && python -m pip install -e . --no-deps --no-build-isolation"
            ),
            (
                f"{_conda_activate_prefix(config)}"
                " && python -m pip check"
            ),
            (
                f"{_conda_activate_prefix(config)}"
                " && python - <<'PY'\n"
                "import json\n"
                "import torch\n"
                "import sglang\n"
                "print(json.dumps({\n"
                "  'torch_version': torch.__version__,\n"
                "  'torch_cuda_version': torch.version.cuda,\n"
                "  'cuda_available': bool(torch.cuda.is_available()),\n"
                "  'device_count': int(torch.cuda.device_count()),\n"
                "  'sglang_version': getattr(sglang, '__version__', 'unknown'),\n"
                "}, ensure_ascii=False))\n"
                "PY"
            ),
        ]
    )

    results = []
    for command in commands:
        result = run_wsl_command(config, command, timeout_ms=timeout_ms)
        results.append(
            {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout[-12_000:],
                "stderr": result.stderr[-12_000:],
            }
        )
        if result.returncode != 0:
            break

    manifest = {
        "wsl_distro": config.sglang.wsl_distro,
        "conda_root": config.sglang.conda_root,
        "conda_env_name": config.sglang.conda_env_name,
        "python_executable": config.sglang.python_executable,
        "source_root": str(config.sglang.source_root),
        "source_commit": config.sglang.source_commit,
        "stable_release_tag": config.sglang.stable_release_tag,
        "stable_package_version": config.sglang.stable_package_version,
        "torch_version": config.sglang.torch_version,
        "torchvision_version": config.sglang.torchvision_version,
        "torchaudio_version": config.sglang.torchaudio_version,
        "local_patch": str(local_patch) if local_patch.exists() else "",
        "results": results,
    }
    manifest_path = output_dir / "bootstrap_sglang_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def probe_sglang_server(config: AppConfig, variant: str) -> dict[str, object]:
    import urllib.request

    url = sglang_server_url(config, variant)
    health_url = f"{url}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=5) as response:
            ok = 200 <= int(response.status) < 300
            detail = response.read().decode("utf-8", errors="ignore")
    except Exception as exc:  # noqa: BLE001
        return {
            "variant": variant,
            "url": url,
            "healthy": False,
            "detail": str(exc),
        }
    return {
        "variant": variant,
        "url": url,
        "healthy": ok,
        "detail": detail[:200],
    }
