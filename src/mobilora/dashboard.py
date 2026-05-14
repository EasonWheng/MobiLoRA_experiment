from __future__ import annotations

import json
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mobilora.config import load_config
from mobilora.sglang_manager import probe_sglang_server, sglang_server_url
from mobilora.types import AppConfig


FALLBACK_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>MobiLoRA Local Dashboard</title>
    <style>
      :root {
        --bg: #f3efe5;
        --panel: rgba(255,255,255,0.86);
        --ink: #15231c;
        --muted: #55655e;
        --line: rgba(21,35,28,0.12);
        --accent: #c85f3d;
        --accent-2: #0f6c63;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Segoe UI", "PingFang SC", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(200,95,61,0.18), transparent 28%),
          radial-gradient(circle at top right, rgba(15,108,99,0.16), transparent 22%),
          linear-gradient(180deg, #f7f3ea 0%, var(--bg) 100%);
      }
      .shell {
        max-width: 1180px;
        margin: 0 auto;
        padding: 32px 20px 48px;
      }
      .hero {
        display: grid;
        grid-template-columns: 1.25fr 1fr;
        gap: 20px;
        margin-bottom: 20px;
      }
      .card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 22px;
        backdrop-filter: blur(18px);
        box-shadow: 0 20px 50px rgba(47, 54, 42, 0.09);
      }
      h1 { margin: 0 0 8px; font-size: 40px; line-height: 1.03; }
      h2 { margin: 0 0 12px; font-size: 22px; }
      p { margin: 0; color: var(--muted); line-height: 1.6; }
      .grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 20px;
      }
      .metric {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        padding: 12px 0;
        border-top: 1px solid var(--line);
      }
      .metric:first-child { border-top: 0; padding-top: 0; }
      .label { color: var(--muted); }
      .value { font-weight: 700; }
      .stack { display: grid; gap: 16px; }
      .panel-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
      }
      textarea, select, input, button {
        width: 100%;
        border-radius: 16px;
        border: 1px solid var(--line);
        padding: 12px 14px;
        font: inherit;
      }
      textarea { min-height: 132px; resize: vertical; background: white; }
      button {
        background: linear-gradient(135deg, var(--accent), #d67f57);
        color: white;
        cursor: pointer;
        font-weight: 700;
      }
      pre {
        white-space: pre-wrap;
        margin: 0;
        font-family: "Cascadia Code", Consolas, monospace;
        font-size: 13px;
      }
      .hint { font-size: 12px; color: var(--muted); margin-top: 10px; }
      @media (max-width: 900px) {
        .hero, .grid { grid-template-columns: 1fr; }
        h1 { font-size: 34px; }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <div class="hero">
        <section class="card">
          <h1>MobiLoRA Local Dashboard</h1>
          <p>
            This fallback UI is served directly by the FastAPI backend so the local
            experiment loop stays usable before the Vite toolchain is installed.
            It can still call your local SGLang deployment and inspect paper outputs.
          </p>
        </section>
        <section class="card">
          <div class="stack" id="server-status"></div>
        </section>
      </div>
      <div class="grid">
        <section class="card">
          <div class="panel-title"><h2>Live Prompt</h2></div>
          <div class="stack">
            <select id="variant">
              <option value="sglang_mobilora">sglang_mobilora</option>
              <option value="sglang_stock_lora">sglang_stock_lora</option>
            </select>
            <input id="adapter" placeholder="Adapter alias" />
            <input id="appId" placeholder="App id" value="demo-app" />
            <select id="appState">
              <option value="foreground">foreground</option>
              <option value="background">background</option>
              <option value="killed">killed</option>
            </select>
            <input id="sessionId" placeholder="Session id" value="session-001" />
            <textarea id="prompt">Explain what cross-adapter prefix reuse means in MobiLoRA.</textarea>
            <button id="submitBtn">Send Request</button>
            <pre id="liveOutput">No request yet.</pre>
          </div>
        </section>
        <section class="card">
          <div class="panel-title"><h2>Experiment Runs</h2></div>
          <pre id="summaryOutput">Loading summary...</pre>
        </section>
        <section class="card">
          <div class="panel-title"><h2>Trace Explorer</h2></div>
          <pre id="traceOutput">Loading traces...</pre>
          <p class="hint">Install frontend dependencies and build the Vite app to replace this fallback UI.</p>
        </section>
      </div>
    </div>
    <script>
      async function fetchJson(url, options) {
        const response = await fetch(url, options);
        if (!response.ok) {
          throw new Error(await response.text());
        }
        return await response.json();
      }
      async function refresh() {
        const status = await fetchJson('/api/server-status');
        document.getElementById('server-status').innerHTML = status.items.map((item) => `
          <div class="metric"><span class="label">${item.variant}</span><span class="value">${item.healthy ? 'healthy' : 'offline'}</span></div>
        `).join('');

        const summary = await fetchJson('/api/results/summary');
        document.getElementById('summaryOutput').textContent = JSON.stringify(summary, null, 2);

        const traces = await fetchJson('/api/results/traces');
        document.getElementById('traceOutput').textContent = JSON.stringify(traces, null, 2);
      }
      document.getElementById('submitBtn').onclick = async () => {
        const body = {
          variant: document.getElementById('variant').value,
          lora_name: document.getElementById('adapter').value || null,
          app_id: document.getElementById('appId').value || 'demo-app',
          app_state: document.getElementById('appState').value,
          session_id: document.getElementById('sessionId').value || 'session-001',
          prompt: document.getElementById('prompt').value,
          max_new_tokens: 96
        };
        document.getElementById('liveOutput').textContent = 'Sending...';
        try {
          const result = await fetchJson('/api/live/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
          });
          document.getElementById('liveOutput').textContent = JSON.stringify(result, null, 2);
        } catch (error) {
          document.getElementById('liveOutput').textContent = String(error);
        }
      };
      refresh().catch((error) => {
        document.getElementById('summaryOutput').textContent = String(error);
      });
    </script>
  </body>
</html>
"""


def _read_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:  # noqa: BLE001
        return []


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return parsed


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        parsed = int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return parsed


def _read_jsonl_tail(path: Path, limit: int) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:  # noqa: BLE001
        return []
    for line in reversed(lines[-limit:]):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _nested_number(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        parsed = _coerce_float(value, default=float("nan"))
        if parsed == parsed:
            return parsed
    return None


def _build_live_record(
    *,
    body: dict[str, object],
    payload: dict[str, object],
) -> dict[str, object]:
    meta = payload.get("meta_info")
    meta_info = meta if isinstance(meta, dict) else {}
    raw_text = str(payload.get("text") or "")
    cached_tokens = _coerce_int(meta_info.get("cached_tokens"))
    reuse_tokens = _coerce_int(
        meta_info.get("mobilora_hit_tokens"),
        default=cached_tokens,
    )
    completion_tokens = _coerce_int(meta_info.get("completion_tokens"))
    prompt_tokens = _coerce_int(meta_info.get("prompt_tokens"))
    compression_ratio = _nested_number(
        meta_info,
        "mobilora_compression_ratio",
        "mobilora_delta_ratio",
        "delta_ratio",
        "compression_ratio",
    )
    finish_reason = meta_info.get("finish_reason")
    if isinstance(finish_reason, dict):
        finish_reason = finish_reason.get("type") or json.dumps(finish_reason, ensure_ascii=False)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": str(meta_info.get("id") or ""),
        "variant": str(body.get("variant") or "sglang_mobilora"),
        "endpoint": str(payload.get("endpoint") or ""),
        "lora_name": str(body.get("lora_name") or ""),
        "app_id": str(body.get("app_id") or "demo-app"),
        "app_state": str(body.get("app_state") or "foreground"),
        "session_id": str(body.get("session_id") or "session-001"),
        "ttft_ms": round(_coerce_float(payload.get("ttft_ms")), 4),
        "e2e_ms": round(_coerce_float(payload.get("e2e_ms")), 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "reuse_tokens": reuse_tokens,
        "cache_hit": reuse_tokens > 0 or cached_tokens > 0,
        "anchor_id": str(meta_info.get("mobilora_anchor_id") or ""),
        "utility": round(_coerce_float(meta_info.get("mobilora_utility")), 6),
        "priority": _coerce_int(meta_info.get("mobilora_priority")),
        "compression_ratio": compression_ratio,
        "max_new_tokens": _coerce_int(body.get("max_new_tokens")),
        "finish_reason": str(finish_reason or ""),
        "text_preview": raw_text[:1200],
    }


def create_dashboard_app(config: AppConfig, results_dir: Path | None = None):
    from fastapi import Body, FastAPI
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    from mobilora.runtime import call_sglang_endpoint

    app = FastAPI(title="MobiLoRA Local Dashboard", version="0.2.0")
    result_root = results_dir or config.paths.bench_outputs
    frontend_dist = config.dashboard.frontend_dir / "dist"
    if (frontend_dist / "assets").exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(frontend_dist / "assets")),
            name="dashboard-assets",
        )

    @app.get("/", response_class=HTMLResponse)
    async def index():
        built_index = frontend_dist / "index.html"
        if built_index.exists():
            return FileResponse(built_index)
        return FALLBACK_HTML

    @app.get("/api/server-status")
    async def server_status() -> dict[str, object]:
        items = [
            probe_sglang_server(config, "sglang_stock_lora"),
            probe_sglang_server(config, "sglang_mobilora"),
        ]
        return {"items": items}

    @app.get("/api/results/summary")
    async def results_summary() -> dict[str, object]:
        candidates = [
            result_root / "paper_local" / "paper_summary.json",
            result_root / "summary.json",
        ]
        for candidate in candidates:
            payload = _read_json_if_exists(candidate)
            if payload:
                return payload
        return {"detail": "No summary file found yet."}

    @app.get("/api/results/rows")
    async def results_rows() -> dict[str, object]:
        candidates = [
            result_root / "paper_local" / "paper_results.csv",
            result_root / "results.csv",
        ]
        for candidate in candidates:
            rows = _read_csv_if_exists(candidate)
            if rows:
                return {"source": str(candidate), "rows": rows}
        return {"source": "", "rows": []}

    @app.get("/api/results/traces")
    async def results_traces() -> dict[str, object]:
        trace_candidates: list[Path] = []
        for trace_root in (
            result_root / "paper_local" / "sglang_mobilora" / "traces",
            result_root / "paper_local" / "hf_peft" / "traces",
            result_root / "traces",
        ):
            if trace_root.exists():
                trace_candidates.extend(sorted(trace_root.glob("*.jsonl")))
        if not trace_candidates:
            return {"items": []}
        items = []
        for path in trace_candidates[:6]:
            all_lines = path.read_text(encoding="utf-8").splitlines()
            lines = all_lines[:120]
            items.append(
                {
                    "name": path.name,
                    "row_count": len(all_lines),
                    "rows": [json.loads(line) for line in lines if line.strip()],
                }
            )
        return {"items": items}

    @app.get("/api/live/requests")
    async def live_requests(limit: int = 50) -> dict[str, object]:
        bounded_limit = max(1, min(int(limit), 200))
        live_path = result_root / "live_requests.jsonl"
        return {
            "source": str(live_path),
            "items": _read_jsonl_tail(live_path, bounded_limit),
        }

    @app.post("/api/live/generate")
    async def live_generate(body: dict[str, object] = Body(...)) -> JSONResponse:
        variant = str(body.get("variant") or "sglang_mobilora")
        url = sglang_server_url(config, variant)
        lora_name = body.get("lora_name")
        payload = await call_sglang_endpoint(
            backend=variant,
            prompt=str(body.get("prompt") or ""),
            url=url,
            lora_name=str(lora_name) if lora_name else None,
            app_id=str(body.get("app_id") or "demo-app"),
            app_state=str(body.get("app_state") or "foreground"),
            session_id=str(body.get("session_id") or "session-001"),
            max_new_tokens=int(body.get("max_new_tokens") or 96),
        )
        live_path = result_root / "live_requests.jsonl"
        live_record = _build_live_record(body=body, payload=payload)
        _append_jsonl(live_path, live_record)
        payload["live_record"] = live_record
        payload["live_log_path"] = str(live_path)
        return JSONResponse(payload)

    return app


def serve_dashboard(config_path: str, results_dir: str | None = None) -> None:
    import uvicorn

    config = load_config(config_path)
    app = create_dashboard_app(
        config=config,
        results_dir=Path(results_dir) if results_dir else None,
    )
    uvicorn.run(
        app,
        host=config.dashboard.host,
        port=config.dashboard.backend_port,
    )
