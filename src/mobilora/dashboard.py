from __future__ import annotations

import json
from pathlib import Path

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


def create_dashboard_app(config: AppConfig, results_dir: Path | None = None):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel

    from mobilora.runtime import call_sglang_endpoint

    app = FastAPI(title="MobiLoRA Local Dashboard", version="0.2.0")
    result_root = results_dir or config.paths.bench_outputs

    class LiveGenerateRequest(BaseModel):
        prompt: str
        variant: str = "sglang_mobilora"
        lora_name: str | None = None
        app_id: str = "demo-app"
        app_state: str = "foreground"
        session_id: str = "session-001"
        max_new_tokens: int = 96

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
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

    @app.get("/api/results/traces")
    async def results_traces() -> dict[str, object]:
        trace_candidates = list((result_root / "traces").glob("*.jsonl"))
        if not trace_candidates:
            return {"items": []}
        items = []
        for path in trace_candidates[:6]:
            lines = path.read_text(encoding="utf-8").splitlines()[:3]
            items.append(
                {
                    "name": path.name,
                    "rows": [json.loads(line) for line in lines if line.strip()],
                }
            )
        return {"items": items}

    @app.post("/api/live/generate")
    async def live_generate(body: LiveGenerateRequest) -> JSONResponse:
        url = sglang_server_url(config, body.variant)
        payload = await call_sglang_endpoint(
            backend=body.variant,
            prompt=body.prompt,
            url=url,
            lora_name=body.lora_name,
            app_id=body.app_id,
            app_state=body.app_state,
            session_id=body.session_id,
            max_new_tokens=body.max_new_tokens,
        )
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
