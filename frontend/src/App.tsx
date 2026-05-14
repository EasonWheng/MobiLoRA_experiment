import { FormEvent, useEffect, useState } from "react";

type StatusItem = {
  variant: string;
  url: string;
  healthy: boolean;
  detail: string;
};

type TraceItem = {
  name: string;
  rows: Record<string, unknown>[];
};

type ResultRow = {
  system?: string;
  scenario: string;
  workload: string;
  variant: string;
  timing_source: string;
  request_count: string;
  median_prefill_latency_ms: string;
  p95_prefill_latency_ms: string;
  median_end_to_end_latency_ms: string;
  cache_hit_ratio: string;
  compression_ratio: string;
  avg_persisted_kv_mb: string;
  peak_persisted_kv_mb: string;
  quality_score: string;
  avg_generated_tokens: string;
  avg_reuse_tokens: string;
  avg_similarity: string;
  avg_error_bound: string;
};

type FigureSpec = {
  id: string;
  title: string;
  source: string;
  note: string;
};

const figureSpecs: FigureSpec[] = [
  {
    id: "fig5",
    title: "Figure 5 style: KV memory trace",
    source: "Driven by request-level traces",
    note: "Plots persisted KV memory per request, with cache pressure and eviction evidence available in each trace row.",
  },
  {
    id: "fig6",
    title: "Figure 6 style: quality CDF",
    source: "Driven by quality_score",
    note: "Uses the current local quality score as a BERTScore proxy until full BERTScore runs are enabled.",
  },
  {
    id: "fig7",
    title: "Figure 7 style: TTFT ablations",
    source: "Driven by paper_results.csv",
    note: "Compares plain PEFT, prefix-only, no-delta, no-context, and full MobiLoRA variants.",
  },
];

const variantOrder = [
  "plain_peft",
  "prefix_only",
  "mobilora_no_delta",
  "mobilora_no_ctx",
  "mobilora_full",
];

const adapterOptions = ["qwen-email-1300", "qwen-sms-3400"];

const pretty = (value: unknown) => JSON.stringify(value, null, 2);

const asNumber = (value: unknown, fallback = 0) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const formatMs = (value: number) => `${value.toFixed(value >= 100 ? 0 : 1)} ms`;

const shortVariant = (variant: string) =>
  variant
    .replace("mobilora_", "mobi_")
    .replace("plain_peft", "plain")
    .replace("prefix_only", "prefix");

function pickMobiRows(rows: ResultRow[]) {
  return rows
    .filter((row) => (row.system ?? "default") === "sglang_mobilora")
    .sort((a, b) => variantOrder.indexOf(a.variant) - variantOrder.indexOf(b.variant));
}

function latestFullRow(rows: ResultRow[]) {
  return pickMobiRows(rows).find((row) => row.variant === "mobilora_full");
}

function MetricTile({
  label,
  value,
  caption,
}: {
  label: string;
  value: string;
  caption: string;
}) {
  return (
    <article className="metric-tile">
      <span>{label}</span>
      <strong>{value}</strong>
      <p>{caption}</p>
    </article>
  );
}

function BarChart({ rows }: { rows: ResultRow[] }) {
  const values = rows.map((row) => asNumber(row.median_prefill_latency_ms));
  const maxValue = Math.max(...values, 1);

  return (
    <div className="bar-chart" aria-label="TTFT ablation bar chart">
      {rows.map((row) => {
        const value = asNumber(row.median_prefill_latency_ms);
        const height = Math.max((value / maxValue) * 100, 4);
        return (
          <div className="bar-item" key={`${row.scenario}-${row.variant}`}>
            <div className="bar-track">
              <div
                className={`bar-fill ${row.variant === "mobilora_full" ? "winner" : ""}`}
                style={{ height: `${height}%` }}
              />
            </div>
            <strong>{formatMs(value)}</strong>
            <span>{shortVariant(row.variant)}</span>
          </div>
        );
      })}
    </div>
  );
}

function MemoryTrace({ traces }: { traces: TraceItem[] }) {
  const trace =
    traces.find((item) => item.name.includes("mobilora_full")) ?? traces[0];
  const rows = trace?.rows ?? [];
  const values = rows.map((row) => asNumber(row.persisted_kv_mb));
  const peak = Math.max(...values, 1);
  const width = 720;
  const height = 220;
  const points = values.map((value, index) => {
    const x = values.length <= 1 ? 0 : (index / (values.length - 1)) * width;
    const y = height - (value / peak) * (height - 18) - 8;
    return `${x},${y}`;
  });
  const area = points.length
    ? `0,${height} ${points.join(" ")} ${width},${height}`
    : "";

  return (
    <div className="memory-chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="KV memory trace">
        <defs>
          <linearGradient id="memoryFill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#e37a3f" stopOpacity="0.45" />
            <stop offset="100%" stopColor="#e37a3f" stopOpacity="0.04" />
          </linearGradient>
        </defs>
        <line x1="0" x2={width} y1={height - 1} y2={height - 1} className="axis" />
        <line x1="0" x2="0" y1="0" y2={height} className="axis" />
        {area ? <polygon points={area} fill="url(#memoryFill)" /> : null}
        {points.length ? <polyline points={points.join(" ")} className="memory-line" /> : null}
        {values.map((value, index) => {
          const [x, y] = points[index].split(",").map(Number);
          return <circle key={`${value}-${index}`} cx={x} cy={y} r="4" className="dot" />;
        })}
      </svg>
      <div className="chart-caption">
        <span>{trace?.name ?? "No trace loaded"}</span>
        <strong>Peak {peak.toFixed(2)} MB</strong>
      </div>
    </div>
  );
}

function QualityCdf({ traces }: { traces: TraceItem[] }) {
  const series = traces
    .filter((item) => item.name.includes("mobilora"))
    .slice(0, 3)
    .map((item) => ({
      name: item.name.replace("S1-local_conversation_", "").replace(".jsonl", ""),
      values: item.rows
        .map((row) => asNumber(row.quality_score, 1))
        .sort((a, b) => a - b),
    }));

  const width = 720;
  const height = 210;
  const minQuality = 0.94;
  const maxQuality = 1.0;

  return (
    <div className="quality-cdf">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Quality CDF">
        <line x1="0" x2={width} y1={height - 1} y2={height - 1} className="axis" />
        <line x1="0" x2="0" y1="0" y2={height} className="axis" />
        {series.map((item, seriesIndex) => {
          const points = item.values.map((value, index) => {
            const x =
              ((Math.min(Math.max(value, minQuality), maxQuality) - minQuality) /
                (maxQuality - minQuality)) *
              width;
            const y = height - ((index + 1) / Math.max(item.values.length, 1)) * (height - 14);
            return `${x},${y}`;
          });
          return (
            <polyline
              key={item.name}
              points={points.join(" ")}
              className={`cdf-line cdf-${seriesIndex}`}
            />
          );
        })}
      </svg>
      <div className="legend-row">
        {series.map((item, index) => (
          <span className={`legend-item cdf-${index}`} key={item.name}>
            {item.name}
          </span>
        ))}
      </div>
    </div>
  );
}

function ResultsTable({ rows }: { rows: ResultRow[] }) {
  return (
    <div className="result-table-wrap">
      <table className="result-table">
        <thead>
          <tr>
            <th>System</th>
            <th>Variant</th>
            <th>TTFT median</th>
            <th>Hit ratio</th>
            <th>KV compression</th>
            <th>Quality</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.system}-${row.variant}`}>
              <td>{row.system ?? "default"}</td>
              <td>{row.variant}</td>
              <td>{formatMs(asNumber(row.median_prefill_latency_ms))}</td>
              <td>{(asNumber(row.cache_hit_ratio) * 100).toFixed(0)}%</td>
              <td>{asNumber(row.compression_ratio).toFixed(2)}x</td>
              <td>{asNumber(row.quality_score).toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TraceEvidence({ traces }: { traces: TraceItem[] }) {
  const evidenceRows = traces
    .flatMap((trace) =>
      trace.rows.map((row) => ({
        trace: trace.name,
        ...row,
      })),
    )
    .filter((row) => asNumber(row.hit) > 0 || asNumber(row.compression_ratio) > 1)
    .slice(0, 5);

  return (
    <div className="evidence-list">
      {evidenceRows.map((row, index) => (
        <article className="evidence-card" key={`${row.trace}-${index}`}>
          <span>{String(row.variant ?? "variant")}</span>
          <strong>{String(row.request_id ?? "request")}</strong>
          <p>
            anchor {String(row.anchor_id || "none")} · reuse {String(row.reuse_tokens ?? 0)} tokens ·
            compression {asNumber(row.compression_ratio).toFixed(2)}x
          </p>
        </article>
      ))}
      {!evidenceRows.length ? <p className="empty-note">No reusable-prefix trace rows loaded yet.</p> : null}
    </div>
  );
}

export function App() {
  const [status, setStatus] = useState<StatusItem[]>([]);
  const [summary, setSummary] = useState<Record<string, unknown>>({});
  const [traces, setTraces] = useState<TraceItem[]>([]);
  const [rows, setRows] = useState<ResultRow[]>([]);
  const [response, setResponse] = useState<string>("No request yet.");
  const [loading, setLoading] = useState(false);
  const [form, setForm] = useState({
    variant: "sglang_mobilora",
    lora_name: "qwen-email-1300",
    app_id: "mail-app",
    app_state: "foreground",
    session_id: "session-001",
    prompt: "Explain what context-aware eviction means in MobiLoRA.",
    max_new_tokens: 32,
  });

  useEffect(() => {
    void refresh();
  }, []);

  async function refresh() {
    const [statusRes, summaryRes, tracesRes, rowsRes] = await Promise.all([
      fetch("/api/server-status").then((res) => res.json()),
      fetch("/api/results/summary").then((res) => res.json()),
      fetch("/api/results/traces").then((res) => res.json()),
      fetch("/api/results/rows").then((res) => res.json()),
    ]);
    setStatus(statusRes.items ?? []);
    setSummary(summaryRes);
    setTraces(tracesRes.items ?? []);
    setRows(rowsRes.rows ?? []);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setResponse("Sending request...");
    try {
      const res = await fetch("/api/live/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const payload = await res.json();
      setResponse(pretty(payload));
      await refresh();
    } catch (error) {
      setResponse(String(error));
    } finally {
      setLoading(false);
    }
  }

  const mobiRows = pickMobiRows(rows);
  const fullRow = latestFullRow(rows);
  const baselines = summary.baselines && typeof summary.baselines === "object"
    ? Object.keys(summary.baselines as Record<string, unknown>)
    : [];

  return (
    <div className="page-shell">
      <header className="app-hero">
        <nav className="topbar">
          <div className="brand-mark">
            <span>ML</span>
            <strong>MobiLoRA Lab</strong>
          </div>
          <div className="topbar-actions">
            <span>{rows.length} result rows</span>
            <button className="ghost-button" onClick={() => void refresh()}>
              Refresh data
            </button>
          </div>
        </nav>

        <section className="hero-grid">
          <div className="hero-copy">
            <span className="section-label">Local paper reproduction cockpit</span>
            <h1>Turn MobiLoRA paper figures into live, explainable experiment views.</h1>
            <p>
              The dashboard reads local benchmark CSVs and request traces, then
              rebuilds the paper's memory, quality, and TTFT views from the data
              your SGLang service just produced.
            </p>
            <div className="hero-pills">
              {baselines.map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>
          </div>

          <aside className="hero-console panel">
            <div className="panel-header">
              <div>
                <span className="section-label">SGLang status</span>
                <h2>Live serving path</h2>
              </div>
              <span className="pulse-dot" />
            </div>
            <div className="status-list">
              {status.map((item) => (
                <article className="status-item" key={item.variant}>
                  <div>
                    <strong>{item.variant}</strong>
                    <p>{item.url}</p>
                  </div>
                  <span className={item.healthy ? "status-chip up" : "status-chip down"}>
                    {item.healthy ? "healthy" : "offline"}
                  </span>
                </article>
              ))}
            </div>
          </aside>
        </section>
      </header>

      <main className="content-stack">
        <section className="metric-strip">
          <MetricTile
            label="Median TTFT"
            value={fullRow ? formatMs(asNumber(fullRow.median_prefill_latency_ms)) : "--"}
            caption="mobilora_full, local smoke"
          />
          <MetricTile
            label="Cache hit ratio"
            value={fullRow ? `${(asNumber(fullRow.cache_hit_ratio) * 100).toFixed(0)}%` : "--"}
            caption="requests with reusable anchors"
          />
          <MetricTile
            label="KV compression"
            value={fullRow ? `${asNumber(fullRow.compression_ratio).toFixed(2)}x` : "--"}
            caption="raw KV divided by persisted KV"
          />
          <MetricTile
            label="Quality score"
            value={fullRow ? asNumber(fullRow.quality_score).toFixed(3) : "--"}
            caption="BERTScore proxy in smoke mode"
          />
        </section>

        <section className="figure-grid">
          <article className="figure-panel panel wide-panel">
            <div className="panel-header">
              <div>
                <span className="section-label">{figureSpecs[2].source}</span>
                <h2>{figureSpecs[2].title}</h2>
              </div>
              <p>{figureSpecs[2].note}</p>
            </div>
            <BarChart rows={mobiRows} />
          </article>

          <article className="figure-panel panel">
            <div className="panel-header vertical">
              <span className="section-label">{figureSpecs[0].source}</span>
              <h2>{figureSpecs[0].title}</h2>
              <p>{figureSpecs[0].note}</p>
            </div>
            <MemoryTrace traces={traces} />
          </article>

          <article className="figure-panel panel">
            <div className="panel-header vertical">
              <span className="section-label">{figureSpecs[1].source}</span>
              <h2>{figureSpecs[1].title}</h2>
              <p>{figureSpecs[1].note}</p>
            </div>
            <QualityCdf traces={traces} />
          </article>
        </section>

        <section className="workbench-grid">
          <article className="panel live-panel">
            <div className="panel-header">
              <div>
                <span className="section-label">Live Prompt</span>
                <h2>Ask the local SGLang service</h2>
              </div>
            </div>
            <form className="live-form" onSubmit={handleSubmit}>
              <div className="form-row">
                <label>
                  Variant
                  <select
                    value={form.variant}
                    onChange={(event) => setForm({ ...form, variant: event.target.value })}
                  >
                    <option value="sglang_mobilora">sglang_mobilora</option>
                    <option value="sglang_stock_lora">sglang_stock_lora</option>
                  </select>
                </label>
                <label>
                  Adapter alias
                  <select
                    value={form.lora_name}
                    onChange={(event) => setForm({ ...form, lora_name: event.target.value })}
                  >
                    {adapterOptions.map((adapter) => (
                      <option value={adapter} key={adapter}>
                        {adapter}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="form-row">
                <label>
                  App id
                  <input
                    value={form.app_id}
                    onChange={(event) => setForm({ ...form, app_id: event.target.value })}
                  />
                </label>
                <label>
                  App state
                  <select
                    value={form.app_state}
                    onChange={(event) => setForm({ ...form, app_state: event.target.value })}
                  >
                    <option value="foreground">foreground</option>
                    <option value="background">background</option>
                    <option value="killed">killed</option>
                  </select>
                </label>
              </div>
              <div className="form-row">
                <label>
                  Session id
                  <input
                    value={form.session_id}
                    onChange={(event) => setForm({ ...form, session_id: event.target.value })}
                  />
                </label>
                <label>
                  Max new tokens
                  <input
                    type="number"
                    min={1}
                    max={256}
                    value={form.max_new_tokens}
                    onChange={(event) =>
                      setForm({ ...form, max_new_tokens: Number(event.target.value) || 32 })
                    }
                  />
                </label>
              </div>
              <label>
                Prompt
                <textarea
                  value={form.prompt}
                  onChange={(event) => setForm({ ...form, prompt: event.target.value })}
                />
              </label>
              <button className="submit-button" disabled={loading} type="submit">
                {loading ? "Sending..." : "Send live request"}
              </button>
            </form>
            <pre className="output-block response-block">{response}</pre>
          </article>

          <article className="panel evidence-panel">
            <div className="panel-header vertical">
              <span className="section-label">Trace Explorer</span>
              <h2>Direct evidence for the three mechanisms</h2>
              <p>
                Rows below are pulled from JSONL traces and expose cross-adapter
                prefix hits, delta compression, and anchor selection.
              </p>
            </div>
            <TraceEvidence traces={traces} />
          </article>
        </section>

        <section className="panel table-panel">
          <div className="panel-header">
            <div>
              <span className="section-label">Table 2 style</span>
              <h2>TTFT and quality summary</h2>
            </div>
            <p>Rows stay linked to the generated CSV, so every aggregate can be traced back.</p>
          </div>
          <ResultsTable rows={rows} />
        </section>
      </main>
    </div>
  );
}
