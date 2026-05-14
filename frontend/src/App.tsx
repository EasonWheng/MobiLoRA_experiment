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

const pretty = (value: unknown) => JSON.stringify(value, null, 2);

export function App() {
  const [status, setStatus] = useState<StatusItem[]>([]);
  const [summary, setSummary] = useState<Record<string, unknown>>({});
  const [traces, setTraces] = useState<TraceItem[]>([]);
  const [response, setResponse] = useState<string>("No request yet.");
  const [loading, setLoading] = useState(false);
  const [form, setForm] = useState({
    variant: "sglang_mobilora",
    lora_name: "",
    app_id: "demo-app",
    app_state: "foreground",
    session_id: "session-001",
    prompt: "Explain what context-aware eviction means in MobiLoRA.",
    max_new_tokens: 96,
  });

  useEffect(() => {
    void refresh();
  }, []);

  async function refresh() {
    const [statusRes, summaryRes, tracesRes] = await Promise.all([
      fetch("/api/server-status").then((res) => res.json()),
      fetch("/api/results/summary").then((res) => res.json()),
      fetch("/api/results/traces").then((res) => res.json()),
    ]);
    setStatus(statusRes.items ?? []);
    setSummary(summaryRes);
    setTraces(tracesRes.items ?? []);
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

  return (
    <div className="page-shell">
      <header className="hero">
        <div className="hero-copy panel">
          <p className="kicker">Local paper-scale experiment surface</p>
          <h1>MobiLoRA with real local SGLang calls and explainable traces.</h1>
          <p className="hero-text">
            This dashboard is designed for paper-style local experiments on a
            constrained GPU. It keeps the benchmark outputs, live prompt path,
            and trace evidence in one place.
          </p>
        </div>
        <div className="hero-status panel">
          <div className="panel-header">
            <h2>Server status</h2>
            <button className="ghost-button" onClick={() => void refresh()}>
              Refresh
            </button>
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
        </div>
      </header>

      <main className="dashboard-grid">
        <section className="panel live-panel">
          <div className="panel-header">
            <h2>Live Prompt</h2>
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
                <input
                  value={form.lora_name}
                  onChange={(event) => setForm({ ...form, lora_name: event.target.value })}
                  placeholder="Qwen LoRA alias"
                />
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
                    setForm({ ...form, max_new_tokens: Number(event.target.value) || 96 })
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
              {loading ? "Sending..." : "Send request"}
            </button>
          </form>
          <pre className="output-block">{response}</pre>
        </section>

        <section className="panel summary-panel">
          <div className="panel-header">
            <h2>Experiment Runs</h2>
          </div>
          <pre className="output-block">{pretty(summary)}</pre>
        </section>

        <section className="panel trace-panel">
          <div className="panel-header">
            <h2>Trace Explorer</h2>
          </div>
          <div className="trace-stack">
            {traces.map((item) => (
              <article className="trace-card" key={item.name}>
                <h3>{item.name}</h3>
                <pre className="output-block small">{pretty(item.rows)}</pre>
              </article>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
