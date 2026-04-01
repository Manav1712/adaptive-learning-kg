import { useCallback, useEffect, useRef, useState } from "react";
import {
  chat,
  chatStream,
  createSession,
  defaultBase,
  resetSession,
  type PedagogySnapshot,
  type RuntimeEvent,
} from "./api";
import "./App.css";

type ChatRow = { role: "user" | "assistant"; text: string };

function formatEventMeta(m: Record<string, unknown>): string {
  const keys = Object.keys(m);
  if (!keys.length) return "";
  const parts = keys
    .slice(0, 8)
    .map((k) => `${k}=${String(m[k]).slice(0, 48)}`);
  return `  ${parts.join(" · ")}`;
}

export default function App() {
  const baseUrl = defaultBase();
  const bootRunRef = useRef(0);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [rows, setRows] = useState<ChatRow[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState<"boot" | "send" | null>("boot");
  const [bootError, setBootError] = useState<string | null>(null);
  const [sendError, setSendError] = useState<string | null>(null);
  const [statusLines, setStatusLines] = useState<string[]>([]);
  const [useStream, setUseStream] = useState(true);
  const [pedagogySnapshot, setPedagogySnapshot] = useState<PedagogySnapshot>(null);
  const [tutorSessionActive, setTutorSessionActive] = useState(false);
  const [pedagogyPanelOpen, setPedagogyPanelOpen] = useState(true);
  const listRef = useRef<HTMLDivElement>(null);

  const [pedagogyOnlyRuntime, setPedagogyOnlyRuntime] = useState(false);
  const [pedagogyLines, setPedagogyLines] = useState<string[]>([]);

  const pushStatus = useCallback((events: RuntimeEvent[]) => {
    if (!events.length) return;
    setStatusLines((prev) => {
      const next = [...prev];
      for (const e of events) {
        const line = `[${e.phase}] ${e.type}: ${e.message}`;
        next.push(line);
        if (e.phase === "pedagogy" && e.metadata && Object.keys(e.metadata).length) {
          next.push(formatEventMeta(e.metadata as Record<string, unknown>));
        }
      }
      return next.slice(-40);
    });
    setPedagogyLines((prev) => {
      const next = [...prev];
      for (const e of events) {
        if (e.phase !== "pedagogy") continue;
        const line = `[pedagogy] ${e.type}: ${e.message}`;
        next.push(line);
        if (e.metadata && Object.keys(e.metadata).length) {
          next.push(formatEventMeta(e.metadata as Record<string, unknown>));
        }
      }
      return next.slice(-20);
    });
  }, []);

  useEffect(() => {
    let cancelled = false;
    const thisRun = ++bootRunRef.current;
    const controller = new AbortController();
    (async () => {
      setBootError(null);
      setLoading("boot");
      try {
        const data = await createSession(baseUrl, controller.signal);
        if (cancelled || thisRun !== bootRunRef.current) return;
        if (data.error) {
          setBootError(data.error);
          setSessionId(null);
          return;
        }
        setBootError(null);
        setSessionId(data.session_id);
        pushStatus(data.events);
        setPedagogySnapshot(data.pedagogy_snapshot ?? null);
        setTutorSessionActive(Boolean(data.tutor_session_active));
        if (data.greeting) {
          setRows([{ role: "assistant", text: data.greeting }]);
        }
      } catch (e) {
        // #region agent log
        fetch("http://127.0.0.1:7242/ingest/60f20f4a-07ea-4752-b692-563be9fbeb6e", { method: "POST", headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "fbf5ed" }, body: JSON.stringify({ sessionId: "fbf5ed", runId: "run1", hypothesisId: "H4", location: "frontend/src/App.tsx:boot:catch", message: "Boot catch triggered", data: { baseUrl, errorName: e instanceof Error ? e.name : typeof e, errorMessage: e instanceof Error ? e.message : String(e) }, timestamp: Date.now() }) }).catch(() => {});
        // #endregion
        if (
          !cancelled &&
          thisRun === bootRunRef.current &&
          !(e instanceof DOMException && e.name === "AbortError")
        ) {
          setBootError(e instanceof Error ? e.message : String(e));
        }
      } finally {
        if (!cancelled && thisRun === bootRunRef.current) setLoading(null);
      }
    })();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [baseUrl, pushStatus]);

  useEffect(() => {
    listRef.current?.scrollTo({
      top: listRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [rows]);

  const appendAssistant = (text: string) => {
    setRows((r) => [...r, { role: "assistant", text }]);
  };

  const onSend = async () => {
    const msg = input.trim();
    if (!msg || !sessionId || loading) return;
    setInput("");
    setSendError(null);
    setRows((r) => [...r, { role: "user", text: msg }]);
    setLoading("send");

    if (useStream) {
      let assistantText = "";
      await chatStream(
        baseUrl,
        sessionId,
        msg,
        (ev) => pushStatus([ev]),
        (response, meta) => {
          assistantText = response;
          if (meta?.pedagogy_snapshot !== undefined) {
            setPedagogySnapshot(meta.pedagogy_snapshot ?? null);
          }
          if (meta?.tutor_session_active !== undefined) {
            setTutorSessionActive(Boolean(meta.tutor_session_active));
          }
        },
        (err) => setSendError(err),
      );
      if (assistantText) appendAssistant(assistantText);
      setLoading(null);
      return;
    }

    try {
      const { response, events, pedagogy_snapshot, tutor_session_active } =
        await chat(baseUrl, sessionId, msg);
      pushStatus(events);
      if (pedagogy_snapshot !== undefined) setPedagogySnapshot(pedagogy_snapshot ?? null);
      if (tutor_session_active !== undefined) setTutorSessionActive(Boolean(tutor_session_active));
      appendAssistant(response);
    } catch (e) {
      setSendError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(null);
    }
  };

  const onReset = async () => {
    if (!sessionId || loading) return;
    setSendError(null);
    setLoading("send");
    try {
      const { greeting, events, pedagogy_snapshot, tutor_session_active } =
        await resetSession(baseUrl, sessionId);
      pushStatus(events);
      if (pedagogy_snapshot !== undefined) setPedagogySnapshot(pedagogy_snapshot ?? null);
      if (tutor_session_active !== undefined) setTutorSessionActive(Boolean(tutor_session_active));
      setRows(greeting ? [{ role: "assistant", text: greeting }] : []);
    } catch (e) {
      setSendError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(null);
    }
  };

  const displayBase =
    baseUrl === "" ? "(same origin — Vite proxy /api)" : baseUrl;

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Adaptive learning coach</h1>
          <p>Demo UI for the workflow coach, tutor, and FAQ runtime.</p>
        </div>
        <div className="header-actions">
          <span className="api-hint" title="Set VITE_API_BASE_URL if not using proxy">
            API: {displayBase}
          </span>
          <button
            type="button"
            className="btn"
            onClick={onReset}
            disabled={!sessionId || loading !== null}
          >
            Reset session
          </button>
        </div>
      </header>

      {bootError && (
        <div className="banner-error" role="alert">
          <div>
            Could not start session: {bootError}
          </div>
          {/failed to fetch|networkerror|load failed/i.test(bootError) && (
            <div style={{ marginTop: "0.5rem", fontSize: "0.8rem" }}>
              <strong>Network:</strong> Start the API from the repo root:{" "}
              <code style={{ whiteSpace: "nowrap" }}>
                venv/bin/python -m src.workflow_demo.web_api
              </code>{" "}
              (note the <code>/</code> before <code>venv</code>). Or avoid
              cross-origin: unset{" "}
              <code>VITE_API_BASE_URL</code> and rely on the Vite dev proxy so
              requests go to <code>/api</code> on port 5173.
            </div>
          )}
          {!/failed to fetch|networkerror|load failed/i.test(bootError) && (
            <div style={{ marginTop: "0.5rem", fontSize: "0.8rem" }}>
              If this mentions quota or API keys, check{" "}
              <code>OPENAI_API_KEY</code> and billing for embeddings.
            </div>
          )}
        </div>
      )}
      {loading === "boot" && !bootError && (
        <p className="banner-loading">Initializing coach (may take a minute)…</p>
      )}

      {sessionId && (
        <>
          {statusLines.length > 0 && (
            <div className="status-bar" aria-live="polite">
              <div className="status-bar-head">
                <strong>Runtime</strong>
                <label className="status-filter">
                  <input
                    type="checkbox"
                    checked={pedagogyOnlyRuntime}
                    onChange={(e) => setPedagogyOnlyRuntime(e.target.checked)}
                  />
                  Pedagogy only
                </label>
              </div>
              {(pedagogyOnlyRuntime
                ? statusLines.filter((l) => l.includes("[pedagogy]"))
                : statusLines
              )
                .slice(-12)
                .map((line, i) => (
                  <div key={`${line}-${i}`} className="status-line">
                    {line}
                  </div>
                ))}
            </div>
          )}

          {pedagogyLines.length > 0 && (
            <div className="status-bar status-bar-pedagogy" aria-live="polite">
              <strong>Pedagogy events</strong>
              {pedagogyLines.map((line, i) => (
                <div key={`p-${line}-${i}`} className="status-line">
                  {line}
                </div>
              ))}
            </div>
          )}

          {(tutorSessionActive || pedagogySnapshot != null) && (
            <div className="pedagogy-panel">
              <button
                type="button"
                className="pedagogy-panel-toggle"
                onClick={() => setPedagogyPanelOpen((o) => !o)}
                aria-expanded={pedagogyPanelOpen}
              >
                Tutor pedagogy (backend snapshot) {pedagogyPanelOpen ? "▼" : "▶"}
              </button>
              {pedagogyPanelOpen &&
                (pedagogySnapshot ? (
                  <pre className="pedagogy-panel-pre">
                    {JSON.stringify(pedagogySnapshot, null, 2)}
                  </pre>
                ) : (
                  <p className="pedagogy-panel-empty">No snapshot yet (first turn may be sparse).</p>
                ))}
            </div>
          )}

          <div className="messages" ref={listRef}>
            {rows.map((row, i) => (
              <div
                key={i}
                className={`msg ${row.role === "user" ? "msg-user" : "msg-assistant"}`}
              >
                <div className="msg-meta">
                  {row.role === "user" ? "You" : "Assistant"}
                </div>
                {row.text}
              </div>
            ))}
          </div>

          {sendError && (
            <div className="banner-error" role="alert">
              {sendError}
            </div>
          )}

          <div className="composer">
            <div className="composer-options">
              <label>
                <input
                  type="checkbox"
                  checked={useStream}
                  onChange={(e) => setUseStream(e.target.checked)}
                />
                Stream status (SSE)
              </label>
            </div>
            <div className="composer-row">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask a tutoring or course question…"
                rows={3}
                disabled={loading !== null}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void onSend();
                  }
                }}
              />
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => void onSend()}
                disabled={loading !== null || !input.trim()}
              >
                {loading === "send" ? "…" : "Send"}
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
