const defaultBase = () =>
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "";

export type RuntimeEvent = {
  id: string;
  type: string;
  phase: string;
  message: string;
  created_at: string;
  metadata: Record<string, unknown>;
};

/** Backend-owned tutor pedagogy snapshot (Phase 8). */
export type PedagogySnapshot = Record<string, unknown> | null;

export type ChatTurnMeta = {
  pedagogy_snapshot?: PedagogySnapshot;
  tutor_session_active?: boolean;
};

export async function createSession(
  baseUrl: string,
  signal?: AbortSignal,
): Promise<{
  session_id: string;
  greeting: string;
  error?: string;
  events: RuntimeEvent[];
  pedagogy_snapshot?: PedagogySnapshot;
  tutor_session_active?: boolean;
}> {
  const url = `${baseUrl}/api/session`;
  // #region agent log
  fetch("http://127.0.0.1:7242/ingest/60f20f4a-07ea-4752-b692-563be9fbeb6e", { method: "POST", headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "fbf5ed" }, body: JSON.stringify({ sessionId: "fbf5ed", runId: "run1", hypothesisId: "H3", location: "frontend/src/api.ts:createSession:url", message: "Preparing session request URL", data: { baseUrl, url, envBase: import.meta.env.VITE_API_BASE_URL ?? null, origin: typeof window !== "undefined" ? window.location.origin : null }, timestamp: Date.now() }) }).catch(() => {});
  // #endregion
  try {
    // #region agent log
    fetch("http://127.0.0.1:7242/ingest/60f20f4a-07ea-4752-b692-563be9fbeb6e", { method: "POST", headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "fbf5ed" }, body: JSON.stringify({ sessionId: "fbf5ed", runId: "run1", hypothesisId: "H1", location: "frontend/src/api.ts:createSession:beforeFetch", message: "Calling /api/session", data: { url }, timestamp: Date.now() }) }).catch(() => {});
    // #endregion
    const res = await fetch(url, { method: "POST", signal });
    // #region agent log
    fetch("http://127.0.0.1:7242/ingest/60f20f4a-07ea-4752-b692-563be9fbeb6e", { method: "POST", headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "fbf5ed" }, body: JSON.stringify({ sessionId: "fbf5ed", runId: "run1", hypothesisId: "H2", location: "frontend/src/api.ts:createSession:afterFetch", message: "Received /api/session response", data: { url, ok: res.ok, status: res.status, statusText: res.statusText, contentType: res.headers.get("content-type") }, timestamp: Date.now() }) }).catch(() => {});
    // #endregion
    if (!res.ok) {
      throw new Error(`Session failed: ${res.status}`);
    }
    return res.json();
  } catch (error) {
    // #region agent log
    fetch("http://127.0.0.1:7242/ingest/60f20f4a-07ea-4752-b692-563be9fbeb6e", { method: "POST", headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "fbf5ed" }, body: JSON.stringify({ sessionId: "fbf5ed", runId: "run1", hypothesisId: "H1", location: "frontend/src/api.ts:createSession:catch", message: "Session request failed", data: { url, errorName: error instanceof Error ? error.name : typeof error, errorMessage: error instanceof Error ? error.message : String(error) }, timestamp: Date.now() }) }).catch(() => {});
    // #endregion
    throw error;
  }
}

export async function chat(
  baseUrl: string,
  sessionId: string,
  message: string,
): Promise<{
  response: string;
  events: RuntimeEvent[];
  pedagogy_snapshot?: PedagogySnapshot;
  tutor_session_active?: boolean;
}> {
  const res = await fetch(`${baseUrl}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Chat failed: ${res.status}`);
  }
  return res.json();
}

export async function resetSession(
  baseUrl: string,
  sessionId: string,
): Promise<{
  greeting: string;
  events: RuntimeEvent[];
  pedagogy_snapshot?: PedagogySnapshot;
  tutor_session_active?: boolean;
}> {
  const res = await fetch(`${baseUrl}/api/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) {
    throw new Error(`Reset failed: ${res.status}`);
  }
  const data = (await res.json()) as {
    greeting: string;
    events: RuntimeEvent[];
    pedagogy_snapshot?: PedagogySnapshot;
    tutor_session_active?: boolean;
  };
  return {
    greeting: data.greeting,
    events: data.events,
    pedagogy_snapshot: data.pedagogy_snapshot,
    tutor_session_active: data.tutor_session_active,
  };
}

function parseSseBlocks(buffer: string): { lines: string[]; rest: string } {
  const parts = buffer.split("\n\n");
  const rest = parts.pop() ?? "";
  const lines: string[] = [];
  for (const block of parts) {
    const line = block
      .split("\n")
      .find((l) => l.startsWith("data: "));
    if (line) lines.push(line.slice(6).trim());
  }
  return { lines, rest };
}

function emitDone(
  onDone: (
    response: string,
    meta?: { pedagogy_snapshot?: PedagogySnapshot; tutor_session_active?: boolean },
  ) => void,
  data: Record<string, unknown>,
) {
  onDone(String(data.response ?? ""), {
    pedagogy_snapshot: (data.pedagogy_snapshot ?? null) as PedagogySnapshot,
    tutor_session_active: Boolean(data.tutor_session_active),
  });
}

export async function chatStream(
  baseUrl: string,
  sessionId: string,
  message: string,
  onEvent: (ev: RuntimeEvent) => void,
  onDone: (
    response: string,
    meta?: { pedagogy_snapshot?: PedagogySnapshot; tutor_session_active?: boolean },
  ) => void,
  onError: (msg: string) => void,
): Promise<void> {
  const res = await fetch(`${baseUrl}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!res.ok || !res.body) {
    onError(`Stream failed: ${res.status}`);
    return;
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const { lines, rest } = parseSseBlocks(buf);
      buf = rest;
      for (const json of lines) {
        if (!json) continue;
        let data: Record<string, unknown>;
        try {
          data = JSON.parse(json) as Record<string, unknown>;
        } catch {
          continue;
        }
        if (data.kind === "event" && data.event) {
          onEvent(data.event as RuntimeEvent);
        } else if (data.kind === "done") {
          emitDone(onDone, data);
        } else if (data.kind === "error") {
          onError(String(data.message ?? "Unknown error"));
        }
      }
    }
    if (buf.trim()) {
      const { lines } = parseSseBlocks(buf.endsWith("\n\n") ? buf : `${buf}\n\n`);
      for (const json of lines) {
        if (!json) continue;
        let data: Record<string, unknown>;
        try {
          data = JSON.parse(json) as Record<string, unknown>;
        } catch {
          continue;
        }
        if (data.kind === "event" && data.event) {
          onEvent(data.event as RuntimeEvent);
        } else if (data.kind === "done") {
          emitDone(onDone, data);
        } else if (data.kind === "error") {
          onError(String(data.message ?? "Unknown error"));
        }
      }
    }
  } catch (e) {
    onError(e instanceof Error ? e.message : String(e));
  }
}

export { defaultBase };
