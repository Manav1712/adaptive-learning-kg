"""
FastAPI bridge for the workflow_demo coach runtime (local demo UI).

Run: python -m src.workflow_demo.web_api
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from src.workflow_demo.coach_agent import CoachAgent
from src.workflow_demo.runtime_factory import build_coach_runtime

_REPO_ROOT = Path(__file__).resolve().parents[2]
if load_dotenv:
    load_dotenv(_REPO_ROOT / ".env")

_DEFAULT_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
]
_cors_raw = os.getenv("WORKFLOW_DEMO_CORS_ORIGINS", "")
ALLOWED_ORIGINS = (
    [o.strip() for o in _cors_raw.split(",") if o.strip()]
    if _cors_raw
    else _DEFAULT_ORIGINS
)

_executor = ThreadPoolExecutor(max_workers=4)
_DEBUG_LOG_PATH = _REPO_ROOT / ".cursor" / "debug-fbf5ed.log"


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    payload = {
        "sessionId": "fbf5ed",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
    }
    try:
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


class ThreadSafeEventSink:
    """Collects runtime events; optional live queue for SSE (same object coach keeps)."""

    def __init__(self) -> None:
        self._batch: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._live: Optional[queue.Queue] = None
        self._live_lock = threading.Lock()

    def set_live_queue(self, q: Optional[queue.Queue]) -> None:
        with self._live_lock:
            self._live = q

    def __call__(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self._batch.append(event)
        with self._live_lock:
            live = self._live
        if live is not None:
            live.put(event)

    def drain(self) -> List[Dict[str, Any]]:
        with self._lock:
            out = list(self._batch)
            self._batch.clear()
            return out


class WebCoachSession:
    """One browser session: coach + mutex + shared event sink."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.sink = ThreadSafeEventSink()
        self.coach: CoachAgent = build_coach_runtime(
            session_memory_path=None,
            event_callback=self.sink,
        )
        self.lock = threading.Lock()
        self.startup_error: Optional[str] = None

    def reset(self) -> None:
        with self.lock:
            self.sink = ThreadSafeEventSink()
            self.coach = build_coach_runtime(
                session_memory_path=None,
                event_callback=self.sink,
            )
            self.sink.set_live_queue(None)


_SESSIONS: Dict[str, WebCoachSession] = {}
_SESSIONS_LOCK = threading.Lock()


def _get_or_404(session_id: str) -> WebCoachSession:
    with _SESSIONS_LOCK:
        sess = _SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return sess


# Any localhost / 127.0.0.1 port (covers Vite on 5173, other dev ports, and
# cross-host usage like page at localhost + API at 127.0.0.1).
_LOCAL_DEV_ORIGIN_REGEX = r"https?://(127\.0\.0\.1|localhost)(:\d+)?$"

app = FastAPI(title="Adaptive Learning Coach API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=_LOCAL_DEV_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionResponse(BaseModel):
    session_id: str
    greeting: str = ""
    error: Optional[str] = None
    events: List[Dict[str, Any]] = Field(default_factory=list)
    pedagogy_snapshot: Optional[Dict[str, Any]] = None
    tutor_session_active: bool = False


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    events: List[Dict[str, Any]] = Field(default_factory=list)
    pedagogy_snapshot: Optional[Dict[str, Any]] = None
    tutor_session_active: bool = False


class ResetRequest(BaseModel):
    session_id: str


class ResetResponse(BaseModel):
    ok: bool = True
    greeting: str = ""
    events: List[Dict[str, Any]] = Field(default_factory=list)
    pedagogy_snapshot: Optional[Dict[str, Any]] = None
    tutor_session_active: bool = False


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/session", response_model=SessionResponse)
def create_session(request: Request) -> SessionResponse:
    session_id = uuid.uuid4().hex
    # region agent log
    _debug_log(
        run_id="run2",
        hypothesis_id="H1",
        location="src/workflow_demo/web_api.py:create_session:entry",
        message="Received /api/session request",
        data={
            "session_id": session_id,
            "origin": request.headers.get("origin"),
            "host": request.headers.get("host"),
        },
    )
    # endregion
    try:
        sess = WebCoachSession(session_id)
        greeting = sess.coach.initial_greeting()
        events = sess.sink.drain()
    except Exception as exc:  # noqa: BLE001 — demo boundary
        # region agent log
        _debug_log(
            run_id="run2",
            hypothesis_id="H5",
            location="src/workflow_demo/web_api.py:create_session:exception",
            message="Session initialization failed",
            data={"session_id": session_id, "error": str(exc)},
        )
        # endregion
        return SessionResponse(
            session_id=session_id,
            greeting="",
            error=str(exc),
            events=[],
        )

    with _SESSIONS_LOCK:
        _SESSIONS[session_id] = sess
    # region agent log
    _debug_log(
        run_id="run2",
        hypothesis_id="H5",
        location="src/workflow_demo/web_api.py:create_session:success",
        message="Session initialized successfully",
        data={"session_id": session_id, "events_count": len(events)},
    )
    # endregion
    return SessionResponse(
        session_id=session_id,
        greeting=greeting or "",
        events=events,
        pedagogy_snapshot=sess.coach.get_pedagogy_snapshot_for_api(),
        tutor_session_active=sess.coach.tutor_session_active_for_api(),
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    sess = _get_or_404(body.session_id)
    loop = asyncio.get_running_loop()

    def run_turn() -> str:
        with sess.lock:
            sess.sink.drain()
            return sess.coach.process_turn(body.message)

    try:
        text = await loop.run_in_executor(_executor, run_turn)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    events = sess.sink.drain()
    return ChatResponse(
        response=text or "",
        events=events,
        pedagogy_snapshot=sess.coach.get_pedagogy_snapshot_for_api(),
        tutor_session_active=sess.coach.tutor_session_active_for_api(),
    )


def _sse_format(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, default=str)}\n\n"


@app.post("/api/chat/stream")
async def chat_stream(body: ChatRequest) -> StreamingResponse:
    sess = _get_or_404(body.session_id)
    live: queue.Queue = queue.Queue()
    loop = asyncio.get_running_loop()
    result_holder: Dict[str, Any] = {}
    error_holder: Dict[str, Any] = {}

    def run_turn_streaming() -> None:
        try:
            with sess.lock:
                sess.sink.set_live_queue(live)
                try:
                    sess.sink.drain()
                    result_holder["text"] = sess.coach.process_turn(body.message)
                finally:
                    sess.sink.set_live_queue(None)
        except Exception as exc:  # noqa: BLE001
            error_holder["err"] = str(exc)
        finally:
            live.put(None)  # type: ignore[arg-type]

    thread = threading.Thread(target=run_turn_streaming, daemon=True)
    thread.start()

    async def event_gen():
        while True:
            item = await loop.run_in_executor(_executor, live.get)
            if item is None:
                break
            yield _sse_format({"kind": "event", "event": item})
        thread.join(timeout=300)
        if error_holder.get("err"):
            yield _sse_format({"kind": "error", "message": error_holder["err"]})
        else:
            yield _sse_format(
                {
                    "kind": "done",
                    "response": result_holder.get("text", ""),
                    "pedagogy_snapshot": sess.coach.get_pedagogy_snapshot_for_api(),
                    "tutor_session_active": sess.coach.tutor_session_active_for_api(),
                }
            )

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/reset", response_model=ResetResponse)
def reset_session(body: ResetRequest) -> ResetResponse:
    sess = _get_or_404(body.session_id)
    with sess.lock:
        sess.sink.drain()
        sess.reset()
        greeting = sess.coach.initial_greeting()
        events = sess.sink.drain()
    return ResetResponse(
        ok=True,
        greeting=greeting or "",
        events=events,
        pedagogy_snapshot=sess.coach.get_pedagogy_snapshot_for_api(),
        tutor_session_active=sess.coach.tutor_session_active_for_api(),
    )


def main() -> None:
    import uvicorn

    host = os.getenv("WORKFLOW_DEMO_API_HOST", "127.0.0.1")
    port = int(os.getenv("WORKFLOW_DEMO_API_PORT", "8001"))
    uvicorn.run(
        "src.workflow_demo.web_api:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
