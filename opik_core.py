"""
opik_core.py
------------
Parsing + normalisation layer for Opik exports (CSV / XLSX / JSON / JSONL).

Opik's data model, in one paragraph:

    Thread  -> a conversation (grouping key: thread_id)
      Trace -> one end-to-end request (id, name, input, output, start/end time,
               feedback_scores, metadata, tags, total_estimated_cost, usage)
        Span -> a unit of work inside a trace (id, trace_id, parent_span_id,
                type = llm | tool | general | guardrail, model, provider,
                usage{prompt_tokens, completion_tokens, total_tokens})

Spans form a tree via parent_span_id (root spans have none). Traces group into
threads via thread_id. Everything JSON-ish (input/output/metadata/usage/
feedback_scores) arrives as a *string* in CSV exports, so it must be re-parsed.

This module is UI-free so it can be unit tested or reused in a notebook.
"""

from __future__ import annotations

import ast
import io
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

# --------------------------------------------------------------------------
# Column vocabulary
# --------------------------------------------------------------------------

JSON_COLUMNS = (
    "input",
    "output",
    "metadata",
    "usage",
    "feedback_scores",
    "tags",
    "error_info",
    "attributes",
    "comments",
)

# Aliases seen across the UI CSV export, the REST API and the SDK.
ALIASES: dict[str, str] = {
    "trace id": "trace_id",
    "traceid": "trace_id",
    "span id": "id",
    "spanid": "id",
    "parent span id": "parent_span_id",
    "parentspanid": "parent_span_id",
    "parent_id": "parent_span_id",
    "thread id": "thread_id",
    "threadid": "thread_id",
    "start time": "start_time",
    "starttime": "start_time",
    "end time": "end_time",
    "endtime": "end_time",
    "duration (ms)": "duration",
    "duration_ms": "duration",
    "total estimated cost": "total_estimated_cost",
    "total_cost": "total_estimated_cost",
    "estimated_cost": "total_estimated_cost",
    "span type": "type",
    "span_type": "type",
    "created_at": "start_time",
    "project": "project_name",
}

SPAN_HINTS = ("parent_span_id", "span_id", "span_type")
TRACE_HINTS = ("thread_id", "trace_count", "input_output")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def read_any(file_obj: Any, filename: str) -> pd.DataFrame:
    """Load a CSV / XLSX / JSON / JSONL upload into a DataFrame."""
    name = (filename or "").lower()
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(file_obj)
    if name.endswith(".jsonl") or name.endswith(".ndjson"):
        raw = _as_text(file_obj)
        rows = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
        return pd.json_normalize(rows, max_level=1)
    if name.endswith(".json"):
        raw = _as_text(file_obj)
        payload = json.loads(raw)
        if isinstance(payload, dict):
            # REST responses wrap the rows: {"content": [...]} / {"data": [...]}
            for key in ("content", "data", "traces", "spans", "items", "results"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
            else:
                payload = [payload]
        return pd.json_normalize(payload, max_level=1)
    # default: delimited text
    return pd.read_csv(file_obj)


def _as_text(file_obj: Any) -> str:
    data = file_obj.read() if hasattr(file_obj, "read") else file_obj
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def classify(df: pd.DataFrame) -> str:
    """Guess whether a sheet holds spans or traces."""
    cols = {str(c).strip().lower() for c in df.columns}
    cols |= {ALIASES.get(c, c) for c in cols}
    if "parent_span_id" in cols or "span_id" in cols:
        return "spans"
    if "trace_id" in cols and "id" in cols:
        return "spans"
    if "trace_id" in cols and "id" not in cols:
        return "spans"
    return "traces"


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def normalise(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Canonical column names, real types, parsed JSON payloads."""
    out = df.copy()
    out.columns = [ALIASES.get(str(c).strip().lower(), str(c).strip().lower()) for c in out.columns]
    out = out.loc[:, ~out.columns.duplicated()]

    for col in ("id", "trace_id", "parent_span_id", "thread_id", "name", "type", "project_name"):
        if col not in out.columns:
            out[col] = None
        out[col] = out[col].map(_clean_scalar)

    for col in JSON_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map(parse_maybe_json)

    for col in ("start_time", "end_time"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce", utc=True)
        else:
            out[col] = pd.NaT

    out["duration_ms"] = _durations(out)

    if "total_estimated_cost" in out.columns:
        out["cost"] = pd.to_numeric(out["total_estimated_cost"], errors="coerce").fillna(0.0)
    else:
        out["cost"] = 0.0

    tokens = out["usage"].map(_tokens) if "usage" in out.columns else pd.Series([{}] * len(out))
    out["prompt_tokens"] = [t.get("prompt", 0) for t in tokens]
    out["completion_tokens"] = [t.get("completion", 0) for t in tokens]
    out["total_tokens"] = [t.get("total", 0) for t in tokens]

    out["scores"] = out["feedback_scores"].map(_scores) if "feedback_scores" in out.columns else [{} for _ in range(len(out))]
    out["has_error"] = out["error_info"].map(bool) if "error_info" in out.columns else False

    if kind == "spans":
        out["type"] = out["type"].fillna("general").replace({None: "general"})
    out["kind"] = kind
    return out


def _clean_scalar(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    return None if s in ("", "nan", "None", "null") else s


def _durations(df: pd.DataFrame) -> pd.Series:
    if "duration" in df.columns:
        d = pd.to_numeric(df["duration"], errors="coerce")
        if d.notna().any():
            # Opik exports ms; SDK sometimes gives seconds. Sniff the magnitude.
            if d.dropna().median() < 60:
                d = d * 1000.0
            return d.fillna(0.0)
    delta = (df["end_time"] - df["start_time"]).dt.total_seconds() * 1000.0
    return delta.fillna(0.0)


def parse_maybe_json(v: Any) -> Any:
    """CSV exports stringify JSON. Recover the object, tolerating Python reprs."""
    if v is None or isinstance(v, (dict, list, int, float, bool)):
        return None if (isinstance(v, float) and math.isnan(v)) else v
    s = str(v).strip()
    if not s or s in ("nan", "None", "null", "{}", "[]"):
        return None
    if s[0] in "{[":
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            return ast.literal_eval(s)
        except Exception:
            pass
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            return s
    return s


def _tokens(usage: Any) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {"prompt": 0, "completion": 0, "total": 0}
    flat = {str(k).lower(): v for k, v in usage.items()}

    def pick(*keys: str) -> int:
        for k in keys:
            for fk, fv in flat.items():
                if fk.endswith(k) and isinstance(fv, (int, float)):
                    return int(fv)
        return 0

    p = pick("prompt_tokens", "input_tokens")
    c = pick("completion_tokens", "output_tokens")
    t = pick("total_tokens") or (p + c)
    return {"prompt": p, "completion": c, "total": t}


def _scores(raw: Any) -> dict[str, float]:
    """feedback_scores comes as [{'name': 'hallucination', 'value': 0.0}, ...]."""
    out: dict[str, float] = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and "name" in item:
                try:
                    out[str(item["name"])] = float(item.get("value", 0))
                except (TypeError, ValueError):
                    out[str(item["name"])] = item.get("value")
    elif isinstance(raw, dict):
        for k, v in raw.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                out[str(k)] = v
    return out


# --------------------------------------------------------------------------
# Tree + thread assembly
# --------------------------------------------------------------------------


@dataclass
class SpanNode:
    row: dict
    children: list["SpanNode"] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.row.get("id") or ""

    @property
    def name(self) -> str:
        return self.row.get("name") or self.row.get("type") or "span"

    @property
    def duration_ms(self) -> float:
        return float(self.row.get("duration_ms") or 0.0)


def build_span_tree(spans: pd.DataFrame, trace_id: str) -> list[SpanNode]:
    """Rebuild the parent/child tree Opik draws in its trace sidebar."""
    if spans is None or spans.empty:
        return []
    subset = spans[spans["trace_id"] == trace_id]
    if subset.empty:
        return []
    nodes = {r["id"]: SpanNode(row=r) for r in subset.to_dict("records") if r.get("id")}
    roots: list[SpanNode] = []
    for node in nodes.values():
        parent_id = node.row.get("parent_span_id")
        parent = nodes.get(parent_id) if parent_id else None
        (parent.children if parent else roots).append(node)

    def sort_rec(items: list[SpanNode]) -> None:
        items.sort(key=lambda n: (n.row.get("start_time") is None, n.row.get("start_time")))
        for i in items:
            sort_rec(i.children)

    sort_rec(roots)
    return roots


def flatten(nodes: list[SpanNode], depth: int = 0) -> list[tuple[int, SpanNode]]:
    flat: list[tuple[int, SpanNode]] = []
    for n in nodes:
        flat.append((depth, n))
        flat.extend(flatten(n.children, depth + 1))
    return flat


def build_threads(traces: pd.DataFrame) -> pd.DataFrame:
    """Collapse traces into the thread table Opik shows under 'Threads'."""
    if traces is None or traces.empty or "thread_id" not in traces.columns:
        return pd.DataFrame()
    df = traces[traces["thread_id"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby("thread_id", dropna=True)
    rows = []
    for tid, g in grouped:
        g = g.sort_values("start_time")
        rows.append(
            {
                "thread_id": tid,
                "traces": len(g),
                "first_message": humanise(g.iloc[0].get("input"))[:160],
                "last_message": humanise(g.iloc[-1].get("output"))[:160],
                "start_time": g["start_time"].min(),
                "end_time": g["end_time"].max(),
                "duration_ms": float(g["duration_ms"].sum()),
                "total_tokens": int(g["total_tokens"].sum()),
                "cost": float(g["cost"].sum()),
                "errors": int(g["has_error"].sum()) if "has_error" in g else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("start_time", ascending=False, na_position="last")


# --------------------------------------------------------------------------
# The "human view": JSON -> readable text
# --------------------------------------------------------------------------

_TEXT_KEYS = (
    "content", "text", "output", "answer", "response", "result",
    "question", "query", "prompt", "input", "message", "value",
)


def extract_messages(payload: Any) -> list[dict[str, str]]:
    """Pull a chat transcript out of whatever shape the payload has."""
    msgs: list[dict[str, str]] = []
    if isinstance(payload, dict):
        # OpenAI / OpenRouter completion shape
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            for ch in choices:
                if isinstance(ch, dict):
                    m = ch.get("message") or ch.get("delta") or {}
                    if isinstance(m, dict):
                        msgs.append({"role": str(m.get("role", "assistant")), "content": _text_of(m.get("content", m))})
                    elif "text" in ch:
                        msgs.append({"role": "assistant", "content": str(ch["text"])})
            if msgs:
                return msgs
        for key in ("messages", "chat_history", "history", "conversation"):
            seq = payload.get(key)
            if isinstance(seq, list):
                for m in seq:
                    if isinstance(m, dict):
                        role = str(m.get("role") or m.get("type") or "user")
                        msgs.append({"role": role, "content": _text_of(m.get("content", m))})
                return msgs
    if isinstance(payload, list) and payload and isinstance(payload[0], dict) and "role" in payload[0]:
        for m in payload:
            msgs.append({"role": str(m.get("role", "user")), "content": _text_of(m.get("content", m))})
    return msgs


def _text_of(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        parts = []
        for item in v:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(_text_of(item))
        return "\n".join(p for p in parts if p)
    if isinstance(v, dict):
        for k in _TEXT_KEYS:
            if k in v and isinstance(v[k], (str, list)):
                return _text_of(v[k])
        return json.dumps(v, indent=2, ensure_ascii=False, default=str)
    return "" if v is None else str(v)


def humanise(payload: Any, max_chars: int | None = None) -> str:
    """Best-effort plain-text rendering of an Opik input/output blob."""
    if payload is None:
        return ""
    msgs = extract_messages(payload)
    if msgs:
        text = "\n\n".join(f"{m['role'].upper()}: {m['content']}".strip() for m in msgs)
    elif isinstance(payload, str):
        text = payload
    elif isinstance(payload, dict):
        # Single obvious text field -> show just that; otherwise a key: value list.
        hits = [k for k in _TEXT_KEYS if k in payload]
        if len(payload) == 1 or (hits and len(payload) <= 3):
            text = _text_of(payload)
        else:
            lines = []
            for k, v in payload.items():
                rendered = _text_of(v)
                lines.append(f"**{k}**\n{rendered}" if "\n" in rendered else f"**{k}:** {rendered}")
            text = "\n\n".join(lines)
    elif isinstance(payload, list):
        text = "\n".join(f"- {_text_of(i)}" for i in payload)
    else:
        text = str(payload)

    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip() + " …"
    return text


def fmt_duration(ms: float) -> str:
    ms = float(ms or 0)
    if ms < 1000:
        return f"{ms:.0f} ms"
    if ms < 60_000:
        return f"{ms / 1000:.2f} s"
    return f"{int(ms // 60000)}m {(ms % 60000) / 1000:.1f}s"


def fmt_time(ts: Any) -> str:
    if ts is None or (isinstance(ts, float) and math.isnan(ts)) or pd.isna(ts):
        return "—"
    if isinstance(ts, (datetime, pd.Timestamp)):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return str(ts)


# --------------------------------------------------------------------------
# Markdown export
# --------------------------------------------------------------------------


def trace_to_markdown(trace: dict, tree: list[SpanNode]) -> str:
    buf = io.StringIO()
    buf.write(f"# Trace — {trace.get('name') or trace.get('id')}\n\n")
    buf.write(f"- **Trace ID:** `{trace.get('id')}`\n")
    if trace.get("thread_id"):
        buf.write(f"- **Thread ID:** `{trace.get('thread_id')}`\n")
    buf.write(f"- **Started:** {fmt_time(trace.get('start_time'))}\n")
    buf.write(f"- **Duration:** {fmt_duration(trace.get('duration_ms'))}\n")
    buf.write(f"- **Tokens:** {int(trace.get('total_tokens') or 0)}  ")
    buf.write(f"**Cost:** ${float(trace.get('cost') or 0):.6f}\n\n")
    scores = trace.get("scores") or {}
    if scores:
        buf.write("**Feedback scores:** " + ", ".join(f"{k} = {v}" for k, v in scores.items()) + "\n\n")
    buf.write("## Input\n\n" + (humanise(trace.get("input")) or "_empty_") + "\n\n")
    buf.write("## Output\n\n" + (humanise(trace.get("output")) or "_empty_") + "\n\n")
    if tree:
        buf.write("## Execution\n\n")
        for depth, node in flatten(tree):
            pad = "    " * depth
            buf.write(f"{pad}- **{node.name}** `{node.row.get('type')}` · {fmt_duration(node.duration_ms)}\n")
            inp = humanise(node.row.get("input"), 600)
            outp = humanise(node.row.get("output"), 600)
            if inp:
                buf.write(f"{pad}  - in: {inp.splitlines()[0][:200]}\n")
            if outp:
                buf.write(f"{pad}  - out: {outp.splitlines()[0][:200]}\n")
    return buf.getvalue()


def thread_to_markdown(thread_id: str, traces: pd.DataFrame) -> str:
    buf = io.StringIO()
    buf.write(f"# Thread `{thread_id}`\n\n")
    g = traces[traces["thread_id"] == thread_id].sort_values("start_time")
    buf.write(f"{len(g)} turns · {fmt_duration(g['duration_ms'].sum())} · ")
    buf.write(f"{int(g['total_tokens'].sum())} tokens · ${g['cost'].sum():.6f}\n\n---\n\n")
    for i, r in enumerate(g.to_dict("records"), 1):
        buf.write(f"### Turn {i} · {fmt_time(r.get('start_time'))}\n\n")
        buf.write("**User**\n\n" + (humanise(r.get("input")) or "_empty_") + "\n\n")
        buf.write("**Assistant**\n\n" + (humanise(r.get("output")) or "_empty_") + "\n\n")
    return buf.getvalue()
