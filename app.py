"""
Opik Trace Rerouter — a human view for exported traces, spans and threads.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from opik_core import (
    build_span_tree,
    build_threads,
    classify,
    flatten,
    fmt_duration,
    fmt_time,
    humanise,
    normalise,
    read_any,
    thread_to_markdown,
    trace_to_markdown,
)

st.set_page_config(page_title="Opik Trace Rerouter", page_icon="🧵", layout="wide")

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1400px;}
.pill {display:inline-block; padding:2px 9px; border-radius:999px; font-size:11px;
       font-weight:600; letter-spacing:.02em; margin-right:6px; border:1px solid transparent;}
.pill-llm   {background:#eef2ff; color:#4338ca; border-color:#c7d2fe;}
.pill-tool  {background:#ecfdf5; color:#047857; border-color:#a7f3d0;}
.pill-general{background:#f1f5f9; color:#475569; border-color:#cbd5e1;}
.pill-guardrail{background:#fff7ed; color:#c2410c; border-color:#fed7aa;}
.pill-err   {background:#fef2f2; color:#b91c1c; border-color:#fecaca;}
.pill-score {background:#f5f3ff; color:#6d28d9; border-color:#ddd6fe;}
.bubble {border-radius:12px; padding:12px 14px; margin:6px 0 14px 0; font-size:14px;
         line-height:1.55; white-space:pre-wrap; border:1px solid #e2e8f0;}
.bubble-user {background:#f8fafc;}
.bubble-bot  {background:#f5f8ff; border-color:#dbe4ff;}
.rolelabel {font-size:11px; font-weight:700; letter-spacing:.06em; color:#64748b;
            text-transform:uppercase; margin-bottom:2px;}
.wf-row {display:flex; align-items:center; gap:10px; margin:3px 0; font-size:12.5px;}
.wf-label {width:300px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.wf-track {flex:1; background:#f1f5f9; border-radius:4px; height:14px; position:relative;}
.wf-bar {position:absolute; height:14px; border-radius:4px; background:#6366f1;}
.wf-time {width:80px; text-align:right; color:#64748b; font-variant-numeric:tabular-nums;}
.mono {font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; color:#64748b;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- loading ---

@st.cache_data(show_spinner=False)
def load(files: list[tuple[str, bytes]]) -> dict[str, pd.DataFrame]:
    frames: dict[str, list[pd.DataFrame]] = {"traces": [], "spans": []}
    for name, blob in files:
        import io as _io

        df = read_any(_io.BytesIO(blob), name)
        kind = classify(df)
        frames[kind].append(normalise(df, kind))
    out = {}
    for kind, parts in frames.items():
        out[kind] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out


def pill(text: str, cls: str = "general") -> str:
    return f'<span class="pill pill-{cls}">{text}</span>'


def score_pills(scores: dict) -> str:
    return "".join(pill(f"{k}: {v}", "score") for k, v in (scores or {}).items())


# ------------------------------------------------------------------ views ---

def render_payload(label: str, payload, role: str = "user") -> None:
    text = humanise(payload)
    if not text:
        return
    st.markdown(f'<div class="rolelabel">{label}</div>', unsafe_allow_html=True)
    cls = "bubble-bot" if role == "assistant" else "bubble-user"
    safe = text.replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(f'<div class="bubble {cls}">{safe}</div>', unsafe_allow_html=True)
    if isinstance(payload, (dict, list)):
        with st.expander("Raw JSON", expanded=False):
            st.json(payload)


def render_waterfall(tree) -> None:
    flat = flatten(tree)
    if not flat:
        return
    starts = [n.row.get("start_time") for _, n in flat if pd.notna(n.row.get("start_time"))]
    t0 = min(starts) if starts else None
    total = max((n.duration_ms for _, n in flat), default=1) or 1
    if t0 is not None:
        ends = [
            (n.row.get("start_time") - t0).total_seconds() * 1000 + n.duration_ms
            for _, n in flat
            if pd.notna(n.row.get("start_time"))
        ]
        total = max(ends) or 1
    rows = []
    for depth, n in flat:
        st_ms = 0.0
        if t0 is not None and pd.notna(n.row.get("start_time")):
            st_ms = (n.row["start_time"] - t0).total_seconds() * 1000
        left = min(100.0, max(0.0, st_ms / total * 100))
        width = max(1.0, min(100.0 - left, n.duration_ms / total * 100))
        indent = "&nbsp;" * (depth * 4)
        colour = {"llm": "#6366f1", "tool": "#10b981", "guardrail": "#f97316"}.get(
            str(n.row.get("type")), "#94a3b8"
        )
        rows.append(
            f'<div class="wf-row"><div class="wf-label">{indent}{n.name}</div>'
            f'<div class="wf-track"><div class="wf-bar" style="left:{left}%;width:{width}%;'
            f'background:{colour};"></div></div>'
            f'<div class="wf-time">{fmt_duration(n.duration_ms)}</div></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_span_tree(tree) -> None:
    for depth, node in flatten(tree):
        r = node.row
        stype = str(r.get("type") or "general")
        header = "— " * depth + f"{node.name}  ·  {fmt_duration(node.duration_ms)}"
        with st.expander(header, expanded=(depth == 0 and len(tree) == 1)):
            meta = pill(stype, stype if stype in ("llm", "tool", "guardrail") else "general")
            if r.get("model"):
                meta += pill(str(r["model"]), "general")
            if r.get("provider"):
                meta += pill(str(r["provider"]), "general")
            if r.get("total_tokens"):
                meta += pill(f"{int(r['total_tokens'])} tok", "general")
            if r.get("has_error"):
                meta += pill("error", "err")
            meta += score_pills(r.get("scores"))
            st.markdown(meta, unsafe_allow_html=True)
            st.markdown(
                f'<div class="mono">span {r.get("id")} · {fmt_time(r.get("start_time"))}</div>',
                unsafe_allow_html=True,
            )
            render_payload("Input", r.get("input"), "user")
            render_payload("Output", r.get("output"), "assistant")
            if r.get("error_info"):
                st.error(humanise(r["error_info"]))
            if r.get("metadata"):
                with st.expander("Metadata"):
                    st.json(r["metadata"])


def render_trace(trace: dict, spans: pd.DataFrame) -> None:
    st.subheader(trace.get("name") or "trace")
    st.markdown(f'<div class="mono">{trace.get("id")}</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", fmt_duration(trace.get("duration_ms")))
    c2.metric("Tokens", f"{int(trace.get('total_tokens') or 0):,}")
    c3.metric("Cost", f"${float(trace.get('cost') or 0):.6f}")
    c4.metric("Started", fmt_time(trace.get("start_time")))

    chips = score_pills(trace.get("scores"))
    if trace.get("tags"):
        tags = trace["tags"] if isinstance(trace["tags"], list) else [trace["tags"]]
        chips += "".join(pill(str(t)) for t in tags)
    if chips:
        st.markdown(chips, unsafe_allow_html=True)

    tree = build_span_tree(spans, trace.get("id"))

    tab_conv, tab_exec, tab_raw = st.tabs(["Conversation", f"Execution ({len(flatten(tree))} spans)", "Raw"])
    with tab_conv:
        render_payload("Input", trace.get("input"), "user")
        render_payload("Output", trace.get("output"), "assistant")
        if trace.get("error_info"):
            st.error(humanise(trace["error_info"]))
    with tab_exec:
        if tree:
            render_waterfall(tree)
            st.divider()
            render_span_tree(tree)
        else:
            st.info("No spans matched this trace id. Upload the spans export to see the execution tree.")
    with tab_raw:
        st.json({k: v for k, v in trace.items() if not isinstance(v, pd.Timestamp)}, expanded=False)

    st.download_button(
        "⬇ Download this trace as Markdown",
        trace_to_markdown(trace, tree),
        file_name=f"trace_{str(trace.get('id'))[:8]}.md",
        mime="text/markdown",
    )


# ------------------------------------------------------------------- app ----

st.title("🧵 Opik Trace Rerouter")
st.caption("Drop your Opik CSV / Excel / JSON exports and read them the way the Opik app shows them.")

with st.sidebar:
    st.header("Data")
    uploads = st.file_uploader(
        "Traces and/or spans export",
        type=["csv", "xlsx", "xls", "json", "jsonl"],
        accept_multiple_files=True,
        help="Opik UI → select rows → Actions → Export CSV. Upload the traces sheet and the spans sheet together.",
    )
    st.markdown(
        "**How Opik nests things**\n\n"
        "`thread_id` → conversation\n\n"
        "`trace.id` → one request\n\n"
        "`span.parent_span_id` → execution tree"
    )

if not uploads:
    st.info("Upload at least one export to begin. Traces and spans are detected automatically.")
    st.stop()

data = load([(f.name, f.getvalue()) for f in uploads])
traces, spans = data["traces"], data["spans"]

if traces.empty and not spans.empty:
    # Only spans uploaded: synthesise trace rows from root spans.
    roots = spans[spans["parent_span_id"].isna()].copy()
    roots["id"] = roots["trace_id"]
    traces = roots

if traces.empty:
    st.error("Could not find any trace rows in those files.")
    st.stop()

threads = build_threads(traces)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Threads", len(threads))
k2.metric("Traces", len(traces))
k3.metric("Spans", len(spans))
k4.metric("Tokens", f"{int(traces['total_tokens'].sum()):,}")
k5.metric("Cost", f"${traces['cost'].sum():.4f}")
st.divider()

tab_threads, tab_traces, tab_analytics = st.tabs(["Threads", "Traces", "Analytics"])

# --- Threads -----------------------------------------------------------------
with tab_threads:
    if threads.empty:
        st.info("No `thread_id` on these traces — Opik only groups conversations when thread_id is propagated.")
    else:
        view = threads.copy()
        view["duration"] = view["duration_ms"].map(fmt_duration)
        st.dataframe(
            view[["thread_id", "traces", "first_message", "last_message", "duration", "total_tokens", "cost"]],
            use_container_width=True,
            hide_index=True,
        )
        tid = st.selectbox("Open thread", threads["thread_id"].tolist())
        conv = traces[traces["thread_id"] == tid].sort_values("start_time")
        st.markdown(f"**{len(conv)} turns** · {fmt_duration(conv['duration_ms'].sum())} · "
                    f"{int(conv['total_tokens'].sum()):,} tokens")
        for i, r in enumerate(conv.to_dict("records"), 1):
            st.markdown(f"###### Turn {i} · {fmt_time(r.get('start_time'))} · {fmt_duration(r.get('duration_ms'))}")
            render_payload("User", r.get("input"), "user")
            render_payload("Assistant", r.get("output"), "assistant")
            with st.expander("Execution for this turn"):
                tree = build_span_tree(spans, r.get("id"))
                if tree:
                    render_waterfall(tree)
                    render_span_tree(tree)
                else:
                    st.caption("No spans for this trace.")
        st.download_button(
            "⬇ Download thread transcript",
            thread_to_markdown(tid, traces),
            file_name=f"thread_{str(tid)[:12]}.md",
            mime="text/markdown",
        )

# --- Traces ------------------------------------------------------------------
with tab_traces:
    left, right = st.columns([1, 2], gap="large")
    with left:
        q = st.text_input("Search name / input / output", "")
        only_err = st.checkbox("Errors only", value=False)
        pool = traces.copy()
        if only_err and "has_error" in pool.columns:
            pool = pool[pool["has_error"] == True]  # noqa: E712
        if q:
            ql = q.lower()
            mask = pool.apply(
                lambda r: ql in str(r.get("name", "")).lower()
                or ql in humanise(r.get("input")).lower()
                or ql in humanise(r.get("output")).lower(),
                axis=1,
            )
            pool = pool[mask]
        pool = pool.sort_values("start_time", ascending=False, na_position="last")
        st.caption(f"{len(pool)} traces")
        labels = [
            f"{(r.get('name') or 'trace')} · {fmt_duration(r.get('duration_ms'))} · {fmt_time(r.get('start_time'))}"
            for r in pool.to_dict("records")
        ]
        if not labels:
            st.warning("No matches.")
            chosen = None
        else:
            chosen = st.radio("Select", range(len(labels)), format_func=lambda i: labels[i], label_visibility="collapsed")
    with right:
        if chosen is not None:
            render_trace(pool.to_dict("records")[chosen], spans)

# --- Analytics ---------------------------------------------------------------
with tab_analytics:
    a, b = st.columns(2)
    with a:
        st.markdown("**Latency by trace name (mean ms)**")
        st.bar_chart(traces.groupby(traces["name"].fillna("unnamed"))["duration_ms"].mean())
    with b:
        if not spans.empty:
            st.markdown("**Time spent by span type (ms)**")
            st.bar_chart(spans.groupby(spans["type"].fillna("general"))["duration_ms"].sum())

    score_rows = []
    for r in traces.to_dict("records"):
        for k, v in (r.get("scores") or {}).items():
            if isinstance(v, (int, float)):
                score_rows.append({"metric": k, "value": v})
    if score_rows:
        st.markdown("**Feedback scores (mean)**")
        st.bar_chart(pd.DataFrame(score_rows).groupby("metric")["value"].mean())

    st.markdown("**Slowest traces**")
    slow = traces.sort_values("duration_ms", ascending=False).head(15).copy()
    slow["duration"] = slow["duration_ms"].map(fmt_duration)
    st.dataframe(
        slow[["name", "duration", "total_tokens", "cost", "thread_id"]],
        use_container_width=True,
        hide_index=True,
    )
