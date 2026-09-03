import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { api, type RunDetail, type RunEvent } from "../api/client";
import { streamEvents } from "../api/sse";
import { LlmCallTable } from "../components/LlmCallTable";
import { StepTimeline } from "../components/StepTimeline";
import FeedbackLedger from "./FeedbackLedger";
import Results from "./Results";
import { STATUS_LABEL } from "./RunList";
import { Artifacts, Diagnostics, GraphView } from "./Inspect";

const TABS = ["live", "results", "ledger", "graph", "diagnostics", "artifacts"] as const;
type Tab = (typeof TABS)[number];

const TAB_LABEL: Record<Tab, string> = {
  live: "实时进度",
  results: "结果",
  ledger: "反馈账本",
  graph: "知识图谱",
  diagnostics: "诊断",
  artifacts: "产物文件",
};

const ACTIVE = new Set(["queued", "running"]);

export default function RunDetailPage() {
  const { jobId = "" } = useParams();
  const [tab, setTab] = useState<Tab>("live");
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
  const seenRef = useRef(new Set<number>());

  const reload = useCallback(
    () => api.getRun(jobId).then(setDetail).catch((exc) => setError(String(exc))),
    [jobId],
  );

  useEffect(() => {
    void reload();
  }, [reload]);

  const status = detail?.job.status ?? "unknown";
  const live = ACTIVE.has(status);

  // Backfill history once, then stream. The cursor makes a refresh mid-run resume exactly.
  useEffect(() => {
    let cancelled = false;
    let handle: { close(): void } | null = null;

    const append = (incoming: RunEvent[]) => {
      const fresh = incoming.filter((event) => !seenRef.current.has(event.seq));
      if (fresh.length === 0) return;
      for (const event of fresh) seenRef.current.add(event.seq);
      setEvents((prev) => [...prev, ...fresh].sort((a, b) => a.seq - b.seq));
    };

    api
      .eventsPage(jobId, 0)
      .then((page) => {
        if (cancelled) return;
        append(page.events);
        if (!page.active) return;
        handle = streamEvents(jobId, {
          since: page.last_seq,
          onEvent: (event) => {
            append([event]);
            if (event.type.startsWith("job.")) void reload();
          },
          onOpen: () => setStreamError(null),
          onError: (exc) => setStreamError(exc instanceof Error ? exc.message : String(exc)),
        });
      })
      .catch((exc) => !cancelled && setError(String(exc)));

    return () => {
      cancelled = true;
      handle?.close();
    };
  }, [jobId, reload]);

  async function control(action: "cancel" | "resume") {
    setBusy(true);
    setError(null);
    try {
      await (action === "cancel" ? api.cancel(jobId) : api.resume(jobId));
      seenRef.current.clear();
      setEvents([]);
      await reload();
      // A resumed run appends to the same log; remount the stream by reloading the page data.
      if (action === "resume") window.location.reload();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  }

  if (error && !detail) return <div className="err">{error}</div>;
  if (!detail) return <div className="muted">正在加载任务…</div>;

  const job = detail.job;
  const rounds = detail.state?.rounds ?? [];

  return (
    <div>
      <div className="small muted" style={{ marginBottom: 6 }}>
        <Link to="/runs">← 任务列表</Link>
      </div>
      <h2 style={{ marginBottom: 6 }}>
        {job.topic ?? job.job_id}{" "}
        <span className={`pill ${job.status}`}>{STATUS_LABEL[job.status] ?? job.status}</span>
      </h2>
      <div className="small muted" style={{ marginBottom: 16 }}>
        <span className="mono">{job.job_id}</span> · {job.disciplines.join("、") || "—"} · 第{" "}
        {job.rounds_done}
        {job.rounds_target ? `/${job.rounds_target}` : ""} 轮 · {job.model ?? "—"}
        {job.provider ? ` (${job.provider})` : ""}
      </div>

      <div className="row" style={{ marginBottom: 16 }}>
        {TABS.map((name) => (
          <button key={name} className={tab === name ? "primary" : ""} onClick={() => setTab(name)}>
            {TAB_LABEL[name]}
          </button>
        ))}
        <span style={{ flex: 1 }} />
        {live && (
          <button className="danger" disabled={busy} onClick={() => control("cancel")}>
            取消运行
          </button>
        )}
        {/* Only runs this UI started are resumable: a run found under a scan root is a
            finished artifact, and resuming it would rewrite its state.json in place. */}
        {!live && job.managed && detail.artifacts.some((file) => file.name === "config.json") && (
          <button disabled={busy} onClick={() => control("resume")}>
            继续运行
          </button>
        )}
      </div>

      {error && <div className="banner">{error}</div>}
      {streamError && (
        <div className="banner">
          实时流中断（{streamError}），正在按上次游标重连。
        </div>
      )}
      {!job.has_events && (
        <div className="banner info">
          该任务没有 <code>events.jsonl</code> —— 它不是在本界面启动的，因此没有逐次调用遥测和
          墙钟耗时拆解。它的产物文件是完整的。
        </div>
      )}

      {tab === "live" && (
        <div className="split">
          <div className="panel">
            <h3 style={{ marginTop: 0 }}>执行步骤</h3>
            <StepTimeline events={events} live={live} />
          </div>
          <div className="panel">
            <h3 style={{ marginTop: 0 }}>模型调用</h3>
            <LlmCallTable events={events} />
          </div>
        </div>
      )}
      {tab === "results" && <Results jobId={jobId} rounds={rounds} />}
      {tab === "ledger" && <FeedbackLedger jobId={jobId} />}
      {tab === "graph" && <GraphView jobId={jobId} />}
      {tab === "diagnostics" && <Diagnostics jobId={jobId} />}
      {tab === "artifacts" && <Artifacts jobId={jobId} files={detail.artifacts} />}
    </div>
  );
}
