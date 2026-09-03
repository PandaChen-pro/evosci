import { useMemo } from "react";

import type { RunEvent } from "../api/client";
import { formatMs, PHASE_LABEL } from "./StepTimeline";

const TASK_LABEL: Record<string, string> = {
  problem_generation: "生成问题",
  research_notes: "研究笔记",
  idea_generation: "生成想法",
  refinement: "想法精修",
  variation: "实体变异",
  review: "评审",
  meta_review: "元评审",
  pairwise_compare: "两两比较",
};


interface Call {
  call_id: string;
  task: string;
  phase: string;
  round: number | null;
  started: string;
  duration_ms: number | null;
  attempts: number;
  ok: boolean | null;
  error: string | null;
  prompt_chars: number | null;
  response_chars: number | null;
}

/** Folds llm.start / llm.attempt / llm.end into one row per call_id. */
export function useCalls(events: RunEvent[]): Call[] {
  return useMemo(() => {
    const byId = new Map<string, Call>();
    for (const event of events) {
      const data = event.data as Record<string, unknown>;
      const id = String(data.call_id ?? "");
      if (!id) continue;
      if (event.type === "llm.start") {
        byId.set(id, {
          call_id: id,
          task: String(data.task ?? "?"),
          phase: String(data.phase ?? "?"),
          round: (data.round as number | null) ?? null,
          started: event.ts,
          duration_ms: null,
          attempts: 1,
          ok: null,
          error: null,
          prompt_chars: (data.prompt_chars as number | null) ?? null,
          response_chars: null,
        });
      } else if (event.type === "llm.attempt") {
        const call = byId.get(id);
        if (call) {
          call.attempts = Math.max(call.attempts, Number(data.attempt ?? call.attempts));
          if (data.ok === false) call.error = (data.error as string | null) ?? call.error;
        }
      } else if (event.type === "llm.end") {
        const call = byId.get(id);
        if (call) {
          call.duration_ms = (data.duration_ms as number | null) ?? null;
          call.ok = data.ok === undefined ? true : Boolean(data.ok);
          call.error = (data.error as string | null) ?? call.error;
          call.response_chars = (data.response_chars as number | null) ?? null;
          if (data.attempts) call.attempts = Number(data.attempts);
        }
      }
    }
    return Array.from(byId.values());
  }, [events]);
}

export function LlmCallTable({ events }: { events: RunEvent[] }) {
  const calls = useCalls(events);
  if (calls.length === 0) {
    return <div className="empty">还没有记录到模型调用。</div>;
  }
  const rows = [...calls].reverse();
  const failed = calls.filter((call) => call.ok === false).length;
  const retried = calls.filter((call) => call.attempts > 1).length;

  return (
    <div>
      <div className="small muted" style={{ marginBottom: 8 }}>
        共 {calls.length} 次调用 · {retried} 次重试 · {failed} 次失败。字符数只是规模的代理指标，
        不是 token 用量 —— 引擎没有保留 API 响应里的 <code>usage</code> 字段。
      </div>
      <div className="scroll">
        <table>
          <thead>
            <tr>
              <th>调用</th>
              <th>任务</th>
              <th>阶段</th>
              <th>耗时</th>
              <th>尝试</th>
              <th>字符</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((call) => (
              <tr key={call.call_id}>
                <td className="mono small">{call.call_id}</td>
                <td className="small">
                  {TASK_LABEL[call.task] ?? call.task}
                  {call.error && <div className="err small">{call.error}</div>}
                </td>
                <td className="small muted">
                  {PHASE_LABEL[call.phase] ?? call.phase}
                  {call.round ? ` · 第 ${call.round} 轮` : ""}
                </td>
                <td className="small mono">
                  {call.duration_ms === null ? (
                    <span className="muted">进行中…</span>
                  ) : (
                    formatMs(call.duration_ms)
                  )}
                </td>
                <td className={`small mono${call.attempts > 1 ? " attempts-bad" : ""}`}>
                  {call.attempts}
                  {call.ok === false ? " ✕" : ""}
                </td>
                <td className="small mono muted">
                  {call.prompt_chars ?? "—"}
                  {" → "}
                  {call.response_chars ?? "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
