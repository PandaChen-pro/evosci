import { useEffect, useState } from "react";

import { api, type Ledger } from "../api/client";

/** The truncation boundary in _prior_feedback is recorded in no artifact: a suggestion
 *  that fell off the end looks identical to one that was never made. */
export default function FeedbackLedger({ jobId }: { jobId: string }) {
  const [ledger, setLedger] = useState<Ledger | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getLedger(jobId).then(setLedger).catch((exc) => setError(String(exc)));
  }, [jobId]);

  if (error) return <div className="err">{error}</div>;
  if (!ledger) return <div className="muted">正在重建反馈账本…</div>;
  if (!ledger.available) {
    return (
      <div className="empty">
        账本至少需要两轮已完成的运行 —— 反馈只在跨轮时传递。该任务的轮数不足。
      </div>
    );
  }

  return (
    <div>
      <div className="banner info">{ledger.note}</div>
      {ledger.entries.map((entry) => (
        <div className="panel" key={entry.into_round}>
          <h3 style={{ marginTop: 0 }}>
            第 {entry.from_round} 轮 → 第 {entry.into_round} 轮
          </h3>
          <div className="small muted" style={{ marginBottom: 12 }}>
            按 fitness 取前 {entry.considered_ideas} 个想法（该轮共{" "}
            {entry.total_ideas_in_source_round} 个），共贡献 {entry.candidate_count} 条去重后的建议；
            prompt 最多只收 {entry.limit} 条。
          </div>

          {entry.carried.map((item) => (
            <div className="chip" key={`c-${item.text}`}>
              {item.text}
              <div className="meta">
                来自第 {item.source_idea_rank} 名 {item.source_idea_title ?? item.source_idea_id}
                {item.source_idea_fitness !== null &&
                  ` · fitness ${item.source_idea_fitness.toFixed(3)}`}{" "}
                · 原文进入了第 {entry.into_round} 轮的 prompt
              </div>
            </div>
          ))}

          {entry.dropped.length > 0 && (
            <>
              <div className="divider">
                被 [:{entry.limit}] 截断 —— {entry.dropped.length} 条从未到达模型
              </div>
              {entry.dropped.map((item) => (
                <div className="chip dropped" key={`d-${item.text}`}>
                  {item.text}
                  <div className="meta">
                    来自第 {item.source_idea_rank} 名{" "}
                    {item.source_idea_title ?? item.source_idea_id} · 是被条数上限丢弃的，
                    不是被模型舍弃的
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      ))}
    </div>
  );
}
