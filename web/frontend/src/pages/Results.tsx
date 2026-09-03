import { useEffect, useState } from "react";

import { api, type IdeaSummary, type RoundSummary } from "../api/client";
import { ScoreCaveat } from "../components/ScoreCaveat";

const FIELD_LABEL: Record<string, string> = {
  hypothesis: "假设",
  rationale: "理由",
  method: "方法",
  experiment: "实验设计",
  expected_outcome: "预期结果",
  risks: "风险",
};

const SCORE_LABEL: Record<string, string> = {
  novelty: "新颖性",
  feasibility: "可行性",
  validity: "有效性",
  excitement: "吸引力",
  overall: "总体",
  confidence: "置信度",
};

const LIST_LABEL: Record<string, string> = {
  strengths: "优点",
  weaknesses: "不足",
  suggestions: "改进建议",
};


export default function Results({
  jobId,
  rounds,
}: {
  jobId: string;
  rounds: RoundSummary[];
}) {
  const [open, setOpen] = useState<string | null>(null);

  if (rounds.length === 0) {
    return <div className="empty">还没有完成任何一轮。</div>;
  }

  const reviewers = Math.max(1, ...rounds.flatMap((r) => r.ideas.map((i) => i.review_count)));

  return (
    <div>
      <ScoreCaveat reviewers={reviewers} />
      {rounds.map((round) => (
        <div className="panel" key={round.round_index}>
          <h3 style={{ marginTop: 0 }}>
            第 {round.round_index} 轮
            <span className="muted small">
              {" "}
              · {round.ideas.length} 个想法 · {round.problems.length} 个问题
            </span>
          </h3>

          {round.ideas.length === 0 ? (
            <div className="empty">这一轮没有记录到经过评估的想法。</div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>想法</th>
                  <th>fitness</th>
                  <th>元评审分数</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {[...round.ideas]
                  .sort((a, b) => (b.fitness ?? 0) - (a.fitness ?? 0))
                  .map((idea) => (
                    <tr key={idea.id}>
                      <td>
                        {idea.title}
                        <div className="muted small">{idea.hypothesis}</div>
                      </td>
                      <td className="mono small">{idea.fitness?.toFixed(3) ?? "—"}</td>
                      <td className="small muted mono">{scoreLine(idea)}</td>
                      <td>
                        <button
                          className="small"
                          onClick={() => setOpen(open === idea.id ? null : idea.id)}
                        >
                          {open === idea.id ? "收起" : "详情"}
                        </button>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          )}

          {open && round.ideas.some((idea) => idea.id === open) && (
            <IdeaDetail jobId={jobId} ideaId={open} />
          )}

          <EvolutionSummary summary={round.evolution_summary} />
        </div>
      ))}
    </div>
  );
}

function scoreLine(idea: IdeaSummary): string {
  const order = ["novelty", "feasibility", "validity", "excitement", "overall"];
  return order
    .map((key) => `${SCORE_LABEL[key][0]}${idea.meta_scores[key] ?? "—"}`)
    .join(" ");
}

function IdeaDetail({ jobId, ideaId }: { jobId: string; ideaId: string }) {
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setData(null);
    api.getIdea(jobId, ideaId).then(setData).catch((exc) => setError(String(exc)));
  }, [jobId, ideaId]);

  if (error) return <div className="err">{error}</div>;
  if (!data) return <div className="muted small">正在加载想法详情…</div>;

  const idea = (data.idea ?? {}) as Record<string, unknown>;
  const reviews = (data.reviews ?? []) as Record<string, unknown>[];
  const meta = (data.meta_review ?? {}) as Record<string, unknown>;
  const prose = ["hypothesis", "rationale", "method", "experiment", "expected_outcome", "risks"];

  return (
    <div className="panel" style={{ background: "var(--panel-2)" }}>
      <h3 style={{ marginTop: 0 }}>{String(idea.title ?? ideaId)}</h3>
      {prose.map((key) =>
        idea[key] ? (
          <div key={key} style={{ marginBottom: 10 }}>
            <div className="small muted">{FIELD_LABEL[key] ?? key.replace(/_/g, " ")}</div>
            <div style={{ whiteSpace: "pre-wrap" }}>{renderValue(idea[key])}</div>
          </div>
        ) : null,
      )}
      <div className="small muted">
        涉及实体：{(idea.entity_ids as string[] | undefined)?.join("、") || "—"} · 作者：{" "}
        {(idea.authors as string[] | undefined)?.join("、") || "—"}
      </div>

      <h3>元评审</h3>
      <ReviewBlock review={meta} />
      <h3>评审意见（{reviews.length} 份）</h3>
      {reviews.map((review, index) => (
        <ReviewBlock key={index} review={review} />
      ))}
    </div>
  );
}

function ReviewBlock({ review }: { review: Record<string, unknown> }) {
  const scores = ["novelty", "feasibility", "validity", "excitement", "overall", "confidence"];
  const lists = ["strengths", "weaknesses", "suggestions"];
  return (
    <div className="chip">
      <div className="mono small">
        {String(review.reviewer_id ?? "meta")} ·{" "}
        {scores.map((key) => `${SCORE_LABEL[key]}=${review[key] ?? "—"}`).join("  ")}
      </div>
      {lists.map((key) => {
        const items = (review[key] as string[] | undefined) ?? [];
        if (items.length === 0) return null;
        return (
          <div key={key} style={{ marginTop: 6 }}>
            <div className="small muted">{LIST_LABEL[key] ?? key}</div>
            <ul style={{ margin: "2px 0 0 18px", padding: 0 }}>
              {items.map((item, index) => (
                <li key={index} className="small">
                  {item}
                </li>
              ))}
            </ul>
          </div>
        );
      })}
    </div>
  );
}

function renderValue(value: unknown) {
  if (Array.isArray(value)) return value.join("\n");
  return String(value);
}

function EvolutionSummary({ summary }: { summary: RoundSummary["evolution_summary"] }) {
  const crossovers = (summary.crossovers ?? []) as {
    entity_id: string;
    from: string | null;
    to: string | null;
  }[];
  const pruned = (summary.pruned ?? []) as string[];
  const variations = (summary.variations ?? []) as unknown[];
  const selected = (summary.selected ?? []) as unknown[];

  return (
    <div style={{ marginTop: 14 }}>
      <div className="small muted">聚类演化</div>
      <div className="small">
        第 {String(summary.generation ?? "—")} 代 · 选中 {selected.length} 个 · 交叉{" "}
        {crossovers.length} 次 · 变异 {variations.length} 次
      </div>
      {crossovers.length > 0 && (
        <ul style={{ margin: "6px 0 0 18px", padding: 0 }}>
          {crossovers.map((item, index) => (
            <li key={index} className="small mono">
              {item.entity_id}
              {item.from && item.to ? ` : ${item.from} → ${item.to}` : ""}
            </li>
          ))}
        </ul>
      )}
      {!summary.crossovers_detailed && crossovers.length > 0 && (
        <div className="small muted" style={{ marginTop: 4 }}>
          该任务把交叉记录成了裸实体 id —— 它早于保存源聚类和目标聚类的 schema，因此两端无法恢复。
        </div>
      )}
      <div className="small muted" style={{ marginTop: 4 }}>
        {pruned.length === 0
          ? "本轮没有实体被剪除。"
          : `被剪除：${pruned.join("、")}`}
      </div>
    </div>
  );
}
