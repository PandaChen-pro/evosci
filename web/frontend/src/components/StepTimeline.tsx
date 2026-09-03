import type { RunEvent } from "../api/client";

export const PHASE_LABEL: Record<string, string> = {
  init: "初始化",
  problems: "问题空间",
  team: "团队组建",
  review: "评审",
  evolve: "演化",
  tournament: "锦标赛排序",
};

// The engine emits English progress strings; translating them here keeps src/evosci
// untouched. Anything unrecognised is shown verbatim rather than dropped.
const STEP_PATTERNS: [RegExp, (match: RegExpMatchArray) => string][] = [
  [/^Initializing discipline-entity knowledge graph$/, () => "正在初始化学科—实体知识图谱"],
  [/^Round (\d+): constructing problem space$/, (m) => `第 ${m[1]} 轮：构建问题空间`],
  [/^Round (\d+): running research team$/, (m) => `第 ${m[1]} 轮：研究团队生成想法`],
  [/^Round (\d+): reviewing (\d+) ideas$/, (m) => `第 ${m[1]} 轮：评审 ${m[2]} 个想法`],
  [/^Round (\d+): evolving entity population$/, (m) => `第 ${m[1]} 轮：演化实体种群`],
  [
    /^Ranking (\d+) ideas in a (\d+)-round tournament$/,
    (m) => `在 ${m[2]} 轮锦标赛中为 ${m[1]} 个想法排序`,
  ],
];

function stepText(message: string): string {
  for (const [pattern, render] of STEP_PATTERNS) {
    const match = message.match(pattern);
    if (match) return render(match);
  }
  return message;
}

export function StepTimeline({ events, live }: { events: RunEvent[]; live: boolean }) {
  const steps = events.filter((event) => event.type === "step");
  if (steps.length === 0) {
    return (
      <div className="empty">
        还没有步骤事件。不是在本界面启动的任务完全没有事件日志。
      </div>
    );
  }

  const lastSeq = steps[steps.length - 1].seq;

  // Grouped in stream order: the tournament carries no round number, so keying a map
  // would file it under "Setup" and float it above the rounds it actually ranks.
  const groups: { title: string; items: { step: RunEvent; elapsed: number | null }[] }[] = [];
  let seenRound = false;
  steps.forEach((step, index) => {
    const round = step.data.round as number | null;
    if (round) seenRound = true;
    const title = round ? `第 ${round} 轮` : seenRound ? "报告" : "准备";
    // Duration spans to the next step anywhere in the stream, not the next in this
    // group — otherwise every group's last step would read as instant.
    const next = steps[index + 1];
    const elapsed = next
      ? Date.parse(next.ts) - Date.parse(step.ts)
      : live && step.seq === lastSeq
        ? null
        : 0;
    const last = groups[groups.length - 1];
    if (last && last.title === title) last.items.push({ step, elapsed });
    else groups.push({ title, items: [{ step, elapsed }] });
  });

  return (
    <div className="scroll">
      {groups.map(({ title, items }, groupIndex) => (
        <div key={`${title}-${groupIndex}`} style={{ marginBottom: 14 }}>
          <div className="small muted" style={{ marginBottom: 6 }}>
            {title}
          </div>
          {items.map(({ step, elapsed }) => (
            <div key={step.seq} className={`step${live && step.seq === lastSeq ? " active" : ""}`}>
              <div>{stepText(String(step.data.message))}</div>
              <div className="small muted">
                {PHASE_LABEL[String(step.data.phase)] ?? String(step.data.phase)}
                {elapsed === null
                  ? " · 进行中"
                  : elapsed > 0
                    ? ` · ${formatMs(elapsed)}`
                    : ""}
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

export function formatMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} 毫秒`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)} 秒`;
  const minutes = Math.floor(ms / 60000);
  return `${minutes} 分 ${Math.round((ms % 60000) / 1000)} 秒`;
}
