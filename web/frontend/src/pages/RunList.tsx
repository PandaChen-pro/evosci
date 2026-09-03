import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { api, type RunSummary } from "../api/client";

const ACTIVE = new Set(["queued", "running"]);

export const STATUS_LABEL: Record<string, string> = {
  queued: "排队中",
  running: "运行中",
  finished: "已完成",
  failed: "失败",
  cancelled: "已取消",
  interrupted: "已中断",
  archived: "历史归档",
  unknown: "未知",
};

export default function RunList() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () =>
      api
        .listRuns()
        .then((data) => alive && setRuns(data.runs))
        .catch((exc) => alive && setError(String(exc)));
    load();
    // Only the list polls; a run's own page streams instead.
    const timer = setInterval(load, 4000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  if (error) return <div className="err">{error}</div>;
  if (!runs) return <div className="muted">正在加载任务列表…</div>;

  const roots = new Set(runs.map((run) => run.scan_root));

  return (
    <div>
      <h2>任务列表</h2>
      {runs.length === 0 ? (
        <div className="empty">
          在已配置的扫描目录里没有找到任务。从 <Link to="/new">新建任务</Link> 开始。
        </div>
      ) : (
        <>
          <table>
            <thead>
              <tr>
                <th>状态</th>
                <th>主题</th>
                <th>轮次</th>
                <th>模型</th>
                <th>创建时间</th>
                {roots.size > 1 && <th>所在目录</th>}
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.run_dir}>
                  <td>
                    <span className={`pill ${run.status}`}>
                      {STATUS_LABEL[run.status] ?? run.status}
                    </span>
                  </td>
                  <td>
                    <Link to={`/runs/${run.job_id}`}>{run.topic ?? run.job_id}</Link>
                    {run.label && <span className="muted small"> · {run.label}</span>}
                    <div className="muted small mono">{run.job_id}</div>
                  </td>
                  <td className="mono">
                    {run.rounds_done}
                    {run.rounds_target ? ` / ${run.rounds_target}` : ""}
                  </td>
                  <td className="small">{run.model ?? "—"}</td>
                  <td className="small muted">{run.created_at?.replace("T", " ") ?? "—"}</td>
                  {roots.size > 1 && <td className="small muted">{run.scan_root}</td>}
                </tr>
              ))}
            </tbody>
          </table>
          <div className="small muted" style={{ marginTop: 12 }}>
            {runs.filter((run) => ACTIVE.has(run.status)).length} 个进行中 ·{" "}
            {runs.filter((run) => !run.has_events).length} 个无流式遥测（不是在本界面启动的）。
          </div>
        </>
      )}
    </div>
  );
}
