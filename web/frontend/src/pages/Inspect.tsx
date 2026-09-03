import { useEffect, useState } from "react";

import { api } from "../api/client";

export function GraphView({ jobId }: { jobId: string }) {
  const [graph, setGraph] = useState<Awaited<ReturnType<typeof api.getGraph>> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getGraph(jobId).then(setGraph).catch((exc) => setError(String(exc)));
  }, [jobId]);

  if (error) return <div className="err">{error}</div>;
  if (!graph) return <div className="muted">正在加载知识图谱…</div>;

  const edgeCount = Object.values(graph.edges).reduce((sum, list) => sum + list.length, 0);

  return (
    <div>
      <div className="panel">
        <h3 style={{ marginTop: 0 }}>实体聚类</h3>
        {graph.has_clusters ? (
          graph.clusters.length === 0 ? (
            <div className="empty">图谱里有 clusters 字段，但它是空的。</div>
          ) : (
            <div className="scroll">
              {graph.clusters.map((cluster, index) => {
                const item = cluster as {
                  id?: string;
                  name?: string;
                  discipline?: string;
                  entity_ids?: string[];
                  stale_rounds?: number;
                };
                return (
                  <div className="chip" key={item.id ?? index}>
                    <strong>{item.name ?? item.id}</strong>
                    <div className="meta">
                      {item.discipline ?? "—"} · {item.entity_ids?.length ?? 0} 个实体
                      {item.stale_rounds ? ` · 已停滞 ${item.stale_rounds} 轮` : ""}
                    </div>
                    <div className="small mono muted" style={{ marginTop: 4 }}>
                      {item.entity_ids?.join("、")}
                    </div>
                  </div>
                );
              })}
            </div>
          )
        ) : (
          <div className="empty">
            该任务的 <code>graph.json</code> 没有 <code>clusters</code> 字段 —— 它早于聚类 schema。
            下方的实体与边视图是完整的，但无法为它重建聚类级的演化过程。
          </div>
        )}
      </div>

      <div className="panel">
        <h3 style={{ marginTop: 0 }}>实体</h3>
        <div className="small muted" style={{ marginBottom: 10 }}>
          {graph.entities.length} 个实体 · {edgeCount} 条边
        </div>
        {graph.entities.length === 0 ? (
          <div className="empty">该任务目录里没有 graph.json。</div>
        ) : (
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>实体</th>
                  <th>学科</th>
                  <th>邻接数</th>
                </tr>
              </thead>
              <tbody>
                {graph.entities.map((raw, index) => {
                  const entity = raw as { id?: string; name?: string; discipline?: string };
                  const id = entity.id ?? String(index);
                  return (
                    <tr key={id}>
                      <td className="small">
                        {entity.name ?? id}
                        <div className="muted small mono">{id}</div>
                      </td>
                      <td className="small muted">{entity.discipline ?? "—"}</td>
                      <td className="small mono muted">{(graph.edges[id] ?? []).length}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export function Diagnostics({ jobId }: { jobId: string }) {
  const [result, setResult] = useState<{ available: boolean; data: unknown } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getDiagnostics(jobId).then(setResult).catch((exc) => setError(String(exc)));
  }, [jobId]);

  if (error) return <div className="err">{error}</div>;
  if (!result) return <div className="muted">正在加载诊断数据…</div>;
  if (!result.available) {
    return (
      <div className="empty">
        该任务目录里没有 <code>diagnostics.json</code>。重跑或继续运行该任务会生成它。
      </div>
    );
  }
  return <pre className="artifact">{JSON.stringify(result.data, null, 2)}</pre>;
}

export function Artifacts({
  jobId,
  files,
}: {
  jobId: string;
  files: { name: string; media_type: string; bytes: number }[];
}) {
  const [name, setName] = useState<string | null>(null);
  const [body, setBody] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!name) return;
    setBody(null);
    setError(null);
    api
      .getArtifact(jobId, name)
      .then((text) => setBody(typeof text === "string" ? text : JSON.stringify(text, null, 2)))
      .catch((exc) => setError(String(exc)));
  }, [jobId, name]);

  if (files.length === 0) {
    return <div className="empty">还没有写出任何产物文件。</div>;
  }

  return (
    <div>
      <div className="row" style={{ marginBottom: 12 }}>
        {files.map((file) => (
          <button
            key={file.name}
            className={name === file.name ? "primary" : ""}
            onClick={() => setName(file.name)}
          >
            {file.name}
            <span className="muted small"> {formatBytes(file.bytes)}</span>
          </button>
        ))}
      </div>
      {error && <div className="err">{error}</div>}
      {name && !body && !error && <div className="muted">正在加载 {name}…</div>}
      {body && <pre className="artifact">{body}</pre>}
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)}MB`;
}
