import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { ApiError, api, type Defaults } from "../api/client";
import { ConfigField } from "../components/ConfigField";

const PRIMARY_SECTIONS = ["run", "llm"];

// Section keys come from the dataclass, so they stay in English on the wire; only the
// heading is translated, and an unmapped section falls back to its raw key.
const SECTION_LABEL: Record<string, string> = {
  run: "运行（run）",
  llm: "模型（llm）",
  graph: "知识图谱（graph）",
  evolution: "演化（evolution）",
  review: "评审（review）",
  team: "团队（team）",
  agents: "智能体（agents）",
  problems: "问题空间（problems）",
};

export default function NewRun() {
  const navigate = useNavigate();
  const [defaults, setDefaults] = useState<Defaults | null>(null);
  const [topic, setTopic] = useState("");
  const [disciplines, setDisciplines] = useState("物理学, 计算机科学");
  const [label, setLabel] = useState("");
  const [overrides, setOverrides] = useState<Record<string, unknown>>({});
  const [advanced, setAdvanced] = useState(false);
  const [keyPresent, setKeyPresent] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [invalidKeys, setInvalidKeys] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    api.defaults().then(setDefaults).catch((exc) => setError(String(exc)));
  }, []);

  const effective = (path: string): unknown => {
    if (path in overrides) return overrides[path];
    if (!defaults) return undefined;
    const [section, name] = path.split(".");
    return defaults.config[section]?.[name];
  };

  const provider = String(effective("llm.provider") ?? "");
  const apiKeyEnv = String(effective("llm.api_key_env") ?? "");
  const needsKey = provider === "openai-compatible";

  useEffect(() => {
    if (!needsKey || !apiKeyEnv) {
      setKeyPresent(null);
      return;
    }
    let alive = true;
    api
      .envCheck([apiKeyEnv])
      .then((result) => alive && setKeyPresent(Boolean(result[apiKeyEnv])))
      .catch(() => alive && setKeyPresent(null));
    return () => {
      alive = false;
    };
  }, [needsKey, apiKeyEnv]);

  const sections = useMemo(() => {
    if (!defaults) return [];
    const names = Array.from(new Set(defaults.spec.map((item) => item.section)));
    names.sort((a, b) => {
      const ai = PRIMARY_SECTIONS.indexOf(a);
      const bi = PRIMARY_SECTIONS.indexOf(b);
      if (ai !== -1 || bi !== -1) return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
      return a.localeCompare(b);
    });
    return names;
  }, [defaults]);

  const setField = (path: string, value: unknown) =>
    setOverrides((prev) => ({ ...prev, [path]: value }));

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setInvalidKeys([]);
    setSubmitting(true);
    try {
      const run = await api.createRun({
        topic: topic.trim(),
        disciplines: disciplines
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean),
        overrides,
        label: label.trim() || null,
      });
      navigate(`/runs/${run.job_id}`);
    } catch (exc) {
      if (exc instanceof ApiError && exc.detail && typeof exc.detail === "object") {
        const detail = exc.detail as { message?: string; invalid_keys?: string[] };
        setError(detail.message ?? String(exc));
        setInvalidKeys(detail.invalid_keys ?? []);
      } else {
        setError(exc instanceof Error ? exc.message : String(exc));
      }
    } finally {
      setSubmitting(false);
    }
  }

  if (error && !defaults) return <div className="err">{error}</div>;
  if (!defaults) return <div className="muted">正在加载默认配置…</div>;

  const blocked = needsKey && keyPresent === false;

  return (
    <form onSubmit={submit}>
      <h2>新建任务</h2>

      <div className="panel">
        <div className="row">
          <div style={{ flex: "2 1 320px" }}>
            <label htmlFor="topic">主题</label>
            <input
              id="topic"
              required
              style={{ width: "100%" }}
              value={topic}
              placeholder="例如：小型 Transformer 中的顿悟现象"
              onChange={(event) => setTopic(event.target.value)}
            />
          </div>
          <div style={{ flex: "2 1 260px" }}>
            <label htmlFor="disciplines">学科（英文逗号分隔）</label>
            <input
              id="disciplines"
              required
              style={{ width: "100%" }}
              value={disciplines}
              onChange={(event) => setDisciplines(event.target.value)}
            />
          </div>
          <div style={{ flex: "1 1 160px" }}>
            <label htmlFor="label">备注（可选）</label>
            <input
              id="label"
              style={{ width: "100%" }}
              value={label}
              onChange={(event) => setLabel(event.target.value)}
            />
          </div>
        </div>
      </div>

      {defaults.presets.length > 0 && (
        <div className="panel">
          <label>从预设开始</label>
          <div className="row">
            {defaults.presets.map((preset) => (
              <button
                key={preset.name}
                type="button"
                onClick={() => {
                  const next: Record<string, unknown> = {};
                  for (const [section, values] of Object.entries(preset.config)) {
                    for (const [name, value] of Object.entries(values)) {
                      next[`${section}.${name}`] = value;
                    }
                  }
                  setOverrides(next);
                }}
              >
                {preset.name}
              </button>
            ))}
            <button type="button" onClick={() => setOverrides({})}>
              恢复默认值
            </button>
          </div>
        </div>
      )}

      {needsKey && (
        <div className={keyPresent ? "banner info" : "banner"}>
          {keyPresent ? (
            <>
              服务端已设置环境变量 <code>{apiKeyEnv}</code>。密钥本身绝不会发送到此页面。
            </>
          ) : (
            <>
              provider 为 <code>openai-compatible</code> 时需要服务端环境里有{" "}
              <code>{apiKeyEnv}</code>，但它未设置。请 export 后重启服务，或把 provider 改成{" "}
              <code>heuristic</code> 做一次离线运行。
            </>
          )}
        </div>
      )}

      {sections
        .filter((section) => advanced || PRIMARY_SECTIONS.includes(section))
        .map((section) => (
          <div className="panel" key={section}>
            <h3 style={{ marginTop: 0 }}>{SECTION_LABEL[section] ?? section}</h3>
            <div className="grid">
              {defaults.spec
                .filter((item) => item.section === section)
                .map((item) => (
                  <ConfigField
                    key={item.path}
                    spec={item}
                    value={overrides[item.path]}
                    onChange={setField}
                  />
                ))}
            </div>
            {section === "run" && (
              <div className="small muted" style={{ marginTop: 10 }}>
                服务端将轮数上限限制为 {defaults.limits.max_rounds}。
              </div>
            )}
          </div>
        ))}

      <div className="panel">
        <label>
          <input
            type="checkbox"
            checked={advanced}
            onChange={(event) => setAdvanced(event.target.checked)}
          />{" "}
          显示全部配置分组（共 {sections.length} 组）
        </label>
      </div>

      {error && (
        <div className="banner">
          <div className="err">{error}</div>
          {invalidKeys.length > 0 && (
            <div className="small mono" style={{ marginTop: 6 }}>
              被拒绝的配置键：{invalidKeys.join(", ")}
            </div>
          )}
        </div>
      )}

      <button className="primary" type="submit" disabled={submitting || blocked}>
        {submitting ? "提交中…" : "开始运行"}
      </button>
    </form>
  );
}
