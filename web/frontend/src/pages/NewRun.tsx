import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { ApiError, api, type Defaults, type KeyEntry } from "../api/client";
import { ConfigField } from "../components/ConfigField";

const PRIMARY_SECTIONS = ["run", "llm"];

// Submitting navigates away, so without this the whole form — topic, disciplines, and
// every config override — is gone by the time you come back to start a second run.
const DRAFT_KEY = "evosci.newrun.draft";

interface Draft {
  topic: string;
  disciplines: string;
  label: string;
  overrides: Record<string, unknown>;
  advanced: boolean;
}

function loadDraft(): Draft | null {
  try {
    const raw = localStorage.getItem(DRAFT_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    return {
      topic: String(parsed.topic ?? ""),
      disciplines: String(parsed.disciplines ?? "物理学, 计算机科学"),
      label: String(parsed.label ?? ""),
      overrides:
        parsed.overrides && typeof parsed.overrides === "object" ? parsed.overrides : {},
      advanced: Boolean(parsed.advanced),
    };
  } catch {
    return null;
  }
}

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
  const draft = useMemo(loadDraft, []);
  const [defaults, setDefaults] = useState<Defaults | null>(null);
  const [topic, setTopic] = useState(draft?.topic ?? "");
  const [disciplines, setDisciplines] = useState(
    draft?.disciplines ?? "物理学, 计算机科学",
  );
  const [label, setLabel] = useState(draft?.label ?? "");
  const [overrides, setOverrides] = useState<Record<string, unknown>>(draft?.overrides ?? {});
  const [advanced, setAdvanced] = useState(draft?.advanced ?? false);
  const [keyEntry, setKeyEntry] = useState<KeyEntry | null>(null);
  const [keyInput, setKeyInput] = useState("");
  const [keyBusy, setKeyBusy] = useState(false);
  const [keyError, setKeyError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [invalidKeys, setInvalidKeys] = useState<string[]>([]);
  const [rejected, setRejected] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    api.defaults().then(setDefaults).catch((exc) => setError(String(exc)));
  }, []);

  useEffect(() => {
    const draftNow: Draft = { topic, disciplines, label, overrides, advanced };
    try {
      localStorage.setItem(DRAFT_KEY, JSON.stringify(draftNow));
    } catch {
      // A full or disabled localStorage costs the convenience, not the form.
    }
  }, [topic, disciplines, label, overrides, advanced]);

  const effective = (path: string): unknown => {
    if (path in overrides) return overrides[path];
    if (!defaults) return undefined;
    const [section, name] = path.split(".");
    return defaults.config[section]?.[name];
  };

  const provider = String(effective("llm.provider") ?? "");
  const apiKeyEnv = String(effective("llm.api_key_env") ?? "");
  const needsKey = provider === "openai-compatible";
  const keyPresent = keyEntry?.present ?? null;

  useEffect(() => {
    if (!needsKey || !apiKeyEnv) {
      setKeyEntry(null);
      return;
    }
    let alive = true;
    api
      .envCheck([apiKeyEnv])
      .then((result) => {
        if (!alive) return;
        setKeyEntry({ name: apiKeyEnv, present: Boolean(result[apiKeyEnv]), source: null });
      })
      .catch(() => alive && setKeyEntry(null));
    return () => {
      alive = false;
    };
  }, [needsKey, apiKeyEnv]);

  async function saveKey() {
    setKeyBusy(true);
    setKeyError(null);
    try {
      setKeyEntry(await api.putKey(apiKeyEnv, keyInput));
      setKeyInput("");
    } catch (exc) {
      setKeyError(
        exc instanceof ApiError && typeof exc.detail === "string"
          ? exc.detail
          : exc instanceof Error
            ? exc.message
            : String(exc),
      );
    } finally {
      setKeyBusy(false);
    }
  }

  async function removeKey() {
    setKeyBusy(true);
    setKeyError(null);
    try {
      setKeyEntry(await api.deleteKey(apiKeyEnv));
    } catch (exc) {
      setKeyError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setKeyBusy(false);
    }
  }

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
    setRejected({});
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
        const detail = exc.detail as {
          message?: string;
          invalid_keys?: string[];
          rejected?: Record<string, string>;
        };
        setError(detail.message ?? String(exc));
        setInvalidKeys(detail.invalid_keys ?? []);
        setRejected(detail.rejected ?? {});
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

      {provider === "heuristic" && (
        <div className="banner">
          <strong>当前选择的是离线启发式后端。</strong> 它不会发出任何模型请求，几毫秒就能跑完，
          产出的想法是本地模板生成的，不是模型写的。下面的 <code>model</code> 与{" "}
          <code>base_url</code> 在这个模式下<strong>完全不会被读取</strong>。
          要跑真实模型，请把 <code>provider</code> 改成 <code>openai-compatible</code>。
        </div>
      )}

      {needsKey && (
        <div className={keyPresent ? "banner info" : "banner"}>
          {keyPresent ? (
            <>
              已找到 API key <code>{apiKeyEnv}</code>
              {keyEntry?.source === "ui" ? "（本界面保存）" : ""}。密钥本身绝不会回传到此页面。
            </>
          ) : (
            <>
              provider 为 <code>openai-compatible</code> 时需要名为 <code>{apiKeyEnv}</code>{" "}
              的密钥，但服务端没有。在下面填入即可 —— 它保存在服务端，不会写进任务目录。
            </>
          )}
          <div className="row" style={{ marginTop: 10 }}>
            <div style={{ flex: "2 1 280px" }}>
              <label htmlFor="apikey">
                API 密钥（存为 <code>{apiKeyEnv}</code>）
              </label>
              <input
                id="apikey"
                type="password"
                autoComplete="off"
                style={{ width: "100%" }}
                value={keyInput}
                placeholder={keyPresent ? "已保存 —— 填入新值可替换" : "sk-…"}
                onChange={(event) => setKeyInput(event.target.value)}
              />
            </div>
            <button
              type="button"
              className="primary"
              disabled={keyBusy || !keyInput.trim() || !apiKeyEnv}
              onClick={saveKey}
            >
              {keyBusy ? "保存中…" : "保存密钥"}
            </button>
            {keyEntry?.source === "ui" && (
              <button type="button" className="danger" disabled={keyBusy} onClick={removeKey}>
                删除
              </button>
            )}
          </div>
          <div className="small muted" style={{ marginTop: 6 }}>
            上面 <code>api_key_env</code> 那一栏填的是<strong>变量名</strong>（如{" "}
            <code>PRISMLLM_API_KEY</code>），不是密钥。密钥填在这里。
          </div>
          {keyError && (
            <div className="err small" style={{ marginTop: 6 }}>
              {keyError}
            </div>
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
              未知或不允许的配置键：{invalidKeys.join(", ")}
            </div>
          )}
          {Object.entries(rejected).map(([path, reason]) => (
            <div className="small" style={{ marginTop: 6 }} key={path}>
              <code>{path}</code> —— {reason}
            </div>
          ))}
        </div>
      )}

      <div className="row">
        <button className="primary" type="submit" disabled={submitting || blocked}>
          {submitting ? "提交中…" : "开始运行"}
        </button>
        <button
          type="button"
          onClick={() => {
            setTopic("");
            setLabel("");
            setDisciplines("物理学, 计算机科学");
            setOverrides({});
          }}
        >
          清空表单
        </button>
        <span className="small muted">表单内容会自动保存，提交后回到这里仍在。</span>
      </div>
    </form>
  );
}
