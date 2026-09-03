import type { FieldSpec } from "../api/client";

/** Controls are generated from the backend's dataclass spec, never from a hand-copied
 *  mirror of config.py — that copy would drift the moment a default changes. */
export function ConfigField({
  spec,
  value,
  onChange,
}: {
  spec: FieldSpec;
  value: unknown;
  onChange(path: string, value: unknown): void;
}) {
  const id = `f-${spec.path}`;
  const shown = value === undefined ? spec.default : value;

  if (spec.type === "bool") {
    return (
      <div>
        <label htmlFor={id}>{spec.name}</label>
        <input
          id={id}
          type="checkbox"
          checked={Boolean(shown)}
          onChange={(event) => onChange(spec.path, event.target.checked)}
        />
        <div className="small muted">默认 {String(spec.default)}</div>
      </div>
    );
  }

  const numeric = spec.type === "int" || spec.type === "float";

  if (spec.choices) {
    return (
      <div>
        <label htmlFor={id}>{spec.name}</label>
        <select
          id={id}
          style={{ width: "100%" }}
          value={shown === null || shown === undefined ? "" : String(shown)}
          onChange={(event) => {
            const raw = event.target.value;
            onChange(spec.path, raw === "" && spec.optional ? null : raw);
          }}
        >
          {spec.choices.map((choice) => (
            <option key={choice} value={choice}>
              {choice === "" ? "（不设置）" : choice}
            </option>
          ))}
        </select>
        <div className="small muted">
          默认 {spec.default === null || spec.default === "" ? "无" : String(spec.default)}
        </div>
      </div>
    );
  }

  return (
    <div>
      <label htmlFor={id}>{spec.name}</label>
      <input
        id={id}
        style={{ width: "100%" }}
        type={numeric ? "number" : "text"}
        step={spec.type === "float" ? "any" : 1}
        value={shown === null || shown === undefined ? "" : String(shown)}
        placeholder={spec.optional ? "未设置" : ""}
        onChange={(event) => {
          const raw = event.target.value;
          if (raw === "") {
            onChange(spec.path, spec.optional ? null : "");
            return;
          }
          if (spec.type === "int") onChange(spec.path, Number.parseInt(raw, 10));
          else if (spec.type === "float") onChange(spec.path, Number.parseFloat(raw));
          else onChange(spec.path, raw);
        }}
      />
      <div className="small muted">
        {spec.type}
        {spec.optional ? " · 可选" : ""} · 默认{" "}
        {spec.default === null ? "无" : String(spec.default)}
      </div>
    </div>
  );
}
