import { useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";

import { api, clearToken, getToken, setToken } from "./api/client";
import NewRun from "./pages/NewRun";
import RunList from "./pages/RunList";
import RunDetailPage from "./pages/RunDetail";

export default function App() {
  const [token, setTokenState] = useState(getToken());
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    if (!token) {
      setChecked(true);
      return;
    }
    let alive = true;
    api
      .defaults()
      .then(() => alive && setChecked(true))
      .catch(() => {
        if (!alive) return;
        clearToken();
        setTokenState("");
        setChecked(true);
      });
    return () => {
      alive = false;
    };
  }, [token]);

  if (!token) {
    return <TokenGate onSubmit={(value) => { setToken(value); setTokenState(value); }} />;
  }
  if (!checked) return <div className="main muted">正在校验 token…</div>;

  return (
    <div className="shell">
      <aside className="sidebar">
        <h1>EvoSci</h1>
        <div className="tagline">仿生学科研想法演化</div>
        <nav>
          <NavLink to="/runs" className={({ isActive }) => (isActive ? "active" : "")}>
            任务列表
          </NavLink>
          <NavLink to="/new" className={({ isActive }) => (isActive ? "active" : "")}>
            新建任务
          </NavLink>
        </nav>
        <button
          className="small"
          style={{ marginTop: 24, width: "100%" }}
          onClick={() => {
            clearToken();
            setTokenState("");
          }}
        >
          退出登录
        </button>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Navigate to="/runs" replace />} />
          <Route path="/runs" element={<RunList />} />
          <Route path="/runs/:jobId" element={<RunDetailPage />} />
          <Route path="/new" element={<NewRun />} />
          <Route path="*" element={<div className="empty">页面不存在。</div>} />
        </Routes>
      </main>
    </div>
  );
}

function TokenGate({ onSubmit }: { onSubmit(value: string): void }) {
  const [value, setValue] = useState("");
  return (
    <div className="gate panel">
      <h2>EvoSci</h2>
      <p className="muted small">
        粘贴服务端启动时打印的访问 token。它保存在 sessionStorage 里，关闭此标签页即清除。
      </p>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (value.trim()) onSubmit(value);
        }}
      >
        <label htmlFor="token">访问 token</label>
        <input
          id="token"
          type="password"
          autoFocus
          style={{ width: "100%" }}
          value={value}
          onChange={(event) => setValue(event.target.value)}
        />
        <button className="primary" style={{ marginTop: 12, width: "100%" }} type="submit">
          进入
        </button>
      </form>
    </div>
  );
}
