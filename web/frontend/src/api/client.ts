/** Token lives in sessionStorage and travels only in a request header.
 *
 * Never in the URL: query strings land in server access logs and Referer headers, and the
 * SSE endpoint is the one place a browser API would push us that way (EventSource cannot
 * set headers). See sse.ts for the fetch-based reader that avoids it.
 */

const TOKEN_KEY = "evosci.token";

export function getToken(): string {
  return sessionStorage.getItem(TOKEN_KEY) ?? "";
}

export function setToken(value: string): void {
  sessionStorage.setItem(TOKEN_KEY, value.trim());
}

export function clearToken(): void {
  sessionStorage.removeItem(TOKEN_KEY);
}

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown) {
    super(typeof detail === "string" ? detail : `Request failed (${status})`);
    this.status = status;
    this.detail = detail;
  }
}

export function authHeaders(): HeadersInit {
  return { Authorization: `Bearer ${getToken()}` };
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: { ...authHeaders(), ...(init.headers ?? {}) },
  });
  if (!response.ok) {
    let detail: unknown = await response.text();
    try {
      const parsed = JSON.parse(detail as string);
      detail = parsed.detail ?? parsed;
    } catch {
      // A bare 401 has no body by design.
    }
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  const type = response.headers.get("content-type") ?? "";
  return (type.includes("json") ? response.json() : response.text()) as Promise<T>;
}

export interface FieldSpec {
  path: string;
  section: string;
  name: string;
  type: string;
  optional: boolean;
  default: unknown;
}

export interface Defaults {
  config: Record<string, Record<string, unknown>>;
  spec: FieldSpec[];
  limits: { max_rounds: number };
  presets: { name: string; config: Record<string, Record<string, unknown>> }[];
}

export interface RunSummary {
  job_id: string;
  run_dir: string;
  scan_root: string;
  status: string;
  topic: string | null;
  disciplines: string[];
  label: string | null;
  created_at: string | null;
  finished_at: string | null;
  rounds_done: number;
  rounds_target: number | null;
  model: string | null;
  provider: string | null;
  managed: boolean;
  has_events: boolean;
}

export interface IdeaSummary {
  id: string;
  title: string;
  hypothesis: string;
  round_index: number;
  entity_ids: string[];
  authors: string[];
  fitness: number | null;
  review_count: number;
  meta_scores: Record<string, number | null>;
  suggestions: string[];
}

export interface RoundSummary {
  round_index: number;
  problems: Record<string, unknown>[];
  ideas: IdeaSummary[];
  evolution_summary: Record<string, unknown> & { crossovers_detailed: boolean };
}

export interface RunDetail {
  job: RunSummary;
  state: { topic: string; disciplines: string[]; rounds: RoundSummary[] } | null;
  config: Record<string, Record<string, unknown>> | null;
  artifacts: { name: string; media_type: string; bytes: number }[];
}

export interface RunEvent {
  seq: number;
  ts: string;
  type: string;
  data: Record<string, unknown>;
}

export const api = {
  health: () => request<{ ok: boolean; version: string; runs_root: string }>("/health"),
  defaults: () => request<Defaults>("/config/defaults"),
  envCheck: (names: string[]) =>
    request<Record<string, boolean>>(`/env/check?names=${encodeURIComponent(names.join(","))}`),
  createRun: (body: {
    topic: string;
    disciplines: string[];
    overrides: Record<string, unknown>;
    label?: string | null;
  }) =>
    request<RunSummary>("/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  listRuns: () => request<{ runs: RunSummary[] }>("/runs"),
  getRun: (id: string) => request<RunDetail>(`/runs/${id}`),
  getIdea: (id: string, ideaId: string) =>
    request<Record<string, unknown>>(`/runs/${id}/ideas/${ideaId}`),
  getGraph: (id: string) =>
    request<{
      entities: Record<string, unknown>[];
      edges: Record<string, string[]>;
      clusters: Record<string, unknown>[];
      has_clusters: boolean;
    }>(`/runs/${id}/graph`),
  getDiagnostics: (id: string) =>
    request<{ available: boolean; data: unknown }>(`/runs/${id}/diagnostics`),
  getLedger: (id: string) => request<Ledger>(`/runs/${id}/feedback-ledger`),
  getArtifact: (id: string, name: string) => request<string>(`/runs/${id}/artifacts/${name}`),
  eventsPage: (id: string, since: number) =>
    request<{ events: RunEvent[]; last_seq: number; active: boolean }>(
      `/runs/${id}/events/page?since=${since}&limit=2000`,
    ),
  cancel: (id: string) => request<RunSummary>(`/runs/${id}/cancel`, { method: "POST" }),
  resume: (id: string) => request<RunSummary>(`/runs/${id}/resume`, { method: "POST" }),
};

export interface LedgerSuggestion {
  text: string;
  source_idea_id: string | null;
  source_idea_title: string | null;
  source_idea_rank: number;
  source_idea_fitness: number | null;
}

export interface Ledger {
  entries: {
    into_round: number;
    from_round: number;
    candidate_count: number;
    carried: LedgerSuggestion[];
    dropped: LedgerSuggestion[];
    limit: number;
    considered_ideas: number;
    total_ideas_in_source_round: number;
  }[];
  available: boolean;
  note: string;
}
