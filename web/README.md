# EvoSci web UI

Submit a run from a browser, watch it step through in real time, and read the results.

**This is a tool for a trusted LAN, not an internet service.** See [Security](#security).

The `evosci` package itself stays dependency-free. FastAPI and uvicorn live only here.

## Setup

```bash
python -m venv web/.venv
web/.venv/bin/pip install -r web/requirements.txt

cd web/frontend && npm install && npm run build   # builds frontend/dist
```

## Running

```bash
cp web/.env.example web/.env      # fill in PRISMLLM_API_KEY
set -a; . web/.env; set +a

PYTHONPATH=src web/.venv/bin/python -m web.serve
```

It prints the URL and a freshly generated token, and writes the token to `web/.token`
(mode 600, gitignored). Paste the token into the browser's gate; it is held in
`sessionStorage`, so closing the tab clears it.

For frontend development, run `npm run dev` in `web/frontend` instead — Vite serves on
5173 and proxies `/api` to port 8000.

## What it does

**New run** — the form is generated from `dataclasses.fields(EvoSciConfig)` at request
time, so it cannot drift from `config.py`. Presets come from `examples/*.toml`. An
indicator shows whether the API-key environment variable named by the config is set;
submission is blocked when a real provider is selected and it is not.

**Live** — the left column groups progress steps by round with per-step wall time; the
right lists every model call with its duration, retry count, and outcome. Both stream over
SSE backed by an append-only `events.jsonl`, so refreshing mid-run resumes at the cursor
rather than restarting. The stream survives a service restart, because each run is its own
process group rather than a thread.

**Results** — ideas per round with their reviews and meta-review. Ranking comes from the
tournament in `report.md`.

**Feedback ledger** — the signature view. For each round after the first it reconstructs
what `engine.py` actually carried into the prompt: top-3 ideas by fitness → their
meta-review suggestions → deduplicated → truncated at 8. Carried suggestions show their
source; the ones past the cut are greyed out and labelled as truncated. This causal edge is
reproducible but recorded in no artifact.

It is a provenance record of **what went into the prompt**, not a claim about what the
model adopted.

**Graph / diagnostics / artifacts** — entities, clusters, and raw files.

## Honest limits

- **No token counts or costs.** `llm.py` discards the `usage` field from the response, so
  the UI reports response size in characters as a labelled proxy.
- **Scores are not comparable across ideas.** With `reviewer_count = 1` each idea is scored
  in isolation with no shared anchor, so there is no leaderboard and no trend line. A
  banner says so wherever scores appear.
- **Runs started outside this UI have no telemetry.** They are listed as `archived` with
  complete artifacts but no step timeline or per-call log, and they are **read-only** —
  resuming one would rewrite its `state.json` in place, so the API refuses it (409). Copy
  the directory under the runs root first.
- **Older runs degrade rather than break.** Runs predating the cluster schema have no
  `clusters` key and runs predating the crossover schema record bare entity ids; both
  render an explicit note instead of an empty panel.
- **Cancelling loses the round in flight**, not the completed ones. There is no
  pause-within-a-round.

## Security

Every `/api` route except `/api/health` requires `Authorization: Bearer <token>`. The token
is compared with `compare_digest` and travels only in a header — never a URL or cookie, so
it stays out of access logs and there is no CSRF surface. More than 10 failures per minute
from one address gets a 429.

The API key is never in a request body, never in a response, and never in a TOML file. It
reaches a run only through the child process's environment, and a run is given only the one
variable its own config names. `asdict(EvoSciConfig())` cannot leak it — `api_key` is a
property, so only `api_key_env` serializes.

Artifact names are checked against a fixed whitelist and never path-joined with user input.

Binding to `0.0.0.0` prints an explicit warning and refuses to start if the token is
shorter than 24 characters.

**Residual risk, stated plainly.** Plaintext HTTP on a LAN means the token can be sniffed
on open Wi-Fi. Someone holding it can read every run's topics, ideas, and reviews, and
submit runs that spend API credit. They cannot obtain the API key and cannot read arbitrary
files. `EVOSCI_MAX_CONCURRENT` and `EVOSCI_MAX_ROUNDS` bound the spend.

Prompts and generated content are recorded in full. Do not put unpublished sensitive
material into third-party model prompts.
