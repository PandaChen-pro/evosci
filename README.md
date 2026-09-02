# EvoSci Reproduction

An independent, runnable implementation of the workflow described in **EvoSci:
A Bio-Inspired Multi-Agent Framework for the Evolution of Scientific
Discovery** (ACL 2026). This repository is not the authors' official code.

The implementation covers the complete operational loop described in the
paper:

1. construct a discipline-entity knowledge graph;
2. turn a topic and target disciplines into structured problem clusters;
3. assemble role-matched scientist agents and collect research notes;
4. generate and refine falsifiable research ideas;
5. run independent reviewers and an aggregated meta-review;
6. use review fitness for entity selection, crossover, variation, inheritance,
   and pruning;
7. feed the evolved entity population into the next round;
8. checkpoint every round and produce a tournament-ranked Markdown report.

## What is reproduced and what is inferred

The paper publishes the high-level workflow and prompts but not implementation
details such as crossover rules, variation rates, selection ratios, graph
thresholds, pruning rules, exact model sampling settings, or evaluator runtime.
This implementation therefore makes those rules explicit in TOML rather than
presenting them as the authors' original choices.

The default entity fitness is:

```text
fitness = (0.30 * novelty
         + 0.20 * feasibility
         + 0.25 * validity
         + 0.15 * excitement
         + 0.10 * overall) / 10
```

Referenced entities receive the evaluated signal. Unreferenced entities decay.
The best fraction of each discipline population is inherited, top entities are
recombined, model-proposed variations are added, and low-fitness or overflow
entities are pruned.

## Requirements

- Python 3.11 or newer
- No mandatory third-party Python dependencies
- An API key only when using the OpenAI-compatible backend

## Quick start: offline deterministic demo

The offline backend is designed for tests and workflow inspection. It does not
claim to generate publication-quality science.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

evosci run \
  --topic "neural network grokking" \
  --disciplines "computer science,physics,cognitive science" \
  --config examples/offline.toml
```

The command prints the run directory. Each run contains:

```text
config.json   exact resolved configuration
graph.json    evolving discipline-entity graph and weighted edges
state.json    problems, ideas, individual reviews, and evolution events
report.md     tournament-ranked final research proposals
```

Inspect or resume a run:

```bash
evosci inspect runs/<run-directory>
evosci resume runs/<run-directory>
```

`resume` continues from the last completed checkpoint until `run.rounds` is
reached. A completed run is a no-op except for regenerating the report.

## Run with a real model

The backend calls the standard `POST /chat/completions` interface and asks for a
JSON object. Edit the model and endpoint in
`examples/openai-compatible.toml`, then run:

```bash
export OPENAI_API_KEY="..."

evosci run \
  --topic "neural network grokking" \
  --disciplines "computer science,physics,cognitive science" \
  --config examples/openai-compatible.toml
```

For another compatible provider, change `base_url`, `model`, and `api_key_env`.
Some nominally compatible services do not implement `response_format`; remove
that field in `OpenAICompatibleBackend` if required by the service.

For reasoning models behind a short gateway timeout, set
`reasoning_effort = "low"` and lower `max_completion_tokens`. Both controls are
passed through to `/chat/completions`.

Omit `max_completion_tokens` from TOML to omit it from the API request entirely.
Set `stream = true` to consume standard chat-completion SSE events, which can
avoid gateway timeouts for long reasoning requests.

The included `examples/prismllm.toml` is configured for the PrismLLM endpoint
and `gpt-5.6-sol`. Set `PRISMLLM_API_KEY` in the environment before using it;
never commit the key to a TOML file.

When enabled, the retrieval layer queries the arXiv Atom API. Network failures
degrade to an empty literature set so a checkpointed run can continue.

## Architecture

```text
Topic + disciplines
       |
       v
MentorAgent -> problem clusters <- discipline-entity KnowledgeGraph
       |
       v
ScientistRegistry -> ResearchTeam -> refined ideas
                                      |
                                      v
                              ReviewerPanel
                         individual + meta-review
                                      |
                                      v
                             EntityEvolution
                      select/crossover/vary/prune
                                      |
                                      +----> next round
```

Core modules:

- `knowledge.py`: seed entities, text similarity, edges, clustering, persistence;
- `agents.py`: mentor, scientist selection, research team, reviewer panel, tournament;
- `evolution.py`: feedback propagation and population operations;
- `retrieval.py`: optional arXiv retrieval;
- `engine.py`: round orchestration, checkpoints, resume, and reports;
- `llm.py`: deterministic and OpenAI-compatible backends;
- `models.py`: serializable domain objects.

## Testing

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The suite covers configuration validation, graph construction and
serialization, feedback-driven evolution, multi-round orchestration, artifact
generation, and resume behavior.

## Important limitations

- It is a clean-room reproduction, not an exact replication of undisclosed
  author code.
- Scientist profiles included here are synthetic role profiles. The paper's
  Digital Scientist dataset can be loaded by constructing `ScientistProfile`
  objects and passing a custom `ScientistRegistry`.
- The built-in text similarity is dependency-free lexical similarity. A
  production implementation should replace it with a scientific embedding
  model while keeping the graph API unchanged.
- LLM review is a search signal, not evidence that an idea is scientifically
  correct. Human review and executed experiments remain necessary.
- The implementation records full generated content. Do not put secrets or
  sensitive unpublished material into third-party model prompts.

## License

MIT. The paper and its datasets retain their respective licenses.
