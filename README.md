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
6. use review fitness to select, cross over, vary, inherit, and prune **entity
   clusters**, which the paper calls the basic units of evolution, and to split
   or retire the clusters themselves;
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

Clusters, not individual entities, are the unit of evolution. Review feedback is
attributed to a cluster by how much of its membership an idea cited, and only a
cluster that received feedback moves — an unexamined cluster is left untouched
rather than decayed, since a round that did not look at it is absence of
evidence, not a negative verdict. A credited cluster then passes its fitness to
its own members.

The four operators run at cluster level. **Selection** filters clusters, with
staleness and crowding terms so a narrow early lead cannot monopolise the
population. **Crossover** migrates one entity between clusters of the same
discipline; a cluster donates when it is fitter *or* larger than the
destination, so migration both propagates proven material and relieves crowding.
It mints no new entity, which is what keeps names from concatenating and the
graph from fusing into one blob. **Variation** injects model-proposed concepts
into a specific cluster, and **pruning** removes entities that are both in the
weakest fraction and below a floor relative to their discipline's mean — never
taking a cluster below `min_cluster_size`.

Clusters persist across rounds rather than being recomputed, and the *set* of
them changes too, so a run can replace one line of inquiry with another instead
of only reshuffling members. A cluster that has grown into two unrelated groups
splits in two when the split improves intra-cluster similarity by
`fission_min_gain`. A cluster no idea has touched for `cluster_max_stale` rounds
is retired, and its members rejoin whichever cluster they are nearest — so
retirement discards a grouping, not the concepts in it. Staleness rather than low
fitness is the trigger, because an unexamined cluster's fitness never moves and
so can never express failure.

Both are deliberately rate-limited, since extinction that is merely permitted
becomes the collapse this design exists to prevent. At most one cluster is born
and one retired per discipline per round, and retirement stops at a floor of
three clusters — what crossover and the no-collapse criterion need. The limit is
per *round*, not per call: a round reads its clusters many times over — fitness
attribution, selection, pruning, diagnostics — and structural change on every
read compounds instead of stepping.

Each run also writes `diagnostics.md`, which measures the structural properties
the paper claims in Appendix D.3-D.4 (heritable variation, fitness-guided
selection, diversity maintained without collapse). It deliberately does not
score idea quality: the paper reports quality gains as non-monotonic, and the
offline heuristic reviewer scores partly on length, so it cannot evidence real
quality.

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
config.json      exact resolved configuration
graph.json       evolving discipline-entity graph, weighted edges, and clusters
state.json       problems, ideas, individual reviews, and evolution events
diagnostics.json structural health metrics per round
diagnostics.md   the same metrics as readable tables
report.md        tournament-ranked final research proposals
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
             cluster select/migrate/vary/prune/split/retire
                                      |
                                      +----> next round
```

Core modules:

- `knowledge.py`: seed entities, text similarity, edges, clustering, persistence;
- `agents.py`: mentor, scientist selection, research team, reviewer panel, tournament;
- `evolution.py`: feedback propagation and cluster-level operators;
- `diagnostics.py`: read-only structural metrics over a finished or partial run;
- `retrieval.py`: optional arXiv retrieval;
- `engine.py`: round orchestration, checkpoints, resume, and reports;
- `llm.py`: deterministic and OpenAI-compatible backends;
- `models.py`: serializable domain objects.

## Testing

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The suite covers configuration validation, graph construction and
serialization, cluster persistence and membership invariants, feedback-driven
evolution, structural diagnostics, multi-round orchestration, artifact
generation, and resume behavior. Most of it is regression tests for structural
failures this implementation has actually exhibited and is built to avoid:
clusters collapsing into one blob, entity names growing by concatenation, a
duplicate cluster id emptying crossover's candidate list, extinction wiping a
discipline down to a single cluster in one round, structural change firing on
every read instead of once per round, and one cluster becoming a sink that
absorbs every migration without ever being selected. Each of those tests fails
against the version of the code that had the defect.

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
