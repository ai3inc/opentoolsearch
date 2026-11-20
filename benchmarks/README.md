# Benchmark data

Everything needed to check or re-run the comparisons in the main README.

## 1. `toolret-per-source/` - the public benchmark

ToolRet (Shi et al., ACL 2025 Findings) in its per-source setting: each question is searched
only among the tools of its own source dataset, with the benchmark's own labels.

- `runs_ids.json` - ids only: the 769 questions used (`cases`), the tool pool of each of the 32
  subsets (`pools`), and each system's top results per question (`runs`). ToolRet's texts are
  not redistributed here; they are read from Hugging Face at pinned revisions
  (`opentoolsearch/data.py`).
- Selection rule: every subset whose own pool has at most 2,000 tools (ToolBench, APIGen and
  ToolACE have 3,605 to 16,072 and are left out), 25 questions per subset, `random.seed(11)`.
- Rebuild and re-run: `scripts/v4_public_prep.py` (pools, questions, open models),
  `scripts/v4_run_gemini.py`, `scripts/v4_run_decisions.py`, `scripts/v4_time_ours.py`,
  `scripts/v4_score.py`. Scores: `results/public_head_to_head.json` (with a per-subset breakdown).

## 2. `ots-mcp-1k/` - our own set

One library of 1,000 MCP tools and 200 requests about MCP servers held out of training.

- `bench.json` - the requests with their labelled tools, the 1,000 tool declarations as sent to
  the APIs, the interchangeable-tool judgments (`equiv`), and the open models' scores.
- `runs/` - what each hosted system returned for every request, with its usage and latency.
- Source: [Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) (Apache-2.0):
  real MCP-server tools and synthetic requests labelled with the tools that solve them.
  Interchangeability was judged by one LLM (`scripts/judge_equiv.py`), with no human check.
- Built by `scripts/v2_build_dev.py`, `scripts/v2_equiv_candidates.py`, `scripts/v3_bench_prep.py`;
  scored by `scripts/v3_score.py`. Scores: `results/head_to_head.json`.
- This set was built by the authors of the model it evaluates. It resembles the model's
  training data, the candidates for "interchangeable" were chosen with the same embedder the
  system uses, and the published weights saw training rows from these servers.

## 3. ToolRet, full corpus

All 44,453 tools in one pool, 7,961 questions. Every test run we made is listed in
[`results/LEDGER.md`](../results/LEDGER.md); per-run metrics are under `results/`.

## Dates and prices

API runs: 21 Nov 2026. Gemini list prices used: gemini-3.6-flash $0.75 / $3.75 and
gemini-4-hydrogen $1.50 / $7.50 per million input / output tokens. Jev and Gemini Decision were
called through a gateway's `/v1/decisions`; their cost is the amount it billed, in KRW, converted
at 1,400 per USD.
