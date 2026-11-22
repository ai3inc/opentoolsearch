# Benchmark data

Everything needed to check or re-run the comparisons in the [README](../README.md) and [BENCHMARKS.md](../BENCHMARKS.md).

## 1. `toolret-per-source/` - the public benchmark

ToolRet (Shi et al., ACL 2025 Findings) in its per-source setting: each question is searched
only among the tools of its own source dataset, with the benchmark's own labels.

- `runs_ids.json` - ids only: the 769 questions used (`cases`), the tool pool of each of the 32
  subsets (`pools`), and each system's top results per question (`runs`). ToolRet's texts are
  not redistributed here; they are read from Hugging Face at pinned revisions
  (`opentoolsearch/data.py`).
- Selection rule: every subset whose own pool has at most 2,000 tools (ToolBench, APIGen and
  ToolACE have 3,605 to 16,072 and are left out), 25 questions per subset, `random.seed(11)`.
- `dense_runs_ids.json` - the same for the dedicated open retrievers (top 20 ids per question,
  the query format they were scored with).
- Rebuild and re-run: `scripts/v4_public_prep.py` (pools, questions, open models),
  `scripts/v4_run_gemini.py`, `scripts/v5_run_openai.py` and `scripts/v5_run_anthropic.py`
  (given `data/v4/bench_public.json` they run in public mode: every tool of the question's
  source dataset deferred, name + a 240-character short description, empty parameters - the same
  conditions as the Gemini runner), `scripts/v4_run_decisions.py`, `scripts/v4_time_ours.py`,
  `scripts/v7_dense_models.py` and `scripts/v7_tool_de.py` (dedicated retrievers; they rebuild the
  pools with the code of `v4_public_prep.py` and check them against `bench_public.json`),
  `scripts/v4_score.py`. Scores: `results/public_head_to_head.json` (with a per-subset
  breakdown) and `results/dense_tool_retrievers.json`.
- The hosted systems' raw outputs for this benchmark carry ToolRet tool names and are kept out
  of the repository; their top results are in `runs_ids.json`. A 256-question sample (8 per subset,
  `random.Random(20261121)`) was used while Claude Opus 5.5 was being run; every system is also
  scored on it (`ndcg10_s8`), and the full 769 are now complete for all of them.

## 2. `ots-mcp-1k/` - our own set

One library of 1,000 MCP tools and 200 requests about MCP servers held out of training.

- `bench.json` - the requests with their labelled tools, the 1,000 tool declarations as sent to
  the APIs, the interchangeable-tool judgments (`equiv`), and the open models' scores.
- `runs/` - what each hosted system returned for every request, with its usage and latency:
  Gemini (4-hydrogen, 3.8-flash, 3.6-flash), OpenAI (gpt-6.2-sol and gpt-6-luna; the first full
  gpt-6.2-sol run, in which its prompt cache never hit, is kept as `*-run1-no-cache-hits`; gpt-5.4),
  Anthropic (Opus 5.5 at effort max; Haiku 5.5 with the search forced and, kept for the record,
  not forced; Fable 5.2), Jev and Gemini Decision.
- `dense_runs.json` - top 20 per request of the dedicated retrievers, in English and in Korean.
- Source: [Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) (Apache-2.0):
  real MCP-server tools and synthetic requests labelled with the tools that solve them.
  Interchangeability was judged by one LLM (`scripts/judge_equiv.py`), with no human check.
- Built by `scripts/v2_build_dev.py`, `scripts/v2_equiv_candidates.py`, `scripts/v3_bench_prep.py`;
  scored by `scripts/v3_score.py`. Scores: `results/head_to_head.json`.
- This set was built by the authors of the model it evaluates. It resembles the model's
  training data, the candidates for "interchangeable" were chosen with the same embedder the
  system uses, and the published weights saw training rows from these servers.

## 2c. `first-stage-swap/` - other first stages behind the reranker

For each of seven first-stage retrievers (including OpenToolSearch Embed 0.6B, the default since 0.2, and its
base Qwen3-Embedding-0.6B): its top-50 candidates (ids and scores) and the final ranking after the published
reranker (top 25, fusion 0.5), on the three tests - `*.json.gz`, written by `scripts/v8_first_stage_rerank.py`
and scored by `scripts/v8_score.py` (`results/first_stage_plus_reranker.json`); intervals in
`results/first_stage_paired_bootstrap.json` (`scripts/v10_paired_bootstrap.py`). The default pair end to end
through the library - quality on OTS-MCP-1k, latency, throughput: `scripts/v10_default_speed.py`,
`results/default_speed*.json`.

## 2b. `ots-mcp-1k-ko/` - the same requests in Korean

- `ko_queries.json` - for every request: the English text, the Korean translation (Claude Sonnet
  5.5, through BizRouter), the check by a second model (gemini-3.6-flash compared both and
  rated 1-5, listing details missing or added: 197 of 200 rated 5), and a `patched` note on the
  three translations we corrected by hand (a stray introduction line, `university`, `facial`).
- `open_scores.json` - the first-stage candidates and the reranker scores of the open models for
  the English and the Korean requests (same tool texts), from which `scripts/v6_score_open.py`
  computes every open system.
- `runs/` - the hosted tool-search runs on the Korean requests and Jev / Gemini Decision at 25 and
  50 candidates (candidates from the Korean first stage).
- Re-run: `scripts/v6_translate_ko.py`, `scripts/v6_korean_open.py` (GPU), the hosted runners with
  `data/v6/bench.json` as the benchmark file, `scripts/v6_score_ko.sh`, `scripts/v6_score_decisions.py`.
  Scores: `results/hosted_tool_search_ko.json`, `korean_open.json`, `korean_decisions.json`.

## 3. ToolRet, full corpus

All 44,453 tools in one pool, 7,961 questions. Every test run we made is listed in
[`results/LEDGER.md`](../results/LEDGER.md); per-run metrics are under `results/`.

## Dates and prices

API runs: 21 Nov 2026. List prices used, USD per million tokens (BizRouter model catalogue,
checked against the vendors' price pages where public):

| Model | Input | Output | Cache read | Cache write |
|---|---|---|---|---|
| gemini-4-hydrogen | 1.50 | 7.50 | 0.075 | - |
| gemini-3.8-flash, gemini-3.6-flash | 0.75 | 3.75 | 0.075 | - |
| gpt-6.2-sol | 2.00 | 10.00 | 0.10 | 2.50 |
| gpt-6-luna | 0.10 | 0.50 | 0.01 | 0.125 |
| gpt-5.4 | 2.50 | 15.00 | 0.25 | - |
| Claude Opus 5.5 | 4.00 | 20.00 | 0.20 | 5.00 |
| Claude Haiku 5.5 | 0.10 | 0.50 | 0.01 | 0.125 |
| Claude Fable 5.2 | 8.00 | 40.00 | 0.20 | 10.00 |

`scripts/hosted_cost.py` turns each API's reported usage into two costs: every input token at
the full rate ("list"), and with the reported cache hits and writes at their own prices
("cache-aware"). Jev and Gemini Decision were called through a gateway's `/v1/decisions`; their
cost is the amount it billed, in KRW, converted at 1,400 per USD. Open-model cost is GPU time
on one B300 at ₩9,931/h ($7.09/h). Every run, its settings, its failures and its cost are in
[`results/LEDGER.md`](../results/LEDGER.md).
