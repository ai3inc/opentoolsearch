---
license: apache-2.0
base_model: Qwen/Qwen3-Reranker-0.6B
pipeline_tag: text-ranking
library_name: transformers
language:
- en
tags:
- tool-retrieval
- tool-search
- function-calling
- mcp
- agents
- reranker
- cross-encoder
---

# OpenToolSearch Rerank 0.6B

**A small reranker that picks the right tools for a request.** Developed by **AI3 Inc.**

Agents now carry hundreds or thousands of tool definitions (function declarations, MCP
tools). Putting all of them in the prompt costs tokens and makes the model choose worse.
*Tool search* fixes that: keep the library out of the prompt, find the few tools a request
needs, load only those. Model APIs have started to offer this on the server side;
OpenToolSearch is the open version you run yourself, with any model.

This repository is the second stage of that search: a 0.6B cross-encoder that reads a
request next to a tool definition and scores how well the tool fits.

| | |
|---|---|
| Developed by | AI3 Inc. |
| Model type | Cross-encoder reranker (yes/no head of a causal LM) |
| Base model | [`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) |
| Parameters | 0.6B (bfloat16, 1.2 GB) |
| Recommended first stage | [`nvidia/Nemotron-3-Embed-1B-BF16`](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16), unchanged |
| Language | English; tested on Korean requests (one set, see Results) |
| Licence | Apache-2.0 |

## Results

[ToolRet](https://github.com/mangopy/tool-retrieval-benchmark) (ACL 2025 Findings): 7,961
questions, 44,453 tools from 35 source datasets, every question searched against **all**
tools in one pool. nDCG@10, mean of the three categories (web, code, customized), each the
mean of its subsets - the paper's aggregation.

**Question + instruction**

| System | Size | nDCG@10 |
|---|---|---|
| MagicSelector (arXiv 2607.17751) | LLM query decomposition + 4B embedder + 4B reranker | 59.90 |
| **OpenToolSearch** (Nemotron-3-Embed-1B + this model) | 1.14B + 0.6B | **58.37** |
| Tool-DE (Tool-Embed-4B + Tool-Rank-4B) | 4B + 4B | 57.01 |
| Open-weights pipeline, nothing trained (Savirinc/toolret-open-weights) | 1.5B + BM25 + 1B | 55.33 |
| callbus-emb-8b, embedding only | 8B | 54.36 |
| Nemotron-3-Embed-1B alone (our run) | 1.14B | 52.42 |

**It is not the top of the table**: 1.5 points below the best reported system, above the
others we found, at roughly a fifth of the size and with no LLM call at query time. Other
systems' numbers are their authors'; we did not re-run them.

| Same system, other views | OpenToolSearch | Best other figure we found |
|---|---|---|
| Question alone (no instruction), top 50 reranked | **42.42** | 41.96 |
| Question + instruction, all questions pooled | **62.47** | 59.36 |

| Detail (question + instruction) | nDCG@10 | Recall@10 | Completeness@10 |
|---|---|---|---|
| Top 100 reranked | 58.37 | 68.00 | 56.26 |
| Top 50 reranked | 58.15 | 67.60 | 55.82 |
| This model's score alone, top 50 (no fusion) | 55.98 | | |

Per category at top 100: web 52.01, code 61.09, customized 62.01.

### How to read these numbers

- The final score fuses this model with the first stage:
  `z(rerank) + w * z(first stage)`, z-normalised over a request's candidates. `w` is the only
  value tuned against test questions and it is cross-fitted: the questions are split in two
  halves, `w` is picked on one half and scored on the other, both ways. A fixed `w = 0.25`
  gives 58.37, `w = 0.5` gives 58.19.
- Which base models to use, and how to train, were decided while looking at test scores.
  Every test run we made, including the ones that lost, is in the evaluation ledger shipped
  in this repository ([`LEDGER.md`](LEDGER.md)).
- ToolRet's instructions were written by a model that was shown the answer tools, so the
  with-instruction setting is easier than a plain request. The no-instruction row is closer
  to real traffic.
- Many subsets have 11 to 50 questions. Differences under about half a point are noise.

### Same first stage, fair fight

![Same first stage, different second stage](https://raw.githubusercontent.com/ai3inc/opentoolsearch/main/assets/fair.png)

Jev and Gemini Decision cannot search a library themselves; here both sides get the same
candidates from our embedder, and their cost is added to the search cost:

| Candidates | Second stage (same first stage underneath) | Public nDCG@10 | Own set nDCG@10 | Cost / 1,000 |
|---|---|---|---|---|
| 25 | **OpenToolSearch reranker** | 70.5 | 86.8 | **$0.24** |
| 25 | Jev | 76.8 | 89.3 | $0.36 |
| 25 | Gemini Decision | 77.7 | 90.0 | $0.48 |
| 50 | **OpenToolSearch reranker** | 70.8 | 87.2 | **$0.48** |
| 50 | Jev | 77.3 | 89.1 | $0.64 |
| 50 | Gemini Decision | 78.2 | 91.2 | $0.80 |

Only ours is the whole pipeline. Swapping the reranker for a hosted decision model costs 1.5 to
1.7 times more for 6 to 9 nDCG points.

### One GPU, fifteen thousand searches an hour - or seventy-five thousand

![How fast can one GPU search](https://raw.githubusercontent.com/ai3inc/opentoolsearch/main/assets/scale.png)

Rerank depth is the dial. On one B300 over 1,000 tools: depth 50 gives 70.3 nDCG@10 at 14,900
searches an hour and $0.48 per 1,000; **depth 25 is the library default** - 70.1 at 29,000 an
hour and $0.24; depth 10 gives 67.1 at 75,600 and $0.09; the embedder alone gives 62.9 at 2.1
million an hour for $0.003.
A large consumer app with 20 million turns a day and 0.3 tool searches each needs about 2,500
searches a second, which is roughly 200 GPUs at full quality.

### Against hosted tool search, decision models and dedicated retrievers

![Public benchmark scorecard](https://raw.githubusercontent.com/ai3inc/opentoolsearch/main/assets/scorecard-toolret.png)

Three tests, the request alone (no instruction). **ToolRet per-source** is the public one: each
question searched among the tools of its own source dataset, the benchmark's labels, 769
questions from 32 subsets. **OTS-MCP-1k** is ours: 1,000 MCP tools, 200 requests about held-out
servers, interchangeable tools credited - built by us, so weigh it below the public one. The
third is the same 200 requests **in Korean** (tools stay English).

Compared: the hosted tool search of Google (gemini-4-hydrogen, 3.8-flash, 3.6-flash), OpenAI
(gpt-6.2-sol, gpt-6-luna) and Anthropic (Opus 5.5 at effort max, Haiku 5.5), each reading the
whole library itself; the hosted decision models Jev and Gemini Decision, which rank the
candidates our first stage hands them; and the open dedicated tool-retrieval models we could
find weights for - Tool-DE (Tool-Embed + Tool-Rank), callbus-emb-8b, the four ToolRet-trained
retrievers - run by us with the authors' weights.

<!-- table:one -->
| System | Kind | ToolRet per-source nDCG@10 | Recall@10 | OTS-MCP-1k nDCG@10 | …in Korean | Cost / 1,000 searches | Median latency |
|---|---|---|---|---|---|---|---|
| Gemini Decision — single choice | hosted decision model (ranks our candidates) | 78.2 | 87.2 | 91.2 | 91.0 | $0.33 | 0.68 s |
| Jev — single choice | hosted decision model (ranks our candidates) | 77.3 | 85.4 | 89.1 | 89.2 | $0.16 | 0.51 s |
| Gemini API tool search · gemini-4-hydrogen | hosted tool search | 73.5 | 79.4 | 88.3 | 83.8 | $30 (list $51) | 10.0 s |
| Tool-DE (Tool-Embed-4B + Tool-Rank-4B) (8.0B) | dedicated open retriever | 72.0 | 83.1 | 91.4 | 89.5 | $0.29 | 0.149 s |
| Gemini Decision — yes/no per tool | hosted decision model (ranks our candidates) | 71.0 | 85.4 | 85.2 | — | $0.73 | 0.72 s |
| Jev — yes/no per tool | hosted decision model (ranks our candidates) | 70.9 | 84.8 | 86.1 | — | $0.20 | 0.52 s |
| Gemini API tool search · gemini-3.8-flash | hosted tool search | 70.6 | 75.3 | 85.6 | 79.0 | $17 (list $28) | 9.95 s |
| **OpenToolSearch — default (depth 25)** | open weights | **70.5** | 82.1 | 86.8 | 84.6 | $0.24 | 0.12 s |
| OpenToolSearch — depth 50 | open weights | 70.3 | 82.4 | 87.9 | 86.7 | $0.47 | 0.22 s |
| Qwen3-Reranker-0.6B as released | open baseline | 69.9 | 81.7 | 87.3 | 85.8 | $0.47 | 0.22 s |
| callbus-emb-8b (7.6B) | dedicated open retriever | 68.6 | 81.5 | 86.6 | 84.2 | $0.04 | 0.020 s |
| Tool-Embed-4B (4.0B) | dedicated open retriever | 67.3 | 80.0 | 86.6 | 81.5 | $0.04 | 0.021 s |
| Gemini API tool search · gemini-3.6-flash | hosted tool search | 67.0 | 71.6 | 88.2 | 84.1 | $18 (list $28) | 6.24 s |
| OpenAI tool search · gpt-6.2-sol | hosted tool search | 66.6 | 68.1 | 82.4 | 81.1 | $34 | 3.43 s |
| Qwen3-Embedding-8B (7.6B) | general embedder, reference | 65.5 | 78.2 | 84.4 | 82.9 | $0.04 | 0.020 s |
| OpenAI tool search · gpt-6-luna | hosted tool search | 65.5 | 65.2 | 88.2 | 85.5 | $0.35 (list $1.74) | 3.15 s |
| Tool-Embed-0.6B (596M) | dedicated open retriever | 64.0 | 77.6 | 86.3 | 78.0 | $0.03 | 0.017 s |
| Qwen3-Embedding-0.6B (596M) | general embedder, reference | 63.6 | 77.4 | 83.6 | 79.0 | $0.03 | 0.016 s |
| ToolRet-trained-bge-large-en-v1.5 (335M) | dedicated open retriever | 63.3 | 77.0 | 81.4 | 14.7 | $0.01 | 0.007 s |
| Nemotron-3-Embed-1B alone | open baseline | 62.9 | 76.6 | 78.4 | 74.2 | $0.00 | 0.002 s |
| ToolRet-trained-e5-large-v2 (335M) | dedicated open retriever | 60.5 | 73.8 | 82.5 | 21.3 | $0.01 | 0.007 s |
| Claude tool search · Opus 5.5 (effort max) | hosted tool search | 60.1 | 66.2 | 61.8 | 58.9 | $62 | 11.8 s |
| ToolRet-trained-e5-base-v2 (109M) | dedicated open retriever | 59.9 | 73.8 | 83.9 | 12.6 | $0.01 | 0.004 s |
| ToolRet-trained-bge-base-en-v1.5 (109M) | dedicated open retriever | 59.8 | 73.5 | 80.9 | 15.0 | $0.01 | 0.005 s |
| Claude tool search · Haiku 5.5 | hosted tool search | 53.3 | 60.9 | 57.1 | 56.5 | $0.52 | 2.53 s |
<!-- /table:one -->

- **Quality: in the middle, and it depends on the benchmark.** Default OpenToolSearch (70.5
  public) is above OpenAI's tool search (66.6, 65.5), Claude's (60.1, 53.3) and Gemini
  3.6-flash (67.0), level with gemini-3.8-flash (70.6), 3 below gemini-4-hydrogen, 7 below the
  decision models. On our own set it is 1.4 below gpt-6-luna and gemini-4-hydrogen, above
  gpt-6.2-sol (82.4) and Claude (57-62). Among open models only Tool-DE is clearly above it.
- **Speed and cost: first among systems that search a whole library.** 0.12 s and $0.24 per
  1,000 searches, against 2.5-12 s and $0.35-105 for hosted tool search; the cheapest hosted
  ones (gpt-6-luna, Haiku 5.5) are within a factor of three on cost.
- **Korean:** this model loses 1.5 points (84.6; hosted Gemini tool search loses 4-7 points,
  the English-only retrievers fall to 13-21). The decision models do not move.
- **Best open competitor: Tool-DE** (Tool-Embed-4B + Tool-Rank-4B, 8.0B parameters, 0.15 s):
  72.0 / 91.4 / 89.5 against this model's 70.5 / 86.8 / 84.6 on the public set, our set and
  Korean - ahead, even though we ran it on plain tool descriptions rather than the LLM-expanded
  ones it was trained for. callbus-emb-8b (7.6B embedding model, no second stage) scores 68.6 /
  86.6 / 84.2 at 20 ms.

Hosted costs are computed from reported token usage at Nov 2026 list prices; the cell shows the
lower of list price and cache-aware cost (list in brackets when the cache made it cheaper).
Open-model cost is GPU time on one NVIDIA B300 at its rental list price. Jev and Gemini
Decision read the whole tool pool when it has at most 200 tools, otherwise the candidates of
our first stage. Every setting (reasoning depth, forced search, failures, cost of each run) is in
the [ledger](LEDGER.md); data, scripts and every caveat:
[github.com/ai3inc/opentoolsearch](https://github.com/ai3inc/opentoolsearch#benchmarks).

<details><summary>Hosted tool search in detail, and the dedicated retrievers</summary>

<!-- table:hosted -->
| Vendor | Model | Role | Setting | ToolRet per-source | OTS-MCP-1k | …in Korean | $ / 1,000 on OTS-MCP-1k: list → with cache hits | Median latency |
|---|---|---|---|---|---|---|---|---|
| Google | gemini-4-hydrogen | top | thinking level not set; search not forced | 73.5 | 88.3 | 83.8 | $88 → $50 | 11.2 s |
| Google | gemini-3.8-flash | cost-efficient (latest) | thinking level not set; search not forced | 70.6 | 85.6 | 79.0 | $47 → $27 | 8.78 s |
| Google | gemini-3.6-flash | cost-efficient (previous) | thinking level not set; search not forced | 67.0 | 88.2 | 84.1 | $47 → $28 | 7.23 s |
| OpenAI | gpt-6.2-sol | latest | reasoning effort not set (default); search forced | 66.6 | 82.4 | 81.1 | $135 → $8.60 | 4.75 s |
| OpenAI | gpt-6-luna | cost-efficient | reasoning effort not set (default); search forced | 65.5 | 88.2 | 85.5 | $6.81 → $0.85 | 3.90 s |
| Anthropic | Opus 5.5 (effort max) | top | effort max, adaptive thinking; search not forced (documented as refused) | 60.1 | 61.8 | 58.9 | $105 → $105 | 9.54 s |
| Anthropic | Haiku 5.5 | cost-efficient | effort not set (default medium); search forced | 53.3 | 57.1 | 56.5 | $0.70 → $0.70 | 2.66 s |
<!-- /table:hosted -->

<!-- table:dense -->
| Model | Weights | Query format | ToolRet per-source | OTS-MCP-1k | …in Korean | ms per search | $ / 1,000 |
|---|---|---|---|---|---|---|---|
| Tool-DE (Tool-Embed-4B + Tool-Rank-4B) (8.0B) | `Lux1997/Tool-Embed-4B + Lux1997/Tool-Rank-4B` | qwen3-default-instruction | 72.0 | 91.4 | 89.5 | 149.4 | $0.29 |
| callbus-emb-8b (7.6B) | `yasinyaman/callbus-emb-8b@v0.2` | callbus-instruction | 68.6 | 86.6 | 84.2 | 20.3 | $0.04 |
| Tool-Embed-4B (4.0B) | `Lux1997/Tool-Embed-4B` | tool-prompt | 67.3 | 86.6 | 81.5 | 21.1 | $0.04 |
| Qwen3-Embedding-8B (7.6B) | `Qwen/Qwen3-Embedding-8B` | callbus-instruction | 65.5 | 84.4 | 82.9 | 20.1 | $0.04 |
| Tool-Embed-0.6B (596M) | `Lux1997/Tool-Embed-0.6B` | tool-prompt | 64.0 | 86.3 | 78.0 | 16.6 | $0.03 |
| Qwen3-Embedding-0.6B (596M) | `Qwen/Qwen3-Embedding-0.6B` | callbus-instruction | 63.6 | 83.6 | 79.0 | 16.3 | $0.03 |
| ToolRet-trained-bge-large-en-v1.5 (335M) | `mangopy/ToolRet-trained-bge-large-en-v1.5` | bge-instruction | 63.3 | 81.4 | 14.7 | 7.4 | $0.01 |
| ToolRet-trained-e5-large-v2 (335M) | `mangopy/ToolRet-trained-e5-large-v2` | query | 60.5 | 82.5 | 21.3 | 7.2 | $0.01 |
| ToolRet-trained-e5-base-v2 (109M) | `mangopy/ToolRet-trained-e5-base-v2` | query | 59.9 | 83.9 | 12.6 | 4.1 | $0.01 |
| ToolRet-trained-bge-base-en-v1.5 (109M) | `mangopy/ToolRet-trained-bge-base-en-v1.5` | plain | 59.8 | 80.9 | 15.0 | 4.5 | $0.01 |
<!-- /table:dense -->

<!-- table:korean -->
| System | OTS-MCP-1k in English | … in Korean | Change | What it returns (Korean) |
|---|---|---|---|---|
| Gemini Decision — single choice (50 candidates) | 91.2 | 91.0 | -0.3 | ranks 50 candidates |
| Tool-DE (Tool-Embed-4B + Tool-Rank-4B) | 91.4 | 89.5 | -1.9 | embedding search, top 20 |
| Jev — single choice (50 candidates) | 89.1 | 89.2 | +0.1 | ranks 50 candidates |
| OpenToolSearch — depth 50 † | 86.8 | 86.7 | -0.1 | ranks 50 candidates |
| Qwen3-Reranker-0.6B as released (depth 50) † | 86.3 | 85.8 | -0.5 | ranks 50 candidates |
| OpenAI tool search · gpt-6-luna | 88.2 | 85.5 | -2.7 | 1.8 tools; searched on 200 of 200 |
| OpenToolSearch — default (depth 25) † | 86.1 | 84.6 | -1.5 | ranks 25 candidates |
| callbus-emb-8b | 86.6 | 84.2 | -2.4 | embedding search, top 20 |
| Gemini API tool search · gemini-3.6-flash | 88.2 | 84.1 | -4.1 | 2.9 tools; searched on 194 of 199 |
| Gemini API tool search · gemini-4-hydrogen | 88.3 | 83.8 | -4.5 | 4.5 tools; searched on 188 of 200 |
| Qwen3-Embedding-8B | 84.4 | 82.9 | -1.5 | embedding search, top 20 |
| Tool-Embed-4B | 86.6 | 81.5 | -5.1 | embedding search, top 20 |
| OpenAI tool search · gpt-6.2-sol | 82.4 | 81.1 | -1.3 | 2.0 tools; searched on 200 of 200 |
| Gemini API tool search · gemini-3.8-flash | 85.6 | 79.0 | -6.6 | 2.7 tools; searched on 180 of 200 |
| Qwen3-Embedding-0.6B | 83.6 | 79.0 | -4.6 | embedding search, top 20 |
| Tool-Embed-0.6B | 86.3 | 78.0 | -8.4 | embedding search, top 20 |
| Nemotron-3-Embed-1B alone † | 79.0 | 74.2 | -4.8 | top 50 |
| Claude tool search · Opus 5.5 (effort max) | 61.8 | 58.9 | -2.8 | 30.8 tools; searched on 198 of 200 |
| Claude tool search · Haiku 5.5 | 57.1 | 56.5 | -0.6 | 9.6 tools; searched on 200 of 200 |
| ToolRet-trained-e5-large-v2 | 82.5 | 21.3 | -61.1 | embedding search, top 20 |
| ToolRet-trained-bge-base-en-v1.5 | 80.9 | 15.0 | -65.9 | embedding search, top 20 |
| ToolRet-trained-bge-large-en-v1.5 | 81.4 | 14.7 | -66.7 | embedding search, top 20 |
| ToolRet-trained-e5-base-v2 | 83.9 | 12.6 | -71.3 | embedding search, top 20 |
<!-- /table:korean -->

</details>

### What it costs on rented GPUs

Searching 1,000 tools, one request at a time, measured at p50 0.216 s and mean 0.233 s per
search on an NVIDIA B300. Priced at what B300s rent for in the market (quotes of 19-21
November 2026):

| B300 price source | $/GPU-hour | 1,000-tool set: $/1,000 searches | Public pools: $/1,000 searches |
|---|---|---|---|
| Runpod, community cloud | 6.94 | 0.41 | 0.36 |
| Runpod, secure cloud | 8.99 | 0.53 | 0.46 |
| Nebius | 9.50 | 0.56 | 0.49 |
| Runpod cluster (on-demand) | 10.79 | 0.63 | 0.55 |
| Spheron | 11.47 | 0.67 | 0.59 |
| Verda | 11.93 | 0.70 | 0.61 |
| Oracle, on-demand | 15.00 | 0.88 | 0.77 |
| Oracle, 3-year commitment | 7.50 | 0.44 | 0.39 |
| AWS capacity block (prepaid) | 16.15 | 0.95 | 0.83 |
| market median (getdeploying) | 10.76 | 0.63 | 0.55 |
| our on-pod list price (₩9,931/h) | 7.09 | 0.42 | 0.36 |

Our own pod rents at the equivalent of about $7.09/h (₩9,931/h), so the $0.36-0.42 rows are
what the numbers elsewhere on this page use. Even at the most expensive quote, a search
costs well under a tenth of a cent - against $9-105 per 1,000 for the flagship hosted tool
search and $0.35-0.86 for the cheapest. On 8 vCPUs, without a GPU, a search takes 57 s and the cost is about $7.61 per
1,000; the H100 is not measured (all rented out), but at Runpod's $4.39/h it would be about
$0.23 per 1,000.

### A second check: MCP servers the model never saw

ToolRet's instructions were written with the answer in view, and its labels accept only one
of several interchangeable tools. So we built a validation set of our own (OTS-MCP dev):
738 questions about 67 MCP servers held out of training, searched in a pool of 24,188 tools;
the instruction, when used, is written from the request alone. *Strict* counts only the
labelled tool; *credited* also accepts a tool an LLM judged interchangeable with it.

nDCG@10, reranker score alone, strict / credited:

| | request alone | request + instruction |
|---|---|---|
| Nemotron-3-Embed-1B alone | 32.84 / 53.26 | 37.72 / 61.23 |
| Qwen3-Reranker-0.6B as released | 39.75 / 66.06 | 41.06 / 69.23 |
| This model's recipe, retrained without the held-out servers | 44.80 / 68.00 | 47.20 / 71.60 |
| + same-tool "negatives" removed, half the questions without instruction (round 2) | 45.21 / 69.02 | 47.35 / 72.90 |

The published weights themselves saw training rows from those servers, so they are not scored
here; the third row is the same recipe trained without them. The round-2 variant is about one
point better on this set and 0.4 lower on ToolRet (57.96) - both within noise - so the
published weights are unchanged.

### How much of the benchmark's "error" is labelling

On ToolRet, this system's top result is a labelled tool for 57.2% of the questions. For the
rest, an LLM judge found the top result interchangeable with a labelled tool in 47.6% of
cases; crediting those, top-1 accuracy is 77.6%. Tool libraries pooled from many sources
contain the same tool many times, and a benchmark that labels one copy undercounts every
system. One judge, no human check; details in `LEDGER.md`.

## Use

### With transformers only

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "ai3inc/opentoolsearch-rerank-0.6b"
tok = AutoTokenizer.from_pretrained(name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).eval()
yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")

PRE = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
       'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT = "Given a user request, retrieve the tools that are needed to fulfil it"

def score(request, tool_json, instruction=None):
    text = f"{PRE}<Instruct>: {instruction or DEFAULT}\n<Query>: {request}\n<Document>: {tool_json}{SUF}"
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        logits = model(**ids).logits[0, -1]
    return (logits[yes] - logits[no]).item()          # higher = better fit

tool = '{"name": "get_weather_forecast", "description": "Get the weather forecast for a city on a given date."}'
print(score("What's the weather in Seoul tomorrow?", tool))
```

### With the OpenToolSearch library

Index, fused two-stage search, and a drop-in `tool_search` function for deferred tool loading:
[github.com/ai3inc/opentoolsearch](https://github.com/ai3inc/opentoolsearch)
(`pip install git+https://github.com/ai3inc/opentoolsearch`). The repository also holds the
evaluation and training code behind every number on this page.

```python
from opentoolsearch import ToolIndex

index = ToolIndex(model="nvidia/Nemotron-3-Embed-1B-BF16",
                  reranker="ai3inc/opentoolsearch-rerank-0.6b")
index.add(tools)                                  # OpenAI / Anthropic / Gemini / MCP style dicts
hits = index.search("refund the customer's last order", k=5)

# Deferred loading: give the model one function instead of the whole library
search_tool = index.search_tool()                 # declaration of `tool_search`
loaded = index.handle(call_arguments)             # full declarations of the best matches
```

`instruction` is optional: one line saying what kind of tool is needed, in ToolRet's style
("Given a `weather` task, retrieve tools that return a forecast for a location and date").
An agent can write it itself when it calls `tool_search`.

**Recommended pipeline.** Embed every tool once with the first-stage model, take the top
50-100 for a request, score those with this model, and rank by
`z(rerank) + 0.25 * z(first stage)`. Reranking fewer candidates is faster and a little
worse (top 30: about 0.4 points lower than top 50 in our runs).

## How it was made

- **Fine-tuned** from `Qwen/Qwen3-Reranker-0.6B` for 300 steps of 48 groups (one relevant
  tool + seven hard negatives; listwise cross-entropy plus a pointwise term), learning rate
  6e-6, on one NVIDIA B300.
- **Weight-averaged** 50/50 with the released base model. On ToolRet the average scores
  55.98 on its own, the fine-tuned weights 51.68 and the base 51.85. The ratio was fixed at
  0.5 beforehand.
- **Layout augmentation.** Tool libraries write the same tool in many layouts. During
  training every tool definition was re-rendered in a random one (renamed and reordered
  keys, wrapped or unwrapped, JSON / YAML-like / Python signature), so that the model
  cannot learn "tools that look like my training set are the relevant ones". Without this,
  fine-tuning raised scores on the training sources and lowered them everywhere else.
- **Hard negatives** are the first-stage model's own top results for each training question.

### Training data

| Source | Rows | Notes |
|---|---|---|
| ToolRet-Training-20w (ToolBench, APIGen, ToolACE) | 188,404 used of 208,826 | **18,422 rows removed** because the question copies or nearly copies a ToolRet test question (normalised exact match, word-set Jaccard >= 0.60, or embedding cosine >= 0.92); 2,000 more held out |
| Toucan-1.5M (real MCP-server tools) | 64,510 | Apache-2.0; same near-copy filter |
| Glaive function-calling v2 | 17,032 | Apache-2.0; same near-copy filter |

The retrieval instructions for the Toucan and Glaive rows were written by an LLM in
ToolRet's style. No ToolRet test question or test label was used for training. Only
14,400 groups were sampled from this pool, so the model has seen a small part of it.

## Limits

- Trained on English. Korean requests over English tools were tested on one set (84.6 nDCG@10
  against 86.1 in English with the same tool texts); other languages were not evaluated.
- Evaluated on ToolRet and on one validation set of our own. Both are built from research
  and synthetic datasets; neither is the traffic of a live agent.
- A cross-encoder runs once per candidate. Budget for 50-100 forward passes per search
  (or fewer, at some cost in quality); the first-stage embedder decides what it gets to see.
- When a library holds several near-identical tools from different providers, nothing in a
  request says which one is "right"; rankings among those are arbitrary.
- ToolRet's datasets state no licence. The training set above is public on the Hugging Face
  Hub, but we make no claim about the terms of the underlying source datasets.

## Licence and credits

Weights: Apache-2.0, as the base model `Qwen/Qwen3-Reranker-0.6B`. The recommended
first-stage model `nvidia/Nemotron-3-Embed-1B-BF16` is NVIDIA's, under the OpenMDW-1.1
licence, and is not included here.

ToolRet: Shi et al., "Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for
Large Language Models", ACL 2025 Findings (arXiv 2503.01763). Toucan-1.5M: Agent-Ark.
Glaive function-calling v2: Glaive AI.

```bibtex
@misc{opentoolsearch2026,
  title  = {OpenToolSearch: small open models for tool retrieval},
  author = {AI3 Inc.},
  year   = {2026},
  url    = {https://huggingface.co/ai3inc/opentoolsearch-rerank-0.6b}
}
```
