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

## Highlights

- **Fast and cheap.** With its first-stage embedder: 0.12 s per search and $0.24 per 1,000 on one GPU
  (29,000 searches an hour) - 20-100 times faster than the hosted tool search of Google, OpenAI and
  Anthropic, and 1.4 to 430 times cheaper depending on the model.
- **Open weights, your hardware.** 1.7B parameters in all; your tool library never leaves your machines.
- **Competitive, not best, on quality.** On the public benchmark it beats OpenAI's and Claude's hosted
  tool search and Gemini 3.6-flash, ties Gemini 3.8-flash and trails gemini-4-hydrogen by 3 points.
  Tool-DE, an 8B open pair, scores higher.
- **Works in Korean.** The same 200 requests in Korean cost it 1.5 points.
- **Second among published systems on ToolRet** (58.4 nDCG@10, question + instruction, whole 44,453-tool corpus).

## Benchmarks

![OpenToolSearch against hosted tool search and open retrievers](https://raw.githubusercontent.com/ai3inc/opentoolsearch/main/assets/summary.png)

<!-- table:summary -->
| System | Kind | ToolRet (public) | OTS-MCP-1k (ours) | …in Korean | Cost / 1,000 searches | Median latency |
|---|---|---|---|---|---|---|
| Gemini tool search · gemini-4-hydrogen | hosted tool search | 73.5 | 88.3 | 83.8 | $30 (list $51) | 10.0 s |
| Tool-DE (4B + 4B) | open weights | 72.0 | 91.4 | 89.5 | $0.29 | 0.15 s |
| Gemini tool search · gemini-3.8-flash | hosted tool search | 70.6 | 85.6 | 79.0 | $17 (list $28) | 9.95 s |
| **OpenToolSearch** (1.14B + 0.6B) | open weights, this repo | **70.5** | **86.8** | **84.6** | **$0.24** | **0.12 s** |
| callbus-emb-8b (8B, embedding only) | open weights | 68.6 | 86.6 | 84.2 | $0.04 | 0.02 s |
| OpenAI tool search · gpt-6.2-sol | hosted tool search | 66.6 | 82.4 | 81.1 | $34 | 3.43 s |
| OpenAI tool search · gpt-6-luna | hosted tool search | 65.5 | 88.2 | 85.5 | $0.35 (list $1.74) | 3.15 s |
| Claude tool search · Opus 5.5 max effort | hosted tool search | 60.1 | 61.8 | 58.9 | $62 | 11.8 s |
| Claude tool search · Haiku 5.5 | hosted tool search | 53.3 | 57.1 | 56.5 | $0.52 | 2.53 s |
<!-- /table:summary -->

nDCG@10. **ToolRet (public)**: 769 questions, each searched within its own source dataset, the
benchmark's labels, request only. **OTS-MCP-1k**: 1,000 MCP tools and 200 requests, built by us - weigh it
below the public one. **In Korean**: the same requests, tools in English. Hosted costs are the lower of
list price and cache-aware, from reported token usage (Nov 2026); open models are GPU time on one B300.
Hosted decision models (Jev, Gemini Decision) are not in the table: they rank the candidates a search
stage hands them and cannot search a library themselves.

Every system, setting and failed run: [github.com/ai3inc/opentoolsearch](https://github.com/ai3inc/opentoolsearch/blob/main/BENCHMARKS.md)
and the [ledger](LEDGER.md).

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

**Recommended pipeline.** Embed every tool once with the first-stage model, take the top 25 for a
request (the library default), score those with this model, and rank by
`z(rerank) + 0.5 * z(first stage)`. Reranking 50 candidates is about as good (70.8 against 70.5 on
the public benchmark) at half the speed; 10 is faster still and 3 points lower.

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
- A cross-encoder runs once per candidate. Budget for 25-50 forward passes per search; the
  first-stage embedder decides what it gets to see.
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
