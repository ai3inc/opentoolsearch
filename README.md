<p align="center">
  <img src="assets/banner.jpg" alt="OpenToolSearch - thousands of tools in the library, only the right few in the prompt" width="100%">
</p>

<p align="center">
  <a href="README.ko.md"><b>한국어</b></a> ·
  <a href="https://huggingface.co/ai3inc/opentoolsearch-embed-0.6b"><b>Embedder</b></a> ·
  <a href="https://huggingface.co/ai3inc/opentoolsearch-rerank-0.6b"><b>Reranker</b></a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="BENCHMARKS.md">Full benchmark tables</a> ·
  <a href="results/LEDGER.md">Every run we made</a>
</p>

# OpenToolSearch

**Open-source tool search for agents.** Give an agent a library of hundreds or thousands of
tools - function declarations, MCP servers, skills - and let it load only the few a request needs.

Model APIs now ship this as a hosted feature. OpenToolSearch is the same idea with open weights
that you run yourself, next to any model that can call a function: a 0.6B embedder finds the
candidates and a 0.6B reranker orders them. Both models are ours and Apache-2.0; either stage can
be swapped for another.

## Why OpenToolSearch

- **Fast and cheap.** 0.10 s per search and $0.19 per 1,000 on one GPU (37,000 searches an hour):
  25-120 times faster than the hosted tool search of Google, OpenAI and Anthropic, and 90-330 times
  cheaper than the flagship ones - the cheapest hosted ones (gpt-6-luna, Haiku 5.5) cost 2-3 times as much.
- **Made for big libraries that keep changing.** Add, edit or remove an MCP tool or a skill and it is searchable 16 ms later - no re-embedding of the
  rest, no vector database ([below](#built-for-libraries-that-are-big-and-keep-changing)).
- **Open weights, your hardware.** 1.2B parameters in all (0.6B embedder, 0.6B reranker), Apache-2.0.
  Your tool library never leaves your machines.
- **Competitive on quality.** On the public benchmark it beats OpenAI's and Claude's hosted tool
  search and Gemini 3.6-flash, ties Gemini 3.8-flash and trails gemini-4-hydrogen by 2.7 points. On
  our own set and in Korean it is level with or above every hosted tool search (our 200 requests
  cannot separate 3 points). Tool-DE, an 8B open pair, still scores higher, mostly through its 4B
  reranker: Tool-Rank-4B behind our embedder gives 71.8 / 90.8 / 90.8 at 2.6 times the time. Behind a
  bigger first stage (callbus-emb-8b) our reranker is 1.0 / 0.4 / 0.8 points better than with ours.
- **English first, Korean tested.** Trained and benchmarked mainly in English, with a small share of
  machine-translated Korean requests in the embedder's training. Korean requests over English tool
  descriptions cost it 0.8 points (hosted Gemini tool search loses 4 to 7); other languages were not
  evaluated.
- Every run we made, the failed ones included, is [on the record](results/LEDGER.md).

## Built for libraries that are big and keep changing

MCP servers and skills pile up - hundreds, thousands - and new ones appear every day, edited ones every hour. Putting them all in the prompt is not an
option (the 1,000 tools of our test library are 118,000 tokens; 59 skills are 16,600), and the usual fixes get awkward when the library moves.

OpenToolSearch keeps the library in a small index inside your own process, and keeping that index current is nearly free:

| On a 1,000-tool library, one B300 | Time | Texts embedded |
|---|---|---|
| Add a new tool or skill | 16 ms | 1 |
| Change a description | 17 ms | 1 |
| Remove one | under 1 ms | 0 |
| Re-scan the library after 35 edits, 5 additions, 5 removals (`sync`) | 143 ms | 40 |
| Save / load the index (4.6 MB) | 14 ms / 4 ms | 0 |
| Build it from nothing | 0.9 s | 1,000 |

```python
import os
from opentoolsearch import ToolIndex

index = ToolIndex.load("tool_index/") if os.path.exists("tool_index/meta.json") else ToolIndex()
index.sync(all_tools_and_skills_now)            # embeds only what is new or changed, drops what is gone
index.save("tool_index/")
hits = index.search(recent_conversation, k=5)   # 0.10 s and $0.19 per 1,000 searches; only these 5 definitions go to the model
```

No vector database and no nightly re-embedding job: a skill a user wrote a minute ago is searchable on the next call. The embedder reads each tool
on its own, so adding or editing one never changes the others - the index holds the same vectors as a full rebuild (top 5 identical on 200 of
200 requests; [measured](BENCHMARKS.md#a-library-that-changes-what-keeping-the-index-current-costs)). `examples/skill_search.py` does this for a folder of `SKILL.md` files.

| | Hosted tool search (Google, OpenAI, Anthropic) | Hosted decision model (Jev, Gemini Decision) | **OpenToolSearch** |
|---|---|---|---|
| Libraries of 1,000+ tools | yes, by sending the library with every call (Gemini, OpenAI: 30,000-67,000 input tokens per search) | no - it ranks candidates it is given; on its own it reads the whole pool only up to about 200 tools (Gemini Decision: at most 255 options) | **yes** - the index is the search |
| When a tool is added | the library the vendor receives changes | nothing to index, but something else must find the candidates | **one vector, 16 ms** |
| Where your tool descriptions go | to the vendor, every call | to the API, every call | **stay in your process** |
| Cost per 1,000 searches | $0.35-105 | $0.07-0.33, on top of a first stage | **$0.19** |
| Time per search | 2.5-12 s | 0.5-0.7 s | **0.10 s** |

**Where it is strongest:** agent platforms and MCP marketplaces where users add servers; per-user or per-team skill libraries that change daily; internal tool
catalogues behind permissions (`allowed=[...]`, descriptions never leave your machines); anything that searches often enough for $0.19 per 1,000 to matter
(37,000 searches an hour on one GPU). **Where it is not the tool:** a catalogue of 200 tools or fewer (give a decision model the whole list, or just the list), and
the last choice among near-identical tools - for that, put a decision model on OpenToolSearch's candidates (the best quality we measured: 76.8-77.3 on the public
benchmark with Jev, [numbers](BENCHMARKS.md#same-first-stage-fair-fight)). A decision model like Jev cannot search a library or keep up with one; OpenToolSearch
does exactly that, so they work together. Skills were checked only on a small private set ([sanity check](BENCHMARKS.md#a-library-that-changes-what-keeping-the-index-current-costs));
the 1,000-tool numbers above are the measured ones, and for a conversation, pass the last few turns as the query, not just the last line.

## Benchmarks

<p align="center"><img src="assets/summary.png" alt="OpenToolSearch against hosted tool search and open retrievers: quality on three tests, cost and latency" width="100%"></p>

<!-- table:summary -->
| System | Kind | ToolRet (public) | OTS-MCP-1k (ours) | …in Korean | Cost / 1,000 searches | Median latency |
|---|---|---|---|---|---|---|
| Gemini tool search · gemini-4-hydrogen | hosted tool search | 73.5 | 88.3 | 83.8 | $30 (list $51) | 10.0 s |
| Tool-DE (4B + 4B) | open weights | 72.0 | 91.4 | 89.5 | $0.29 | 0.15 s |
| **OpenToolSearch** (callbus-emb-8b 7.6B + Rerank-0.6B) | open weights; our reranker, callbus's embedder | **71.8** | **89.4** | **89.0** | $0.20 | 0.10 s |
| **OpenToolSearch** (Embed-0.6B + Tool-Rank-4B = 4.6B) | open weights; our embedder, Tool-DE's reranker | **71.8** | **90.8** | **90.8** | $0.49 | 0.25 s |
| **OpenToolSearch** (Embed-0.6B + Rerank-0.6B = 1.2B) | open weights, this repo | **70.8** | **89.0** | **88.2** | **$0.19** | **0.10 s** |
| Gemini tool search · gemini-3.8-flash | hosted tool search | 70.6 | 85.6 | 79.0 | $17 (list $28) | 9.95 s |
| callbus-emb-8b (8B, embedding only) | open weights | 68.6 | 86.6 | 84.2 | $0.04 | 0.02 s |
| OpenAI tool search · gpt-6.2-sol | hosted tool search | 66.6 | 82.4 | 81.1 | $34 | 3.43 s |
| OpenAI tool search · gpt-6-luna | hosted tool search | 65.5 | 88.2 | 85.5 | $0.35 (list $1.74) | 3.15 s |
| Claude tool search · Opus 5.5 max effort | hosted tool search | 60.1 | 61.8 | 58.9 | $62 | 11.8 s |
| Claude tool search · Haiku 5.5 | hosted tool search | 53.3 | 57.1 | 56.5 | $0.52 | 2.53 s |
<!-- /table:summary -->

nDCG@10. **ToolRet (public)**: 769 questions, each searched within its own source dataset, the
benchmark's labels. **OTS-MCP-1k**: 1,000 MCP tools and 200 requests, built by us - weigh it below
the public one. **In Korean**: the same 200 requests, the tools stay in English. Hosted costs are
the lower of list price and cache-aware, from reported token usage (Nov 2026); open models are GPU
time on one B300. Their latency is the time per search at full throughput (waves of 8 requests); a single request takes a median 66 ms.

- **Quality is competitive, not first.** OpenToolSearch (70.8 public) sits in the middle of the
  hosted tool search of the three vendors; on our own set (89.0) it is level with the best of them
  (88.2-88.3) and in Korean (88.2) above all of them (best: 85.5). Tool-DE (a 4B embedder plus a 4B
  reranker) leads the open models, and the hosted decision models below lead everything on the
  public benchmark.
- **A 4B second stage is the best open-weights quality per size we measured.** Tool-Rank-4B (Tool-DE's
  reranker, Apache-2.0) behind our embedder scores 71.8 / 90.8 / 90.8 with 4.6B parameters, at 0.25 s and
  $0.49 per 1,000: the highest Korean score of any system in the table and within 0.2 / 0.6 points of
  Tool-DE on the other two. It is not first overall - Tool-DE leads on our English set (91.4), and
  Tool-Embed-4B with both rerankers reaches 72.8 on the public benchmark; of the gaps to our default only the Korean
  one is established ([paired bootstrap](BENCHMARKS.md#swapping-the-second-stage-what-makes-tool-de-better)).
- **Speed and cost are where it wins.** 0.10 s against 2.5-12 s for hosted tool search; $0.19 per
  1,000 against $0.35 to $105 (gpt-6-luna and Haiku 5.5 are the cheap end). Time per search is
  measured at full throughput; a single request takes a median 66 ms.
- **Not in the table: Jev and Gemini Decision.** Hosted decision models rank the candidates a search
  stage hands them and cannot search a library themselves; given our first stage's candidates
  they score 77-78 on the public benchmark (candidates from the earlier Nemotron first stage). [How they compare](BENCHMARKS.md#same-first-stage-fair-fight),
  and every setting and failed run, are in [BENCHMARKS.md](BENCHMARKS.md).

## Where it fits

| Your situation | What to use |
|---|---|
| A small catalogue, up to about 200 tools | A hosted decision model (Jev, Gemini Decision) on its own: given the whole pool it scored 85.7-86.3 on the small public subsets, against 76.7 for OpenToolSearch. Gemini Decision takes at most 255 options per question; we did not test either with 1,000. |
| Hundreds to thousands of tools, tool descriptions that must stay in-house, no API dependency, or 0.1 s answers | OpenToolSearch on your own GPU: open weights, no per-search fee. |
| A little more open-weights quality, with 16 GB to spare for it | Our reranker behind a bigger first stage: callbus-emb-8b gives 71.8 / 89.4 / 89.0 against 70.8 / 89.0 / 88.2 with the default embedder, at the same 0.10 s ([measured](BENCHMARKS.md#swapping-the-first-stage), `scripts/v8_first_stage_rerank.py`). |
| Tool-DE-level open-weights quality, 4.6B parameters instead of 8B | Our embedder with Tool-Rank-4B (Tool-DE's reranker) as the second stage: 71.8 / 90.8 / 90.8 against Tool-DE's 72.0 / 91.4 / 89.5, at 0.25 s and $0.49 per 1,000 instead of 0.10 s and $0.19 ([measured](BENCHMARKS.md#swapping-the-second-stage-what-makes-tool-de-better)). |
| The best quality we measured | A first stage (OpenToolSearch's embedder or any retriever) and a decision model as the second: 76.8-77.3 public and 89.1-89.3 on our set at $0.07-0.16 per 1,000 with Jev (candidates from the earlier Nemotron first stage), against 70.8 and 89.0 at $0.19 with our reranker - the gap is 6 points on the public benchmark and closed on our set. A hosted call (about 0.5 s) that sees your tool descriptions. |
| You already call a hosted model and want the least code | Its built-in tool search: one parameter, but seconds per search and $0.35-105 per 1,000. |

## Quick start

```bash
pip install git+https://github.com/ai3inc/opentoolsearch
```

Weights, downloaded on first use: [`ai3inc/opentoolsearch-embed-0.6b`](https://huggingface.co/ai3inc/opentoolsearch-embed-0.6b)
(the embedder) and [`ai3inc/opentoolsearch-rerank-0.6b`](https://huggingface.co/ai3inc/opentoolsearch-rerank-0.6b)
(the reranker), both Apache-2.0.

```python
from opentoolsearch import ToolIndex

index = ToolIndex()                    # both models; ToolIndex(reranker=None) for embedding search only
index.add(tools)                       # list of tool / function declarations (dicts)
# defaults: rerank_top=25, fuse=0.5  (depth 25 ≈ depth 50 in quality, twice the throughput)

hits = index.search("refund the customer's last order", k=5)
for h in hits:
    print(h["name"], round(h["score"], 3))
```

### Use the stages on their own

The two models are separate Hugging Face repositories (Apache-2.0); neither needs the other, and `ToolIndex` takes
any embedder and any reranker in the Qwen3-Reranker format.

```python
# our embedder alone - embedding search, no reranker
index = ToolIndex(reranker=None)

# our embedder, a bigger reranker: Tool-DE's Tool-Rank-4B, ranking by the reranker alone (fuse=0.0).
# Measured through this library: 90.8 / 91.1 on our set (English / Korean), 0.26 s a search
index = ToolIndex(reranker="Lux1997/Tool-Rank-4B", fuse=0.0)

# another embedder in front of our reranker (give the query format that embedder expects)
index = ToolIndex(model="Qwen/Qwen3-Embedding-0.6B",
                  query_prefix="Instruct: Given an agent's request for a tool, retrieve the MCP tool that fulfills it.\nQuery: ")
```

Each model also works without the library: [`opentoolsearch-embed-0.6b`](https://huggingface.co/ai3inc/opentoolsearch-embed-0.6b)
is a plain sentence-transformers model and [`opentoolsearch-rerank-0.6b`](https://huggingface.co/ai3inc/opentoolsearch-rerank-0.6b)
a Qwen3-Reranker-format cross-encoder (usage on their cards). The combinations in the tables
([embedder + Tool-Rank-4B](BENCHMARKS.md#swapping-the-second-stage-what-makes-tool-de-better),
[callbus-emb-8b + our reranker](BENCHMARKS.md#swapping-the-first-stage), ...) are exactly this.

### Deferred tool loading (client-side `tool_search`)

Send the model one function instead of your whole library; load what it asks for.

```python
search_tool = index.search_tool()              # JSON-Schema declaration for `tool_search`
# 1. call the model with tools=[search_tool]
# 2. when it calls tool_search(query=...):
loaded = index.handle(call_arguments)          # -> full declarations of the best matches
# 3. call the model again with tools=[search_tool, *loaded]
```

`examples/deferred_openai.py` is a complete loop against any OpenAI-compatible endpoint.
`index.catalogue()` gives the one-line-per-tool listing if you prefer to show the model
names and short descriptions, and `allowed=[...]` restricts a search to the tools a user
may call.

## More

- [BENCHMARKS.md](BENCHMARKS.md) - every system, condition and table; throughput and cost on rented GPUs; the full ToolRet corpus.
- [DEVELOPMENT.md](DEVELOPMENT.md) - how the model was built and trained, the validation set, how to reproduce.
- [`benchmarks/`](benchmarks/README.md) - raw data and scripts; [`results/LEDGER.md`](results/LEDGER.md) - every test run, with cost.

## Licence

Code: Apache-2.0. Both models: Apache-2.0 - OpenToolSearch Embed is fine-tuned from
Qwen3-Embedding-0.6B and OpenToolSearch Rerank from Qwen3-Reranker-0.6B (Alibaba Cloud, Apache-2.0;
notices in each model repository). ToolRet's data is read from Hugging Face at pinned revisions and
is not redistributed here.
