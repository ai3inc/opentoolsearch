# OpenToolSearch

Open-source **tool retrieval** for function-calling agents: keep hundreds or thousands of
tool definitions out of the prompt, and load only the few a request needs.

It is the open counterpart of the server-side tool search that model APIs have started to
ship (Gemini API "Tool Retrieval", Nov 2026): the same flow, but running on your side, with
open weights, for any model that can call a function.

- **`ToolIndex`** - index tool definitions (OpenAI, Anthropic, Gemini or MCP style), search
  them, and expose a drop-in `tool_search` function for deferred loading.
- **Two small models** - an embedder (1.1B) and a cross-encoder reranker (0.6B) that
  together score **58.4 nDCG@10 on ToolRet** (question + instruction, all 44,453 tools in one
  pool) - second among the published figures we could find, with models about a fifth of
  the size of the first. See [Results](#results).
- **A reproducible harness** - the evaluation, the training code, the de-duplication of
  the training set against the test questions, and a ledger of every test run we made.

## Quick start

```bash
pip install git+https://github.com/ai3inc/opentoolsearch
```

Weights: [`ai3inc/opentoolsearch-rerank-0.6b`](https://huggingface.co/ai3inc/opentoolsearch-rerank-0.6b)
on Hugging Face. Developed by AI3 Inc.

```python
from opentoolsearch import ToolIndex

index = ToolIndex(model="nvidia/Nemotron-3-Embed-1B-BF16",
                  reranker="ai3inc/opentoolsearch-rerank-0.6b")       # reranker is optional
index.add(tools)                       # list of tool / function declarations (dicts)

hits = index.search("refund the customer's last order", k=5)
for h in hits:
    print(h["name"], round(h["score"], 3))
```

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

## Results

ToolRet (ACL 2025): 7,961 questions, 44,453 tools from 35 source datasets. Every question
is searched against **all** tools in one pool. nDCG@10; the headline is the mean of the three
categories (web, code, customized), each the mean of its subsets - the paper's aggregation.

**Question + instruction**

| System | Size | nDCG@10 | Notes |
|---|---|---|---|
| MagicSelector | LLM query decomposition + 4B embedder + 4B reranker | 59.90 | as reported, arXiv 2607.17751 |
| **opentoolsearch, track B** | 1.14B embedder + 0.6B reranker | **58.48** | track A data + the *training* split of Gorilla APIBench; top 50 |
| **opentoolsearch, track A** | 1.14B embedder (as released) + 0.6B reranker | **58.37** | clean ToolRet-train + two outside sources; reranking the top 100 (58.15 with the top 50) |
| Tool-DE (Tool-Embed-4B + Tool-Rank-4B) | 4B + 4B, LLM-expanded tool docs | 57.01 | as listed in the MagicSelector paper (56.44 in its own, arXiv 2510.22670) |
| Open-weights pipeline, nothing trained | 1.5B embedder + BM25 + 1B reranker | 55.33 | as reported, Savirinc/toolret-open-weights |
| callbus-emb-8b, embedding only | 8B | 54.36 (FP8: 54.53) | as reported on its model card |
| Nemotron-3-Embed-1B, as released | 1.14B | 52.42 | our run; the first stage of both tracks |

So: **not the top of the table.** opentoolsearch is 1.5 points (track A) below the best reported
system, and above every other system we found, with models roughly a fifth of the size and
no LLM call at query time. Other systems' numbers are their authors'; we did not re-run them.

Two more views of the same track A run:

| | opentoolsearch track A | Best other figure we found |
|---|---|---|
| Question alone (no instruction), category mean, top 50 | **42.42** | 41.96 (open-weights pipeline) |
| Question + instruction, all questions pooled, top 100 | **62.47** | 59.36 (callbus-emb-8b + Qwen3-Reranker-8B) |

Track A with instruction, top 100: Recall@10 68.00, Completeness@10 56.26.

**How to read these numbers**

- The fusion weight is the only thing tuned against test questions, and it is *cross-fitted*:
  the questions are split in two halves, the weight is picked on one half and scored on the
  other, both ways. With the top 100, a fixed w = 0.25 gives 58.37 and w = 0.5 gives 58.19.
- Model choices (which base, whether to fine-tune the embedder, the augmentation) were made
  while looking at test scores. Every such run is listed in
  [`results/LEDGER.md`](results/LEDGER.md), including the ones that lost.
- ToolRet's instructions were written by a model that was shown the answer tools (the
  paper's "target-aware strategy"), so the with-instruction setting is easier than a plain
  request. The no-instruction row is the closer one to production traffic.
- Many subsets are small (11 to 50 questions). Differences under about half a point between
  runs are within what changes from one training run to the next.
- Track B uses in-domain training data for 3 of the 35 subsets (the Gorilla ones); read it
  as a supervised result for those, not as generalisation.
- The 18,422 training rows we removed are near-copies of test questions. Systems trained on
  ToolRet-train without such a filter have seen (close copies of) some test questions; we
  cannot tell from the outside how much that helps them.

## Round 2: a validation set of our own

ToolRet labels one tool per need even when the pool holds interchangeable copies, and its
instructions were written with the answer in view. Round 2 therefore chose on a separate set,
**OTS-MCP dev** (738 questions about 67 MCP servers held out of training, 24,188-tool pool,
instructions written from the request alone, strict and equivalence-credited scoring), and
touched the ToolRet test questions once.

- Removing training "negatives" that are the same tool as a positive (5.6% of ToolRet-train's,
  21.9% of Toucan's, 49.6% of Glaive's) and showing half the questions without an instruction
  gave about +1 nDCG@10 on OTS-MCP dev and -0.4 on ToolRet (57.96) - both inside noise. The
  published weights stay the round-1 ones.
- On ToolRet, 47.6% of the questions where the top result is "wrong" have a top result an LLM
  judge found interchangeable with a labelled tool; crediting them, top-1 accuracy goes from
  57.2% to 77.6%.

Scripts: `scripts/v2_build_dev.py`, `v2_equiv_candidates.py`, `judge_equiv.py`,
`v2_make_train.py`, `v2_eval_dev.py`, `v2_ablate.sh`, `v2_final.sh`. Numbers: `results/LEDGER.md`.

## How it was built

**Two stages, both small.**

1. *First stage* - `nvidia/Nemotron-3-Embed-1B-BF16` (1.14B parameters), used as released:
   every tool is embedded once, a request is embedded and the top 100 (or 50) tools are taken.
2. *Second stage* - `opentoolsearch-rerank-0.6b`, a cross-encoder fine-tuned from
   `Qwen/Qwen3-Reranker-0.6B`. It reads the request (and the instruction, when there is
   one) next to each candidate's definition.
3. *Fusion* - the final score is `z(rerank) + w * z(first stage)`, both z-normalised over
   the candidates of the request.

**What we found on the way (all numbers in `results/LEDGER.md`).**

- *Fine-tuning a retriever on ToolRet-train teaches it a layout, not just the task.* The
  training set covers three sources (ToolBench, APIGen, ToolACE). After fine-tuning, scores
  on those three rise and scores on the other 32 subsets fall, because the model starts to
  prefer tools written in the three layouts it was trained on.
- *Layout augmentation fixes that.* During training every tool definition is re-rendered in
  a random layout (renamed and reordered keys, wrapped or unwrapped, JSON / YAML-like /
  Python-signature). Without it the embedder dropped from 52.4 to 51.0 in 80 steps; with it
  the 32 unseen-source subsets held (51.3 -> 51.1) while the three seen sources gained
  (64.9 -> 68.6). See `opentoolsearch/augment.py`.
- *A weight average beats both of its halves.* Averaging the fine-tuned reranker's weights
  50/50 with the released reranker's gave 55.98 nDCG@10 on its own, against 51.68 for the
  fine-tuned one and 51.85 for the released one. The ratio was fixed at 0.5 beforehand and
  not tuned.
- *The two stages disagree usefully.* The reranker alone (55.98) and the embedder alone
  (52.42) fuse to 58.2.
- *The embedder is best left alone here.* Fine-tuned embedders gave the reranker a worse
  candidate list than the released one (57.3 vs 58.2 after reranking), so the released
  embedder is the first stage.

**Training data for the reranker.** 14,400 groups (300 steps x 48) of one relevant tool and
seven hard negatives, drawn from:

- ToolRet-train with every row removed whose question copies or nearly copies a ToolRet
  test question: normalised exact match, word-set Jaccard >= 0.60, or embedding cosine
  >= 0.92 against the nearest test questions. **18,422 of 208,826 rows were removed**
  (1,866 distinct test questions had a copy in the training set); 2,000 more rows are held
  out as a dev set. `scripts/prep_train.py`.
- 81,542 rows from two sources that are not among ToolRet's 35: real MCP-server tools from
  Toucan-1.5M and function-calling dialogues from Glaive v2 (both Apache-2.0), filtered with
  the same rule. Their retrieval instructions were written by an LLM in ToolRet's style.
- Hard negatives are the first stage's own top results for each training question.

## Reproduce

```bash
pip install -e ".[train]"
python -m opentoolsearch.evaluate --model nvidia/Nemotron-3-Embed-1B-BF16 --inst --template concat \
    --q_prefix "query: " --doc_prefix "passage: " --out results/first_stage
python -m opentoolsearch.rerank eval --model ai3inc/opentoolsearch-rerank-0.6b --run results/first_stage/run_inst_concat.json \
    --inst --topn 100 --out results/reranked
python scripts/fuse.py results/first_stage/run_inst_concat.json results/reranked/run_inst.json 0 0.25 0.5 1
```

Training data and steps: `scripts/prep_train.py` (clean ToolRet-train), `scripts/prep_external.py`
+ `scripts/gen_instructions.py` (Toucan / Glaive rows), `scripts/mine_negs.py`,
`python -m opentoolsearch.rerank train ...`. Every test evaluation we ran is in
[`results/LEDGER.md`](results/LEDGER.md).

## Status

- `opentoolsearch-rerank-0.6b` is published on Hugging Face (Apache-2.0). It is a weight average
  of `Qwen/Qwen3-Reranker-0.6B` and our fine-tune of it. The first stage,
  `nvidia/Nemotron-3-Embed-1B-BF16`, is NVIDIA's model under the OpenMDW-1.1 licence and is
  used unchanged.
- The track B and round-2 weights are experiments and are not published.
- Submitted to the ToolRet leaderboard (pull request on the Space; not merged at the time of
  writing).
- `scripts/push.sh`, `watch_eval.sh`, `baselines.sh`, `v2_ablate.sh`, `v2_final.sh` are the
  helpers we ran on our own GPU pod; paths in them are ours.

## Licence

Code: Apache-2.0. Models keep their base licences (see the model cards). ToolRet's data is
read from Hugging Face at pinned revisions and is not redistributed here.
