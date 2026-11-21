# Evaluation ledger

Every evaluation that touched the ToolRet test questions, in the order it was run
(2026-11-20, one NVIDIA B300). Nothing is left out: runs that failed,
runs with a wrong input format and runs that made things worse are all here, because
choices below were made with these numbers in view.

Setting for every row unless noted: question + instruction, the whole 44,453-tool
corpus as one pool, nDCG@10. **Avg** = mean of the three categories, each the mean
of its subsets (the ToolRet paper's aggregation). **unseen** = same aggregation over
the 32 subsets that ToolRet-train does not draw on; **seen** = mean of the 3 that it
does (ToolBench, APIGen, ToolACE).

| # | System | Avg | web | code | cust | unseen | seen | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | BM25 (bm25s, stemmed) | 38.41 | 27.90 | 41.74 | 45.58 | 36.92 | 51.67 | paper's BM25s: 36.46 (tokeniser differs) |
| 2 | Qwen3-Embedding-0.6B, as released | 42.57 | 36.62 | 48.70 | 42.40 | 41.05 | 61.06 | published 43.13 (Tool-DE paper) |
| 3 | harrier-oss-v1-0.6b, as released | 45.49 | 41.71 | 47.88 | 46.87 | 44.27 | 61.16 | |
| 4 | Nemotron-3.5-Embed-1B, wrong input format | 9.16 | | | | | | invalid run (prefix/pooling not applied) |
| 5 | embeddinggemma-2, generic instruct format | 36.84 | 37.17 | 31.95 | 41.40 | 35.69 | 55.55 | not its native prompt; not pursued |
| 6 | inf-retriever-v1-1.5b, GS1-0.8B | - | | | | | | failed to load in this harness |
| 7 | harrier h1 (ToolRet-train clean), step 100 | 48.25 | 41.73 | 52.50 | 50.51 | 46.41 | 71.09 | |
| 8 | harrier h1, step 200 | 47.90 | 42.05 | 51.92 | 49.74 | 45.96 | 73.11 | more steps: seen up, unseen down |
| 9 | 0.5 x #3 + 0.5 x #7 (weight average) | 48.98 | 43.52 | 52.22 | 51.20 | 47.38 | 69.08 | |
| 10 | #9, query = instruction only | 46.54 | 37.68 | 52.26 | 49.69 | 44.96 | 62.98 | |
| 11 | #9, instruction repeated in query | 49.29 | 42.92 | 53.16 | 51.79 | 47.64 | 69.23 | within noise of #9 |
| 12 | harrier h2 (+ external data), step 100 | 47.62 | 40.33 | 52.81 | 49.72 | 45.81 | 69.25 | |
| 13 | harrier h2, step 200 | 47.73 | 40.40 | 53.11 | 49.70 | 45.92 | 69.40 | |
| 14 | Nemotron-3-Embed-1B, as released | 52.42 | 46.71 | 54.44 | 56.12 | 51.29 | 64.85 | |
| 15 | Nemotron n1 (ToolRet-train clean + external), step 40 | 52.24 | 45.62 | 53.21 | 57.90 | 50.80 | 68.68 | |
| 16 | Nemotron n1, step 80 | 51.01 | 44.25 | 52.11 | 56.66 | 49.55 | 67.51 | |
| 17 | Nemotron n1, step 120 | 51.06 | 44.24 | 52.17 | 56.77 | 49.61 | 67.45 | |
| 18 | #14 top-50 reranked by Qwen3-Reranker-0.6B as released | 51.85 | 44.40 | 57.25 | 53.90 | 50.31 | 68.99 | reranker score alone |
| 19 | #18 fused with #14, w = 0.5 | 55.56 | 48.90 | 57.86 | 59.91 | 54.24 | 69.97 | z(rerank) + w z(first stage) |
| 20 | #18 fused with #14, w = 1.0 | 55.29 | 48.93 | 57.35 | 59.58 | 54.06 | 68.62 | |
| 21 | #18 fused with #14, w = 2.0 | 54.62 | 48.54 | 56.61 | 58.71 | 53.45 | 67.31 | |
| 22 | Nemotron n2 = n1 recipe + layout augmentation, step 40 | 52.44 | 45.44 | 53.96 | 57.92 | 51.01 | 68.42 | |
| 23 | Nemotron n2, step 80 | 52.52 | 45.56 | 53.73 | 58.28 | 51.08 | 68.61 | compare #16 (51.01) without augmentation |
| 24 | #14 top-50 reranked by r2 step 100 (fine-tuned reranker), alone | 51.32 | 46.06 | 56.32 | 51.58 | 49.68 | 72.33 | |
| 25 | #24 fused with #14, w = 0.5 / 1.0 | 57.18 / 57.22 | 52.18 / 51.83 | 59.47 / 59.26 | 59.90 / 60.57 | 55.71 / 55.83 | 75.80 / 74.16 | |
| 26 | #25 (w = 1) + BM25 z-score x 0.25 / 0.5 | 57.46 / 57.19 | | | | | | not adopted |
| 27 | #25 (w = 1) + released-reranker z-score x 0.5 / 1.0 | 57.59 / 58.03 | | | | | | two reranker passes; not adopted |
| 28 | #25 (w = 1) reranking only the top 30 / top 20 | 56.84 / 56.55 | | | | | | depth matters |
| 29 | r2 step 300, alone | 51.68 | 47.48 | 54.92 | 52.64 | 49.89 | 76.19 | |
| 30 | #29 fused with #14, w = 0.5 / 1.0 | 57.70 / 57.64 | 53.07 / 52.88 | 59.24 / 59.00 | 60.79 / 61.04 | 56.12 / 56.21 | 78.37 / 75.75 | |
| 31 | #30 (w = 1) + released-reranker z-score x 1.0 | 58.01 | 52.76 | 59.71 | 61.56 | 56.53 | 76.44 | not adopted |
| 32 | **opentoolsearch-rerank-0.6b** = 0.5 x released + 0.5 x r2 step 300, alone | 55.98 | 49.78 | 60.35 | 57.80 | 54.33 | 76.06 | ratio fixed at 0.5 beforehand |
| 33 | #32 fused with #14, w = 0.25 / 0.5 / 1.0 | 58.29 / 58.18 / 56.98 | 51.75 / 52.22 / 51.23 | 61.63 / 60.67 / 59.06 | 61.51 / 61.66 / 60.65 | 56.75 / 56.77 / 55.67 | 76.49 / 74.78 / 72.21 | |
| 34 | **Track A, cross-fitted w** (w chosen on the other half of the questions) | **58.15** | 51.76 | 61.21 | 61.47 | 56.65 | 75.69 | Recall@10 67.60, Comp@10 55.82, pooled 61.57 |
| 35 | #32 on the fine-tuned first stage n2 (#23), w = 0 / 0.25 / 0.5 / 1; cross-fitted | 54.92 / 57.28 / 57.34 / 56.48; 57.34 | | | | | | released embedder is the better first stage |
| 36 | #33 + n2 first-stage z-score (0.25, 0.25) | 57.70 | | | | | | not adopted |
| 37 | #33 (w = 0.4) + r2-step-300 z-score x 0.3 | 58.63 | 52.81 | 61.31 | 61.78 | 57.07 | 77.86 | two reranker passes; not adopted |
| 38 | Track B embedder n3 (+ Gorilla APIBench train split), step 120 | 54.05 | 45.68 | 57.60 | 58.88 | 52.60 | 68.95 | embedding only |
| 39 | Track B reranker (0.5 x released + 0.5 x r3) on #38, w = 0 / 0.25 / 0.5 / 1 | 55.73 / 58.33 / 58.48 / 57.89 | | | | | | |
| 40 | **Track B, cross-fitted w** | **58.48** | 50.23 | 63.64 | 61.59 | 56.80 | 77.17 | Recall@10 67.74, Comp@10 56.38, pooled 62.40 |
| 41 | Question alone (no instruction): Nemotron-3-Embed-1B as released | 34.70 | 36.83 | 26.33 | 40.93 | 33.56 | 55.02 | |
| 42 | Question alone: #32 on #41, w = 0 / 0.25 / 0.5 / 1 | 40.58 / 42.42 / 42.16 / 40.69 | | | | | | |
| 43 | **Track A, question alone, cross-fitted w** | **42.42** | 42.28 | 35.06 | 49.91 | 40.77 | 68.57 | Recall@10 49.82, Comp@10 39.00, pooled 50.28 |

Not test runs, for completeness: fusion weight on 1,000 held-out ToolRet-train questions
(`scripts/dev_weight.py`) preferred w = 0 (reranker alone 74.19 vs first stage 54.12). That
dev set is in-domain for the reranker and says nothing about the other 32 subsets, so it was
not used; the weight is cross-fitted on the test questions instead (rows 34, 40, 43).

## What was chosen with test numbers in view

- Base embedder (Nemotron-3-Embed-1B over harrier-0.6b / Qwen3-0.6B): by rows 2, 3, 14.
- Keeping the embedder as released, and not using fine-tuned embedders as first stage: rows 15-17, 22-23, 35.
- Layout augmentation: introduced after rows 7-17 showed the seen/unseen split; kept after rows 22-23.
- Reranker training length (300 steps), negatives without the similarity filter, 0.5 weight
  average: one run each; no alternatives were tried and discarded except the first reranker
  run (r1, filtered negatives), which was stopped at step 80 without being evaluated.
- Not adopted although higher on test: rows 26, 27, 31, 37 (extra passes or signals).
- Fusion weight: cross-fitted (never chosen on the questions it is scored on).

Added after the list above was written:

| # | System | Avg | web | code | cust | unseen | seen | Note |
|---|---|---|---|---|---|---|---|---|
| 44 | #32 reranking the top 100 of #14, w = 0 / 0.25 / 0.5 / 1 | 55.39 / 58.37 / 58.19 / 57.17 | | | | | | |
| 45 | **Track A, top 100, cross-fitted w** | **58.37** | 52.01 | 61.09 | 62.01 | 56.79 | 77.27 | Recall@10 68.00, Comp@10 56.26, pooled 62.47 |

Depth 100 is the benchmark's own retrieval depth; the top-50 figure (row 34, 58.15) is the
cheaper setting and is reported next to it.

# Round 2 (2026-11-20/21): a validation set of our own, and training data that fixes what round 1 found

Round 1 chose things while looking at ToolRet test scores. Round 2 does its choosing on a
separate validation set and touches the ToolRet test questions once, for the selected model.

## OTS-MCP dev

- 738 questions about 67 MCP servers (of 369 seen in Toucan-1.5M) that are held out by a hash
  of the server name; every Toucan training row touching one of those servers was removed
  (11,997 rows). Questions come from Toucan shards round 1 never read, at most 12 per server set.
- Search pool: 24,188 tools = all 4,188 Toucan tools (held-out and not) + 20,000 tools from the
  training pool as distractors.
- Two settings: the request alone; the request plus an instruction written by an LLM that saw
  only the request (not the answer tools - unlike ToolRet's instructions).
- Two scorings: *strict* (only the labelled tool counts) and *equivalence-credited* (a tool that
  an LLM judged interchangeable with a labelled tool fills that tool's slot; a slot counts once).
  2,235 candidate pairs were judged, 1,039 as interchangeable; **all 300 labelled tools have at
  least one interchangeable tool in the pool**. Document-embedding cosine >= 0.90 predicts
  "interchangeable" with 96% precision; that threshold is used below.

nDCG@10 on OTS-MCP dev, reranker score alone (w = 0), strict / equivalence-credited:

| System | request alone | request + instruction | Note |
|---|---|---|---|
| Nemotron-3-Embed-1B alone (first stage) | 32.84 / 53.26 | 37.72 / 61.23 | |
| Qwen3-Reranker-0.6B as released | 39.75 / 66.06 | 41.06 / 69.23 | |
| round-1 model (published v1) | 45.79 / 68.57 | 47.88 / 72.86 | not a fair row: it trained on rows from the held-out servers |
| a1: round-1 recipe, held-out servers removed | 44.80 / 68.00 | 47.20 / 71.60 | |
| a2: a1 + "negatives" that are the same tool as a positive removed | 45.87 / 68.93 | 48.41 / 72.71 | |
| a3: a2 + half the training questions shown without an instruction | 45.21 / 69.02 | 47.35 / 72.90 | **selected** |

Selection rule, fixed before a2 and a3 were run: highest mean of the two equivalence-credited
figures at w = 0.25 (a1 69.65, a2 70.23, a3 70.38). a2 and a3 are within noise of each other.
Fusing with the first stage does not help on this set (w = 0.25 lowers every row by 0.1 to 1.2).

How many training "negatives" were the same tool as a positive (cosine >= 0.90) and were
removed for a2/a3: ToolRet-train 5.6%, Toucan 21.9%, Glaive 49.6%.

## The one ToolRet test evaluation of round 2

The selected model (a3, weight-averaged 50/50 with the released reranker like round 1), on the
same first stage, question + instruction, top 100 reranked:

| System | Avg | web | code | cust | unseen | seen | Note |
|---|---|---|---|---|---|---|---|
| round 1 (published) | 58.37 | 52.01 | 61.09 | 62.01 | 56.79 | 77.27 | row 45 above |
| round 2 (a3), w = 0 / 0.25 / 0.5 / 1 | 55.18 / 57.99 / 58.18 / 57.37 | | | | | | |
| **round 2 (a3), cross-fitted w** | **57.96** | 51.79 | 60.64 | 61.43 | 56.40 | 76.69 | Recall@10 67.66, Comp@10 55.69; top 50: see below |
| round 2 (a3), top 50 reranked, cross-fitted w | 57.80 | 51.48 | 60.76 | 61.15 | 56.27 | 75.91 | round 1 at top 50: 58.15 |
| round 2 (a3), question alone, top 50, cross-fitted w | 42.52 | 42.99 | 34.77 | 49.80 | 40.97 | 67.77 | round 1: 42.42 |

**Reading.** The round-2 data changes (same-tool negatives removed, half the questions without
an instruction, held-out servers removed) moved the ToolRet score by -0.4, which is inside
run-to-run noise, and moved the fairer dev score by about +1. They did not raise the ToolRet
score, as expected: what ToolRet's strict labels reward is picking the labelled copy among
interchangeable tools. The published model stays the round-1 one.

## How much of ToolRet's "error" is a labelling artefact

For the published system's ranking (question + instruction), the top result is a labelled tool
for 4,553 of 7,961 questions (57.2%). For the questions where it is not, an LLM judge was asked
whether the top result could stand in for a labelled tool (`scripts/judge_equiv.py`,
`scripts/toolret_top1_pairs.py`; up to three labelled tools per question; 64 of 5,429 calls
failed and count as "no"). It said yes for 1,622 questions - 47.6% of the "wrong" ones. Crediting
those, top-1 accuracy goes from 57.2% to 77.6%. Highest shares: gta 90%, craft-math-algebra 81%,
mnms 77%, gorilla-huggingface 73%, toolbench 72%, apigen 71%, toolace 69%. One judge
(`deepseek-v4.1-flash`), no human check: treat the size as indicative.

# Head-to-head with hosted services (2026-11-21)

Request only, no instruction. Details and caveats in the main README; raw outputs under `benchmarks/`.

Public: ToolRet per-source (769 questions, 32 subsets with pools of at most 2,000 tools, the
benchmark's labels, macro nDCG@10): Gemini Decision single choice 78.25 · Jev single choice 77.35 ·
Gemini API tool search gemini-4-hydrogen 73.51 · Gemini Decision yes/no 71.05 · Jev yes/no 70.88 ·
**OpenToolSearch 70.26** · Qwen3-Reranker-0.6B as released 69.86 · Gemini API tool search
gemini-3.6-flash 67.02 · Nemotron-3-Embed-1B alone 62.93.

Ours: OTS-MCP-1k (200 requests, 1,000 tools, interchangeable tools credited): Jev single choice 89.08 ·
Gemini tool search 88.29 / 88.21 · **OpenToolSearch 87.93** (retrained without the test servers
87.76) · Qwen3-Reranker-0.6B 87.32 · Jev yes/no 86.13 · Gemini Decision yes/no 85.21 · embedder
alone 78.42.

Median latency / cost per 1,000 searches on the public set: OpenToolSearch 0.16 s / $0.36 of GPU
time · Jev 0.51 s / $0.16 · Gemini Decision 0.68 s / $0.33 · Gemini tool search 6.2-10.0 s / $28-51.
The published weights were chosen before either test existed; nothing was tuned on them.

# Hardware timing and cost, and the missing Gemini Decision row (2026-11-21, later)

- Timing re-run on a fresh B300 (spot): p50 0.216 s, mean 0.233 s - the first run's 0.217 s
  was reproducible. Cost is the same: about $0.42-0.46 of GPU time per 1,000 searches
  (`results/head_to_head_timing.json`, `..._rerun.json`, `..._devices.json`).
- H100: not measured. All eight H100 units have been leased for days, several to pods that are
  idle; a spot request against the dedicated host was refused (`host_id` pinning is only for
  org-owned hosts). Estimate: p50 about 0.28 s, $0.27-0.35 per 1,000 searches, from the B300
  time scaled by the memory-bandwidth ratio (3.35 vs ~8 TB/s) - flagged as an estimate.
- 8 vCPU (cpu_8c_16g), fp32, 8 threads, measured end to end on 5 requests: p50 57.5 s, mean
  52.6 s; indexing the 1,000 tools takes 520 s. Profile: query embedding 0.19 s, reranking
  1.08-1.29 s per candidate pair (`toolsift-rerank` is a cross-encoder, so it runs once per
  candidate). Rented at ₩729/h that is $7.61 per 1,000 searches. CPU is about 265 times slower.
- Gemini Decision single choice on OTS-MCP-1k (the row that was missing): nDCG@10 credited
  91.2, strict 69.8, $0.31 per 1,000 searches, 0.61 s. It leads the set; the figure and table
  above were updated.

# B300 at market prices (2026-11-21)

Cost per 1,000 searches = measured GPU time x the GPU's market price. Nothing else changed.
Measured: p50 0.216 s, mean 0.233 s (1,000-tool set); mean 0.185 s (public pools).

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

Quotes: Runpod pricing page (updated 2026-11-12), Spheron's B300 price survey (2026-11-11),
getdeploying market median (2026-11-19). Full table: `results/head_to_head_cost_by_market.json`.

# Concurrency measurement (2026-11-21)

Closed loop on one B300, 1,000-tool library, 200 requests, served in waves of C:

| C | run 1, ms/search | run 2, ms/search |
|---|---|---|
| 1 | 288.0 | 214.6 |
| 2 | 225.5 | 225.8 |
| 4 | 238.5 | 239.9 |
| 8 | 241.0 | 239.6 |
| 16 | 239.4 | 239.3 |
| 32 | 238.8 | 238.9 |

Sustained throughput: 4.18 searches/s = **15,000 an hour**, 239 ms a search, **$0.46-0.47** per
1,000 at $6.94-7.09/GPU-hour. Concurrency does not lower cost per search: one search's rerank
batch already fills the GPU, and C = 4-32 all land at the same time. The C = 1 column varies
between runs (214.6-288.0 ms) more than the batched ones - single-request latency includes the
GPU idling between calls. `results/head_to_head_throughput.json`.

# Scaling: rerank depth and throughput (2026-11-21)

One B300, 1,000-tool library, 200 requests in waves of 8. Quality recomputed from the stored
per-question scores at fusion weight 0.25; throughput measured.

| Reranked | Public nDCG@10 | Own set (credited) | ms/search | Searches/hour | $/1,000 (our pod) | $/1,000 (Runpod community) |
|---|---|---|---|---|---|---|
| 0 (embedder only) | 62.93 | 78.42 | 1.7 | 2,088,775 | 0.003 | 0.003 |
| 5 | 60.85 | 80.06 | 29.1 | 123,548 | 0.057 | 0.056 |
| 10 | 67.11 | 84.50 | 47.6 | 75,623 | 0.094 | 0.092 |
| 25 | 70.08 | 87.43 | 124.0 | 29,032 | 0.244 | 0.239 |
| 50 (default) | 70.26 | 87.93 | 241.3 | 14,917 | 0.475 | 0.465 |

Time is linear in rerank depth: the cross-encoder runs once per candidate and nothing else
costs measurable time. Depth 25 is the sweet spot - 0.2 nDCG@10 below 50 and twice the
throughput. `results/scale_curve.json`, `results/quality_vs_depth.json`, `scripts/v3_scale_curve.py`.

# Default configuration chosen (2026-11-21)

Fusion weight x rerank depth, nDCG@10 (stored per-question scores, both benchmarks):

| Depth | Public w=0 / .25 / .5 / 1 | Own set w=0 / .25 / .5 / 1 | Searches/hour | $/1,000 |
|---|---|---|---|---|
| 10 | 66.91 / 67.11 / 67.14 / 66.51 | 85.18 / 84.50 / 84.27 / 82.38 | 75,623 | 0.094 |
| 25 | 69.24 / 70.08 / 70.45 / 69.55 | 87.86 / 87.43 / 86.77 / 84.39 | 29,032 | 0.244 |
| 50 | 68.94 / 70.26 / 70.78 / 69.88 | 87.65 / 87.93 / 87.23 / 85.29 | 14,917 | 0.475 |

Chosen: **depth 25, fusion 0.5** - the public benchmark's best value at that depth (70.45), the
own set loses 0.7 (87.43 -> 86.77), and the throughput doubles. Depth 25 with fusion 0.5 is now
the library default (`ToolIndex(rerank_top=25, fuse=0.5)`); the evaluation scripts keep the
benchmark protocols (depth 50 / top 100) they were run with. `results/default_config.json`.
