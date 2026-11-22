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
  time scaled by the memory-bandwidth ratio (3.35 vs about 8 TB/s) - flagged as an estimate.
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

# Fair comparison, same first stage (2026-11-21, corrected 2026-11-22)

Our embedder finds the candidates; the second stage is our reranker, Jev, or Gemini Decision,
at 25 and 50 candidates. The decision models' cost is what they billed for those candidates plus the
embedder, $0.0034 per 1,000 searches.

**Correction (2026-11-22).** The first version of this section, the README table and the figure
added $0.24 / $0.475 to the decision models as the "first stage" cost. Those are the costs of our whole
pipeline (embedder plus reranker at depth 25 / 50); the embedder alone costs $0.0034 per 1,000
(2,088,775 searches an hour on one B300 at $7.09/h). The decision models therefore cost less than our
reranker, not 1.5-1.7 times more. Quality numbers were not affected.

| Candidates | Second stage | Public | Own set | Cost / 1,000 (public / own) |
|---|---|---|---|---|
| 25 | OpenToolSearch | 70.45 | 86.77 | 0.24 / 0.24 |
| 25 | Jev | 76.75 | 89.33 | 0.12 / 0.07 |
| 25 | Gemini Decision | 77.69 | 90.00 | 0.24 / 0.12 |
| 50 | OpenToolSearch | 70.78 | 87.23 | 0.475 / 0.475 |
| 50 | Jev | 77.35 | 89.08 | 0.16 / 0.12 |
| 50 | Gemini Decision | 78.25 | 91.21 | 0.33 / 0.22 |

Raw runs: `benchmarks/*/runs/k25-*.jsonl` (25 candidates) and the existing 50-candidate files
(`jev-choice`, `gdec-choice`, and `gemini-decision-choice` on the own set). The decision
models' "single choice" probability is rounded to two decimals, so ties keep the given order.
The fair figure: `results/fair_comparison.json`, `scripts/make_fair_figure.py`.


## Hosted tool search, decision models and dedicated retrievers on the comparison benchmarks

2026-11-21. These runs are not ToolRet test evaluations of our model, but they share the
public benchmark's questions with the rest of this file, and they are what the comparison
tables of the README are made of - so every run is here, including the ones that failed or were
stopped, and what each cost.

Conditions: request only; the 1,000 MCP tools of OTS-MCP-1k (English and Korean) or the pool of
each question's own source dataset (ToolRet per-source), every tool sent deferred; the model
searches itself. OpenAI and Claude Haiku 5.5 were forced to search (`tool_choice`); Gemini,
Opus 5.5 and Fable 5.2 were not (the Anthropic API returns HTTP 400 for a forced tool choice on
Opus 5.5 and Fable 5.2; Gemini's API has no such option). Reasoning depth: Claude Opus 5.5 at
`effort: max` (adaptive thinking, `max_tokens` 16,000); every other model at its default
(Haiku 5.5: medium; gpt-6 models and Gemini: not set). nDCG@10 as in the README: interchangeable
tools credited on our own set, strict labels on the public one (macro over 32 subsets).
Costs: list price = every input token at the full input rate; cache = the API's reported cache
hits at the cache-read price and writes at the cache-write price; "Run cost" is the cache-aware
figure, i.e. what the run was billed.

### Every run

| Benchmark | System | Answered | nDCG@10 | Recall@10 | Skipped the search | $/1,000 list | $/1,000 with cache | Run cost $ | Median s | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| ToolRet per-source | Gemini API tool search (gemini-4-hydrogen) | 769 | 73.51 | 79.40 | 39 | 51.20 | 30.38 | 23.36 | 10.03 |  |
| ToolRet per-source | Gemini API tool search (gemini-3.8-flash) | 769 | 70.64 | 75.27 | 56 | 27.54 | 17.16 | 13.20 | 9.95 | 1 error counted as a miss |
| ToolRet per-source | Gemini API tool search (gemini-3.6-flash) | 763 | 67.02 | 71.58 | 38 | 28.01 | 17.50 | 13.35 | 6.24 | 6 errors counted as misses |
| ToolRet per-source | OpenAI tool search (gpt-6.2-sol) | 769 | 66.64 | 68.06 | 0 | 34.09 | 41.77 | 32.12 | 3.43 | cache never hit: every request wrote the cache (billed above list) |
| ToolRet per-source | OpenAI tool search (gpt-6-luna) | 769 | 65.45 | 65.24 | 0 | 1.74 | 0.35 | 0.27 | 3.15 |  |
| ToolRet per-source | Claude tool search (Opus 5.5, effort max) | 769 | 60.10 | 66.18 | 18 | 62.19 | 62.19 | 47.82 | 11.77 | search not forced (documented as refused); effort max |
| ToolRet per-source | Claude tool search (Haiku 5.5) | 769 | 53.34 | 60.93 | 0 | 0.52 | 0.52 | 0.40 | 2.53 |  |
| OTS-MCP-1k (English) | gemini-4-hydrogen | 199 | 88.29 | 93.75 | 5 | 87.90 | 49.55 | 9.86 | 11.19 |  |
| OTS-MCP-1k (English) | gemini-3.8-flash | 200 | 85.62 | 89.25 | 9 | 47.07 | 27.19 | 5.44 | 8.78 |  |
| OTS-MCP-1k (English) | gemini-3.6-flash | 200 | 88.21 | 92.25 | 4 | 47.43 | 27.60 | 5.52 | 7.23 |  |
| OTS-MCP-1k (English) | oai-gpt-6.2-sol-run2 | 200 | 82.42 | 85.58 | 0 | 135.33 | 8.60 | 1.72 | 4.75 | second run, cache hit on every request (the first run below: no hits) |
| OTS-MCP-1k (English) | oai-gpt-6.2-sol | 200 | 82.55 | 85.25 | 0 | 135.37 | 168.72 | 33.74 | 4.58 | first run: cache never hit, every request wrote the cache |
| OTS-MCP-1k (English) | oai-gpt-6-luna | 200 | 88.18 | 89.42 | 0 | 6.81 | 0.85 | 0.17 | 3.9 |  |
| OTS-MCP-1k (English) | oai-gpt-5.4 | 200 | 87.27 | 87.83 | 0 | 169.75 | 23.69 | 4.74 | 4.13 | flat deferred list + forced search; superseded by gpt-6 models |
| OTS-MCP-1k (English) | ant-opus-5-5-max | 200 | 61.76 | 70.42 | 0 | 105.48 | 105.48 | 21.10 | 9.54 | search not forced (documented as refused); effort max |
| OTS-MCP-1k (English) | ant-haiku-5-5-forced | 200 | 57.10 | 66.00 | 0 | 0.70 | 0.70 | 0.14 | 2.66 | search forced |
| OTS-MCP-1k (English) | ant-haiku-5-5 | 200 | 47.27 | 53.17 | 31 | 0.65 | 0.65 | 0.13 | 2.78 | search not forced: 31 of 200 requests skipped it |
| OTS-MCP-1k (English) | ant-fable-5-2 | 200 | 64.52 | 74.25 | 0 | 79.46 | 79.46 | 15.89 | 4.38 | search not forced; run once, dropped for cost; forcing it is rejected (HTTP 400) |
| OTS-MCP-1k (Korean) | oai-gpt-6.2-sol | 200 | 81.12 | 84.42 | 0 | 135.48 | 168.84 | 33.77 | 4.59 | cache never hit (billed above list) |
| OTS-MCP-1k (Korean) | oai-gpt-6-luna | 200 | 85.49 | 86.58 | 0 | 6.82 | 0.86 | 0.17 | 4.08 |  |
| OTS-MCP-1k (Korean) | ant-opus-5-5-max | 200 | 58.92 | 65.33 | 2 | 122.28 | 122.28 | 24.46 | 9.72 |  |
| OTS-MCP-1k (Korean) | ant-haiku-5-5 | 200 | 56.51 | 65.25 | 0 | 0.67 | 0.67 | 0.13 | 2.68 |  |
| OTS-MCP-1k (Korean) | gemini-4-hydrogen | 200 | 83.82 | 89.25 | 12 | 87.96 | 49.65 | 9.93 | 10.88 |  |
| OTS-MCP-1k (Korean) | gemini-3.6-flash | 199 | 84.07 | 89.67 | 5 | 47.58 | 28.55 | 5.68 | 5.61 |  |
| OTS-MCP-1k (Korean) | gemini-3.8-flash | 200 | 78.98 | 81.92 | 20 | 45.72 | 26.10 | 5.22 | 9.21 |  |

Sum of the run costs above (cache-aware, as billed): $308.

### What went wrong, and what was decided

What went wrong on the way, in the order it happened:

- **OpenAI, first attempts (gpt-5.4, own set).** Left alone, the model skipped the search on 87 of
  200 requests (nDCG@10 13.6). With `tool_choice: required` and the 1,000 tools as 100 namespaces
  of 10, the hosted search returned nothing for most requests (3.6). With one flat deferred list
  and the search forced it scored 87.27 - the setup used from then on. gpt-5.4-mini returned
  about 0 tools at this scale (0.0; 64 requests, $4) and gpt-5.4-nano is not supported (HTTP 400).
- **Claude Haiku 5.5** skipped the search on 31 of 200 requests when left free (47.27); forced,
  it scored 57.10. **Fable 5.2** answered HTTP 400 to all 200 forced requests (nothing billed) and
  Anthropic documents the same for Opus 5.5, so both ran free. Fable 5.2 searched every time
  (64.52); after that run it was dropped for cost, at the owner's request.
- **gpt-6.2-sol's prompt cache.** In three of its four complete runs (own set first run, Korean,
  public) not one request read the cache: every request wrote its 66,700-token prefix at the
  cache-write price, 25% above list. In the second own-set run, started after twenty sequential
  requests, every request hit (66,700 cached tokens) and the run cost $1.72 instead of $33.74.
  We do not know why; the answers did not change (82.55 against 82.42; the two runs return the
  same tool list for 107 of 200 requests, the rest is ordinary run-to-run variation).
  The README uses the list price where the cache never hit and says so.
- **gpt-6.2-sol against gpt-6-luna.** Sol scores below luna on our set (82.4 against 88.2) and
  above it on the public one (66.6 against 65.5). With the search forced both load about two
  tools per request and search about once; sol writes about a third of the reasoning tokens
  luna does (34 against 97 per request) at the same default setting. We did not raise either
  model's reasoning effort.
- **OpenAI refused one tool name** of the public benchmark: `history`, in the rotbench subset,
  is a reserved namespace (HTTP 400, 25 questions failed for each model). The tool was sent as
  `history_`, the answers mapped back, and the 25 questions re-run.
- **Claude Opus 5.5 on the public benchmark** was started on all 769 questions and stopped at 45
  because the full run looked too expensive ($2.39 spent, discarded). It was restarted on a
  256-question stratified sample (8 per subset), stopped again at 69 and resumed to the end of
  the sample ($15.83; 59.0 on the sample), then run on the other 513 questions on the owner's
  instruction (769 questions, $47.82 in all; 60.1 on the full set).
- **gemini-3.6-flash was used first** (it had run on the first benchmarks), and gemini-3.8-flash,
  the newer cost-efficient model at the same price, was added afterwards on all three
  benchmarks. It is above 3.6-flash on the public benchmark (70.6 against 67.0) and below it on
  ours (85.6 against 88.2) and in Korean (79.0 against 84.1), where it skipped the search on 20 of
  200 requests.
- **Korean translation.** Claude Sonnet 5.5 translated the 200 requests, gemini-3.6-flash checked
  each against the English (197 of 200 rated 5 of 5, three rated 3-4). Three translations were
  corrected by hand: one had an introduction line the translator had added, one had lost
  `university`, one `facial`. About $0.3 in all.
- **Dedicated retrievers.** Each ran with every query format its card documents, and the one
  with the best English own-set nDCG@10 is reported on all benchmarks (table below). That
  choice is made on a benchmark the models are also shown on, which favours them, not us. The
  four ToolRet-trained retrievers are English-only models. Tool-DE (Tool-Embed-4B + Tool-Rank-4B,
  Lux1997) was trained on tool descriptions expanded by an LLM, which we did not generate: it ran
  on the original texts, and its card does not document a prompt, so Tool-Rank was run with the
  Qwen3-Reranker prompt of its base model (it scored best of all open systems regardless, so
  the prompt was evidently close enough); its number is a lower bound. We do not know whether the
  training data of Tool-Embed and Tool-Rank overlaps Toucan, the source of OTS-MCP-1k. MagicSelector has no
  public weights we could find; ToolGen is generative and only published as GGUF; neither was run.
- **Infrastructure.** Two runs on the GPU pods lost their output to a model-download failure
  (the hub's brotli-compressed replies broke the HTTP client; removing the `brotli` package fixed
  it) and were re-run from scratch; no result depends on them.
- **Money.** The run costs above add up to the figure below; discarded partial runs (Opus 5.5 on
  the public benchmark: about $2.4; Fable 5.2 and the failed forced-search attempts: about $0) and
  GPU pod time (about 3 hours of one B300 at $7.09/h) are not in it.

### Dedicated retrievers: every query format that was run

| Model | Query format | OTS-MCP-1k English | Korean | ToolRet per-source | |
|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | callbus-instruction | 83.56 | 78.97 | 63.59 | **used** |
| Qwen3-Embedding-8B | callbus-instruction | 84.37 | 82.91 | 65.48 | **used** |
| Tool-DE (Tool-Embed-4B + Tool-Rank-4B) | callbus-instruction | 89.57 | 87.42 | 71.95 |  |
| Tool-DE (Tool-Embed-4B + Tool-Rank-4B) | qwen3-default-instruction | 91.42 | 89.52 | 71.99 | **used** |
| Tool-DE (Tool-Embed-4B + Tool-Rank-4B) | tool-prompt | 91.38 | 89.36 | 72.04 |  |
| Tool-Embed-0.6B | callbus-instruction | 83.29 | 75.24 | 63.46 |  |
| Tool-Embed-0.6B | qwen3-default-instruction | 82.30 | 74.56 | 62.10 |  |
| Tool-Embed-0.6B | tool-prompt | 86.34 | 77.95 | 64.05 | **used** |
| Tool-Embed-4B | callbus-instruction | 82.67 | 76.43 | 66.48 |  |
| Tool-Embed-4B | qwen3-default-instruction | 83.72 | 77.08 | 65.84 |  |
| Tool-Embed-4B | tool-prompt | 86.57 | 81.49 | 67.27 | **used** |
| ToolRet-trained-bge-base-en-v1.5 | plain | 80.87 | 14.97 | 59.78 | **used** |
| ToolRet-trained-bge-base-en-v1.5 | bge-instruction | 79.93 | 13.11 | 60.05 |  |
| ToolRet-trained-bge-base-en-v1.5 | tool-prompt | 74.63 | 9.84 | 58.44 |  |
| ToolRet-trained-bge-large-en-v1.5 | plain | 79.27 | 11.29 | 63.42 |  |
| ToolRet-trained-bge-large-en-v1.5 | bge-instruction | 81.44 | 14.73 | 63.29 | **used** |
| ToolRet-trained-bge-large-en-v1.5 | tool-prompt | 77.28 | 8.84 | 60.35 |  |
| ToolRet-trained-e5-base-v2 | query | 83.88 | 12.57 | 59.90 | **used** |
| ToolRet-trained-e5-base-v2 | query+tool-prompt | 82.08 | 14.21 | 59.95 |  |
| ToolRet-trained-e5-large-v2 | query | 82.46 | 21.31 | 60.47 | **used** |
| ToolRet-trained-e5-large-v2 | query+tool-prompt | 79.06 | 7.71 | 58.28 |  |
| callbus-emb-8b | callbus-instruction | 86.63 | 84.19 | 68.57 | **used** |

## First-stage swap behind the OpenToolSearch reranker (2026-11-22)

Eight first-stage retrievers (four on 2026-11-22, Nemotron-3-Embed-8B added later the same day, OpenToolSearch Embed 0.6B, its base Qwen3-Embedding-0.6B and Tool-Embed-0.6B after that), each followed by the published reranker on its top 25, fusion 0.5, on
OTS-MCP-1k (English, Korean) and ToolRet per-source: `scripts/v8_first_stage_rerank.py`, scored by
`scripts/v8_score.py`. Two B300 pods, about 40 and 15 minutes (about $7 in all); no API cost. The control
(Nemotron-3-Embed-1B, the library default until 0.2) reproduced the default numbers: 70.4 / 86.2 / 84.7
against 70.45 / 86.1 / 84.6. Query formats: callbus-emb-8b and Qwen3-Embedding-8B the callbus
instruction, Tool-Embed-4B the tool prompt, both Nemotron models `query: ` / `passage: `, Qwen3-Embedding-0.6B and
OpenToolSearch Embed 0.6B the callbus instruction.

<!-- table:firststage -->
| First stage | Alone (public / own / Korean) | + OpenToolSearch reranker | Gain | Every labelled tool in the top 25 |
|---|---|---|---|---|
| callbus-emb-8b (7.6B) | 68.6 / 86.6 / 84.2 | **71.8 / 89.4 / 89.0** | +3.2 / +2.8 / +4.8 | 84% / 100% / 98% |
| Nemotron-3-Embed-8B (8B) | 66.1 / 81.6 / 77.7 | **71.2 / 87.1 / 86.3** | +5.2 / +5.5 / +8.6 | 83% / 95% / 93% |
| Qwen3-Embedding-8B (7.6B) | 65.5 / 84.4 / 82.9 | **71.0 / 88.5 / 88.4** | +5.6 / +4.1 / +5.5 | 84% / 98% / 98% |
| Tool-Embed-4B (4.0B) | 67.3 / 86.6 / 81.5 | **71.0 / 89.2 / 85.8** | +3.7 / +2.6 / +4.3 | 82% / 98% / 95% |
| OpenToolSearch-Embed-0.6B (0.6B) - the library default | 66.0 / 85.9 / 80.4 | **70.8 / 89.0 / 88.2** | +4.8 / +3.1 / +7.7 | 83% / 98% / 98% |
| Qwen3-Embedding-0.6B (0.6B) - the base of OpenToolSearch Embed | 63.6 / 83.6 / 79.0 | **70.6 / 87.3 / 87.2** | +7.0 / +3.7 / +8.2 | 82% / 96% / 94% |
| Nemotron-3-Embed-1B (1.14B) - the library default before 0.2 | 63.0 / 79.0 / 74.0 | **70.4 / 86.2 / 84.7** | +7.4 / +7.2 / +10.7 | 81% / 94% / 92% |
| Tool-Embed-0.6B (0.6B) | 64.0 / 86.3 / 78.0 | **69.3 / 89.3 / 85.8** | +5.2 / +2.9 / +7.9 | 81% / 98% / 94% |
<!-- /table:firststage -->

Reranker alone on the 25 candidates (no fusion): callbus 69.4 / 88.0 / 87.2, Qwen3-Embedding-8B
68.9 / 88.0 / 86.7, Tool-Embed-4B 68.7 / 87.5 / 84.6, Nemotron-1B 69.1 / 86.3 / 84.2, Nemotron-8B 69.8 / 86.4 / 84.5, OpenToolSearch Embed 0.6B 68.7 / 87.7 / 86.1, Qwen3-Embedding-0.6B 68.9 / 86.7 / 84.7 - fusion with the first
stage adds up to 2.5 points (one case, Nemotron on our English set, loses 0.2). Not tried: other embedders, other fusion weights, depth 50.

## OpenToolSearch Embed 0.6B, the new default first stage (2026-11-22)

A 0.6B embedder fine-tuned from Qwen3-Embedding-0.6B (recipe not published; data: ToolRet-train with the near-copy
filter, Toucan rows from servers not held out for OTS-MCP-1k, about 4% machine-translated Korean requests; nothing from
either benchmark's test side). **One full training run**; the last checkpoint is the published one - no checkpoint or variant
was chosen on test numbers. It was then run as a first stage behind the reranker (table above): 70.8 / 89.0 / 88.2, against
70.6 / 87.3 / 87.2 for its base and 70.4 / 86.2 / 84.7 for Nemotron-3-Embed-1B (paired bootstrap in BENCHMARKS.md: only
the gain over Nemotron on our own set is established; the gain over the base is not).

What it cost: one B300 spot pod for about 4.7 hours (about $33, of which about $6 was the pod idling for 50 minutes behind a
wait that never ended) and $1.94 of API calls for data preparation. A second B300 pod, about 25 minutes (about $3), ran the
library end to end (`scripts/v10_default_speed.py`): four runs - the default pair, the same again, the Nemotron pair and the
callbus pair. No hosted-API benchmark was re-run.

| Pair (library run, one B300) | Own set EN / KO | Time per search, waves of 8 | Searches an hour | $ per 1,000 | Single request p50 / p90 / mean |
|---|---|---|---|---|---|
| OpenToolSearch Embed 0.6B + Rerank 0.6B (default) | 88.79 / 88.16 | 96.9 ms | 37,165 | 0.191 | 66.4 / 114.9 / 77.4 ms |
| the same, repeated | 88.79 / 88.16 | 96.9 ms | 37,149 | 0.191 | 66.2 / 115.0 / 77.5 ms |
| Nemotron-3-Embed-1B + Rerank 0.6B | 86.08 / 84.35 | 116.3 ms | 30,958 | 0.229 | 84.0 / 137.6 / 88.0 ms |
| callbus-emb-8b + Rerank 0.6B | 89.79 / 88.64 | 104.0 ms | 34,617 | 0.205 | 70.3 / 138.4 / 81.3 ms |

One at a time the default pair took 79.2 ms a search (45,442 an hour), in waves of 32 94.9 ms. The library run scores 0.2-0.4
below or above the first-stage-swap numbers (the library builds the tool text itself). `results/default_speed*.json`.

Corrections made on the way, to numbers published earlier:

- The summary table and figure showed the callbus-emb-8b + reranker pipeline as "about 0.14 s and $0.28" - the two stages' times
  added. Measured end to end it is 104 ms and $0.205; the estimate was too high and is replaced.
- "Median latency" for our pipelines has always been the time per search at the measured throughput (waves of 8), not a
  single-request median; the single-request median of the default pair is 66 ms. Both are now stated.
- BENCHMARKS.md carried a sentence about the number of B300s a 20-million-turn consumer app would need; its arithmetic did
  not hold and it was removed.

Not done: the depth and fusion sweep and the full-corpus ToolRet run with the new embedder (those numbers are labelled as
Nemotron), the decision models on the new embedder's candidates (hosted API cost), languages other than English and Korean.

## Second-stage combinations behind three first stages (2026-11-22)

Question: is Tool-DE's lead its 4B embedder or its 4B reranker? `scripts/v11_second_stage_combos.py`, scored by `scripts/v11_score.py`
(`results/second_stage_combos.json`; raw scores for every candidate: `benchmarks/second-stage-combos/`). **Fixed before
the run, nothing chosen on the results**: first stages OpenToolSearch Embed 0.6B, Tool-Embed-4B, Tool-Embed-0.6B; top 25 to our reranker, to
Tool-Rank-4B (pointwise yes/no, Qwen3-Reranker prompt, original tool text - as in the Tool-DE row, so a lower bound), or to both; four scorings
(ours + 0.5 first; Tool-Rank alone; Tool-Rank + 0.5 first; ours + Tool-Rank + 0.5 first), all z-scored over the candidates. One B300 spot pod,
about 27 minutes (about $3.2), no API cost; the pod was created after a wait for a free B300. The same pod re-ran the first-stage swap for
Tool-Embed-0.6B (69.3 / 89.3 / 85.8 behind our reranker, in the table above).

Results are in BENCHMARKS.md ("Swapping the second stage"). The control (Tool-Embed-4B, then Tool-Rank-4B alone) gave 72.0 / 91.4 / 89.2 against the
earlier Tool-DE run's 72.0 / 91.4 / 89.5. Behind our embedder Tool-Rank-4B alone gave 71.8 / 90.8 / 90.8; with our reranker added 72.2 / 90.2 / 90.8.
Time per search in waves of 8 (embedding included): our reranker 95.5 ms, Tool-Rank-4B 249.1 ms, both 342.1 ms. The library-default column of the
combination table differs from the first-stage table by up to 0.2 points (bf16 batch composition). Not done: other depths or fusion weights,
LLM-expanded tool texts for Tool-Rank, other first stages.

## Rerank depth 50 against 25 for the new default pair (2026-11-22)

`OTS_DEPTH=50 python scripts/v8_first_stage_rerank.py ...` (quality, three tests) and `OTS_DEPTH=50 python scripts/v10_default_speed.py` (speed), one B300
spot pod, about 30 minutes (about $3.5), no API cost. Fusion 0.5, first stage OpenToolSearch Embed 0.6B. Depth 50: 71.0 / 88.9 / 88.2 against 70.8 / 89.0 / 88.2
at depth 25; paired bootstrap of the difference: public +0.2 [-0.3, +0.7], own English -0.1 [-0.6, +0.3], own Korean +0.1 [-0.7, +1.0]. Speed in waves of 8:
194.3 ms, 18,527 an hour, $0.383 per 1,000 (depth 25 on the same pod: 96.7 ms, 37,245 an hour, $0.190, a third measurement of the default's 96.9 and 96.9 ms).
Decision: depth 25 stays the default. `results/default_speed_depth50.json`, `results/depth50_vs_25_bootstrap.json`,
`benchmarks/first-stage-swap/OpenToolSearch-Embed-0.6B-depth50.json.gz`.

## The stages used separately through the library (2026-11-22)

`pip install git+https://github.com/ai3inc/opentoolsearch` on a fresh B300 spot pod (about 35 minutes with two attempts, about $4; the first attempt failed because the image
had no `git`; an earlier try on an 8 vCPU / 16 GB CPU pod was killed for memory loading Tool-Rank-4B and gave no result), then: embedding only; our embedder +
`Lux1997/Tool-Rank-4B` with `fuse=0.0` and `0.5`; `Qwen/Qwen3-Embedding-0.6B` + `query_prefix` in front of our reranker - all returned sensible results on the 12-tool example.
End to end through `scripts/v10_default_speed.py` with Tool-Rank-4B and `fuse=0.0`: own set 90.78 English / 91.12 Korean, 256.1 ms a search in waves of 8 (14,059 an hour, $0.504 per
1,000), single request p50 144.5 ms. The benchmark script (v11) had 90.77 / 90.8 and 249.1 ms. `results/default_speed_toolrank.json`.

## Keeping the index current: update, remove, sync, save, load (2026-11-22)

New in 0.2.1 (`ToolIndex.update / remove / sync / save / load`, nine unit tests with a stand-in embedder in `tests/test_index.py`). Measured with the real models and
the package installed from GitHub on one B300 spot pod (about 25 minutes, about $3; the B300 was free on the first try): `scripts/v12_live_index.py`, `results/live_index.json`.
1,000 tools: build 0.9 s; add one 15.9 ms, update one 17.0 ms, remove one 0.7 ms (median of 30); `sync` with 35 changed, 5 new, 5 gone: 143 ms and 40 embeddings (rebuild 885 ms);
save 14 ms (4.6 MB), load 4 ms, no embedding; incremental against from-scratch vectors: minimum cosine 0.996, identical top 5 for 200 of 200 requests; saved-and-loaded index identical for 200 of 200;
an edited tool was found first by its new description. All 1,000 tool texts are 117,765 tokens (Qwen tokenizer), the mean top 5 about 589.
Before that, a sanity check on a private folder of 59 skills on a spot CPU pod (8 vCPU, 16 GB; about 3 hours of CPU pods in all, counting the failed Tool-Rank-4B run, about 2,000 won): 38 requests we wrote, the embedder alone and with the reranker
first for 38 of 38; with vague last messages 3 of 5, with the previous turn added 5 of 5; adding one skill 134 ms, building 28 s, one reranked search 50 s (CPU). Not reproducible by others and not a benchmark - the requests
were written by us and the descriptions carry trigger phrases. A first attempt to run Tool-Rank-4B on the 16 GB CPU pod (earlier today) died from memory and is in the section above.

