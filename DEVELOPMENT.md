# OpenToolSearch - how it was built, and how to reproduce it

Training, the validation set of our own, the checks on the benchmark's labels, and the commands to
re-run everything. The numbers are in [BENCHMARKS.md](BENCHMARKS.md); every test evaluation we ran is
listed in [`results/LEDGER.md`](results/LEDGER.md).

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

The comparison benchmarks (hosted tool search, decision models, dedicated retrievers, Korean)
are re-run with the `scripts/v3_*` - `v7_*` runners and scored by `scripts/v4_score.py`,
`v5_score_own.sh`, `v6_score_ko.sh` and `v7_score_dense.py`; the tables and figures are made by
`scripts/make_tables.py` and `scripts/make_figures.py`. Details in
[`benchmarks/README.md`](benchmarks/README.md).

Training data and steps: `scripts/prep_train.py` (clean ToolRet-train), `scripts/prep_external.py`
and `scripts/gen_instructions.py` (Toucan / Glaive rows), `scripts/mine_negs.py`,
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
