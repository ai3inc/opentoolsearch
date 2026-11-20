#!/bin/bash
# Round-2 reranker ablations: train (300 steps) -> 50/50 weight average with the base -> dev eval.
cd /workspace/toolsift; export HF_HOME=/workspace/hf PYTHONPATH=/workspace/toolsift TOKENIZERS_PARALLELISM=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "rows left with none" /workspace/logs/v2_make_train.log 2>/dev/null; do sleep 20; done
BASE=Qwen/Qwen3-Reranker-0.6B
run() {  # name data p_drop_inst
  if [ ! -f /workspace/runs/$1/final/model.safetensors ]; then
    python -m opentoolsearch.rerank train --base $BASE --data $2 --out /workspace/runs/$1 --bs 24 --accum 2 --negs 7 \
      --steps 300 --lr 6e-6 --warmup 30 --p_aug 0.5 --p_drop_inst $3 > /workspace/logs/train_$1.log 2>&1
  fi
  python scripts/soup_lm.py /workspace/runs/$1_soup50 $BASE:0.5 /workspace/runs/$1/final:0.5 > /workspace/logs/soup_$1.log 2>&1
  python scripts/v2_eval_dev.py data/v2 $1=/workspace/runs/$1_soup50 2>&1 | grep -E "^\[.*$1" >> /workspace/logs/v2_dev_ablate.log
  echo "done $1 $(date +%H:%M)" >> /workspace/logs/v2_dev_ablate.log
}
run a1 data/v2/train_v1h.jsonl 0.1     # round-1 recipe, held-out servers removed
run a2 data/v2/train_v2.jsonl 0.1      # + same-tool negatives removed
run a3 data/v2/train_v2.jsonl 0.5      # + half the questions without an instruction
echo ALLDONE >> /workspace/logs/v2_dev_ablate.log
