#!/bin/bash
# Zero-shot baselines, w/ instruction, whole-corpus pool.
cd /workspace/toolsift; export HF_HOME=/workspace/hf PYTHONPATH=/workspace/toolsift TOKENIZERS_PARALLELISM=true
for m in "$@"; do
  n=$(echo $m | tr '/' '_')
  python -m opentoolsearch.evaluate --model $m --inst --bs 256 --out results/zs_$n > /workspace/logs/zs_$n.log 2>&1
  echo "== $m"; grep -E "NDCG@10|Error|error" /workspace/logs/zs_$n.log | tail -2
done
echo ALLDONE
