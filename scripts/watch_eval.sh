#!/bin/bash
# watch_eval.sh RUN CKPT...  : evaluate checkpoints (w/ inst) as they appear
cd /workspace/toolsift; export HF_HOME=/workspace/hf PYTHONPATH=/workspace/toolsift TOKENIZERS_PARALLELISM=true
RUN=$1; shift
for c in "$@"; do
  until [ -f /workspace/runs/$RUN/$c/model.safetensors ]; do sleep 30; done; sleep 20
  eval "python -m opentoolsearch.evaluate --model /workspace/runs/$RUN/$c --inst --bs 256 $EVAL_ARGS --out results/${RUN}_$c" > /workspace/logs/eval_${RUN}_$c.log 2>&1
  echo "== $RUN/$c $(grep 'NDCG@10' /workspace/logs/eval_${RUN}_$c.log)" >> /workspace/logs/evals.log
done
