#!/bin/bash
# Final ToolRet test evaluation of one reranker: with instruction (top 100) and without (top 50).
# usage: v2_final.sh NAME      (weights at /workspace/runs/NAME)
cd /workspace/toolsift; export HF_HOME=/workspace/hf PYTHONPATH=/workspace/toolsift TOKENIZERS_PARALLELISM=true
N=$1; F=results/zs_nemotron3_1b
python -m opentoolsearch.rerank eval --model /workspace/runs/$N --run $F/run_inst_concat.json --inst --topn 100 --bs 384 --d_len 640 --out results/final_$N > /workspace/logs/final_${N}_inst.log 2>&1
OUTJSON=results/final_$N/board_inst.json python scripts/fuse.py $F/run_inst_concat.json results/final_$N/run_inst.json 0 0.25 0.5 1 2>&1 | grep -E "w=|cross" > /workspace/logs/final_${N}_inst.txt
TOPN=50 python scripts/fuse.py $F/run_inst_concat.json results/final_$N/run_inst.json 0 0.25 0.5 1 2>&1 | grep -E "cross" | head -1 >> /workspace/logs/final_${N}_inst.txt
python -m opentoolsearch.rerank eval --model /workspace/runs/$N --run $F/run_noinst_concat.json --topn 50 --bs 384 --d_len 640 --out results/final_$N > /workspace/logs/final_${N}_noinst.log 2>&1
OUTJSON=results/final_$N/board_noinst.json python scripts/fuse.py $F/run_noinst_concat.json results/final_$N/run_noinst.json 0 0.25 0.5 1 2>&1 | grep -E "w=|cross" > /workspace/logs/final_${N}_noinst.txt
echo FINALDONE >> /workspace/logs/final_${N}_noinst.txt
