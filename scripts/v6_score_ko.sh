#!/bin/sh
# Korean requests (data/v6): hosted tool search -> results/hosted_tool_search_ko.json; open systems and the decision models -> results/korean_open.json
cd "$(dirname "$0")/.." || exit 1
PRICES="oai-gpt-6.2-sol:2:10:0.1:2.5,oai-gpt-6-luna:0.1:0.5:0.01:0.125,ant-opus-5-5-max:4:20:0.2:5,ant-haiku-5-5:0.1:0.5:0.01:0.125,\
gemini-4-hydrogen:1.5:7.5:0.075,gemini-3.6-flash:0.75:3.75:0.075,gemini-3.8-flash:0.75:3.75:0.075" \
python3 scripts/v5_score_hosted.py data/v6 results/hosted_tool_search_ko.json \
  oai-gpt-6.2-sol=data/v6/oai-gpt-6.2-sol.jsonl oai-gpt-6-luna=data/v6/oai-gpt-6-luna.jsonl \
  ant-opus-5-5-max=data/v6/ant-opus-5-5-max.jsonl ant-haiku-5-5=data/v6/ant-haiku-5-5.jsonl \
  gemini-4-hydrogen=data/v6/gemini-4-hydrogen.jsonl gemini-3.6-flash=data/v6/gemini-3.6-flash.jsonl gemini-3.8-flash=data/v6/gemini-3.8-flash.jsonl
python3 scripts/v6_score_open.py data/v3/bench.json data/v6/open_scores.json results/korean_open.json > /dev/null
