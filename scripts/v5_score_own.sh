#!/bin/sh
# Score every hosted tool-search run on the own set (OTS-MCP-1k) -> results/hosted_tool_search.json
# Prices: USD per 1M tokens (input:output:cache read:cache write), list prices of Nov 2026 (BizRouter model catalogue, vendor price pages).
cd "$(dirname "$0")/.." || exit 1
PRICES="oai-gpt-6.2-sol:2:10:0.1:2.5,oai-gpt-6.2-sol-run2:2:10:0.1:2.5,oai-gpt-6-luna:0.1:0.5:0.01:0.125,oai-gpt-5.4:2.5:15:0.25,\
ant-opus-5-5-max:4:20:0.2:5,ant-haiku-5-5-forced:0.1:0.5:0.01:0.125,ant-haiku-5-5:0.1:0.5:0.01:0.125,ant-fable-5-2:8:40:0.2:10,\
gemini-4-hydrogen:1.5:7.5:0.075,gemini-3.6-flash:0.75:3.75:0.075,gemini-3.8-flash:0.75:3.75:0.075" \
python3 scripts/v5_score_hosted.py data/v3 results/hosted_tool_search.json \
  oai-gpt-6.2-sol-run2=data/v3/oai-gpt-6.2-sol-run2.jsonl oai-gpt-6.2-sol=data/v3/oai-gpt-6.2-sol.jsonl \
  oai-gpt-6-luna=data/v3/oai-gpt-6-luna.jsonl oai-gpt-5.4=data/v3/oai-gpt-5.4.jsonl \
  ant-opus-5-5-max=data/v3/ant-opus-5-5-max.jsonl ant-haiku-5-5-forced=data/v3/ant-haiku-5-5-forced.jsonl \
  ant-haiku-5-5=data/v3/ant-haiku-5-5.jsonl ant-fable-5-2=data/v3/ant-fable-5-2.jsonl \
  gemini-4-hydrogen=data/v3/gemini-4-hydrogen.jsonl gemini-3.6-flash=data/v3/gemini-3.6-flash.jsonl gemini-3.8-flash=data/v3/gemini-3.8-flash.jsonl
