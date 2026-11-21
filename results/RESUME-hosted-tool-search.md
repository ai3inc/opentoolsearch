# 진행 중 : 개최형(호스팅) 도구 검색 비교 (2026-11-21 밤)

## 이미 끝난 것 (results/hosted_tool_search.json, 자체 세트 1,000도구·200요청·요청만)

| 시스템 | nDCG@10 (대체 도구 인정) | Recall@10 | 평균 반환 도구 | 검색 안 함 | $/1,000 | 지연 p50 |
|---|---|---|---|---|---|---|
| Claude Fable 5.2 (내장 도구 검색, BM25) | 64.5 | 74.2 | 16.7 | 0/200 | $29.8 | 4.4 s |
| Claude Haiku 5.5 (같음) | 47.3 | 53.2 | 10.2 | 31/200 | $6.5 | 2.8 s |
| OpenAI gpt-5.4 (호스팅 도구 검색) | **미완** (아래 참조) | | | | | |
| 참고: OpenToolSearch (기본 깊이 25) | **86.8** | | 50 후보 중 10 | — | $0.24 | 0.12 s |
| 참고: Gemini API 도구 검색 3.6-flash / 4-hydrogen | 88.2 / 88.3 | | 2.9 / 4.5 | 38 / 39 | $28 / $51 | 6~10 s |

## OpenAI 에서 걸린 것

- 첫 실행: 200요청 중 **87건이 검색을 하지 않고** 그냥 답했다(도구 1,000개를 넘겨도 모델이 검색을 생략). nDCG 13.6.
- 해결: 요청에 `"tool_choice": "required"` 를 넣으면 **검색이 강제된다**(실측 확인). 이 옵션을 넣어 재실행 중이었다.
- 맥 전원이 꺼져 재실행이 중단됨. 남은 파일 `benchmarks/ots-mcp-1k/runs/openai-gpt-5.4-partial.jsonl`.

## 다시 돌리는 방법 (약 15분)

```bash
cd ~/projects/opentoolsearch
OK=$(grep -m1 -E '^OPENAI_API_KEY=' ~/.config/ko-llm/env | sed -E 's/^OPENAI_API_KEY=//; s/^"//; s/"$//')
OPENAI_API_KEY="$OK" WORKERS=3 python3 -u scripts/v5_run_openai.py data/v3/bench.json data/v3/oai-gpt-5.4.jsonl gpt-5.4
```
(이미 끝난 문항은 건너뛰고 이어서 돈다. `data/v3/oai-gpt-5.4.jsonl` 를 남겨 두면 부분 결과부터 이어짐.)

그 다음 점수:
```bash
PRICES="oai-gpt-5.4:2.50:15.00,ant-haiku-5-5:1.00:5.00,ant-fable-5-2:3.00:15.00" \
  python3 scripts/v5_score_hosted.py data/v3 results/hosted_tool_search.json \
  oai-gpt-5.4=data/v3/oai-gpt-5.4.jsonl ant-haiku-5-5=data/v3/ant-haiku-5-5.jsonl ant-fable-5-2=data/v3/ant-fable-5-2.jsonl
```

## 아직 문서에 반영 안 함

- README·HF 카드·공정 비교 그림에 위 세 줄을 아직 넣지 않았다(OpenAI 결과가 미완이라).
- 넣을 때 주의: **호스팅 모델은 도구 1,000개 전체를 받고 스스로 검색**하게 한 조건이고, 우리는 임베더로 25개를 먼저 추린 뒤 리랭크하는 조건이다. 표에 조건 열을 따로 두어야 한다.
- Fable 5.2 는 64.5 로 Claude 계열 중 최고지만 우리 86.8 보다 22점 낮다. 다만 **공개 벤치(ToolRet)** 로는 아직 안 쟀다 — 공정 비교를 위해 돌린다면 `data/v4/bench_public.json` 으로 같은 스크립트를 쓰면 된다(공개 세트는 풀이 2,000개 이하라 더 빠름).
