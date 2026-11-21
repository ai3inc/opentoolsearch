"""First stage and rerankers on the 200 requests of OTS-MCP-1k, in English and in Korean, same tool texts and pipeline.

Run on a GPU. Stores, per language, the top-50 candidates of the first stage (Nemotron-3-Embed-1B) with their scores and the
score each reranker gives them; fusion and scoring happen afterwards (scripts/v6_score_open.py).

  python scripts/v6_korean_open.py data/v3/bench.json data/v6/ko_queries.json OUT.json name=reranker_path [...]
"""
import json, sys, torch
from sentence_transformers import SentenceTransformer
from opentoolsearch.evaluate import encode
from opentoolsearch.rerank import Reranker

B = json.load(open(sys.argv[1])); KO = {r["id"]: r["ko"] for r in json.load(open(sys.argv[2]))}
OUT, specs, TOPN = sys.argv[3], [s.split("=", 1) for s in sys.argv[4:]], 50
tools = [json.dumps({"name": t["api_name"], "description": t["description"], "parameters": t["parameters"]}, ensure_ascii=False) for t in B["tools"]]
langs = {"en": [q["query"] for q in B["questions"]], "ko": [KO[q["id"]] for q in B["questions"]]}
m = SentenceTransformer("nvidia/Nemotron-3-Embed-1B-BF16", model_kwargs={"torch_dtype": torch.bfloat16}); m.eval()
E = encode(m, ["passage: " + t for t in tools], 256, 512, "library")
res = {}
for lg, qs in langs.items():
    Q = encode(m, ["query: " + q for q in qs], 256, 384, f"{lg} questions")
    s, j = (Q @ E.T).topk(TOPN, dim=1)
    res[lg] = {"first": {"idx": j.cpu().tolist(), "score": s.cpu().tolist()}, "rerank": {}}
del m; torch.cuda.empty_cache()
for name, path in specs:
    rr = Reranker(path, dtype=torch.bfloat16); rr.model.eval()
    for lg, qs in langs.items():
        pairs = [("", q, tools[i]) for qi, q in enumerate(qs) for i in res[lg]["first"]["idx"][qi]]
        flat = []
        with torch.no_grad():
            for b in range(0, len(pairs), 256):
                flat += rr.score(pairs[b:b + 256]).tolist()
        res[lg]["rerank"][name] = [flat[q * TOPN:(q + 1) * TOPN] for q in range(len(qs))]
        print("scored", name, lg, flush=True)
    del rr; torch.cuda.empty_cache()
    json.dump(res, open(OUT, "w"))          # after every reranker: a failure later must not lose the earlier scores
print("wrote", OUT)
