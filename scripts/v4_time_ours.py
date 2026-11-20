"""Time OpenToolSearch one request at a time on the public set (ToolRet per-source pools).
  python scripts/v4_time_ours.py data/v4/bench_public.json RERANKER OUT.json
"""
import json, re, statistics, sys, time, collections, torch
from sentence_transformers import SentenceTransformer
from opentoolsearch.data import load_tools
from opentoolsearch.rerank import Reranker
B = json.load(open(sys.argv[1]))
doc = {str(t["id"]): t["documentation"] for t in load_tools()}
m = SentenceTransformer("nvidia/Nemotron-3-Embed-1B-BF16", model_kwargs={"torch_dtype": torch.bfloat16}); m.eval(); m.max_seq_length = 512
rr = Reranker(sys.argv[2], dtype=torch.bfloat16); rr.model.eval()
E, texts, idx_s = {}, {}, 0.0
with torch.no_grad():
    for task, pool in B["pools"].items():
        texts[task] = [doc[x["id"]] for x in pool]
        t0 = time.time()
        E[task] = m.encode(["passage: " + x for x in texts[task]], batch_size=128, convert_to_tensor=True, normalize_embeddings=True).float()
        torch.cuda.synchronize(); idx_s += time.time() - t0
z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
lat, ntok = [], []
with torch.no_grad():
    for k, c in enumerate(B["cases"][:5] + B["cases"]):
        t0 = time.time()
        qe = m.encode(["query: " + c["query"]], convert_to_tensor=True, normalize_embeddings=True).float()
        s, j = (qe @ E[c["task"]].T).topk(min(50, len(texts[c["task"]])), dim=1)
        pairs = [("", c["query"], texts[c["task"]][i]) for i in j[0].tolist()]
        r = rr.score(pairs); fused = z(r) + 0.25 * z(s[0]); _ = fused.argsort(descending=True)[:10].tolist()
        torch.cuda.synchronize()
        if k >= 5:
            lat.append(time.time() - t0); ntok.append(sum(len(rr.tok.encode(p[1] + p[2][:6000], add_special_tokens=False)) for p in pairs))
lat.sort()
out = {"requests": len(lat), "index_all_pools_s": round(idx_s, 1), "tools_indexed": sum(len(v) for v in texts.values()),
       "p50_s": round(statistics.median(lat), 3), "mean_s": round(statistics.mean(lat), 3), "p95_s": round(lat[int(0.95 * len(lat))], 3),
       "rerank_tokens_per_search_mean": round(statistics.mean(ntok)), "gpu": torch.cuda.get_device_name(0)}
print(json.dumps(out)); json.dump(out, open(sys.argv[3], "w"), indent=1)
