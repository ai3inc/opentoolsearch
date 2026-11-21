"""Time OpenToolSearch one request at a time on the head-to-head set, end to end:
embed the request -> top 50 of the 1,000-tool library -> rerank 50 -> fuse.
  python scripts/v3_time_ours.py data/v3/bench.json RERANKER OUT.json
"""
import json, os, statistics, sys, time, torch
from sentence_transformers import SentenceTransformer
from opentoolsearch.rerank import Reranker
B = json.load(open(sys.argv[1]))
m = SentenceTransformer("nvidia/Nemotron-3-Embed-1B-BF16", model_kwargs={"torch_dtype": torch.bfloat16}, device=os.environ.get("OTS_DEVICE") or None); m.eval(); m.max_seq_length = 512
rr = Reranker(sys.argv[2], dtype=torch.bfloat16, device=os.environ.get("OTS_DEVICE") or None); rr.model.eval()
texts = [json.dumps({"name": t["api_name"], "description": t["description"], "parameters": t["parameters"]}) for t in B["tools"]]
t0 = time.time()
with torch.no_grad():
    E = m.encode(["passage: " + x for x in texts], batch_size=64 if not torch.cuda.is_available() else 128, convert_to_tensor=True, normalize_embeddings=True).float()
torch.cuda.synchronize() if torch.cuda.is_available() else None; index_s = time.time() - t0
lat, ntok = [], []
def z(v):
    return (v - v.mean()) / (v.std() + 1e-9)
with torch.no_grad():
    for k, q in enumerate(B["questions"][:5] + B["questions"]):          # 5 warm-up requests, not counted
        t0 = time.time()
        qe = m.encode(["query: " + q["query"]], convert_to_tensor=True, normalize_embeddings=True).float()
        s, j = (qe @ E.T).topk(50, dim=1)
        pairs = [("", q["query"], texts[i]) for i in j[0].tolist()]
        r = rr.score(pairs)
        fused = z(r) + 0.25 * z(s[0])
        top = fused.argsort(descending=True)[:10].tolist()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        if k >= 5:
            lat.append(time.time() - t0)
            ntok.append(sum(len(rr.tok.encode(p[1] + p[2], add_special_tokens=False)) for p in pairs))
lat.sort()
out = {"requests": len(lat), "index_1000_tools_s": round(index_s, 2), "p50_s": round(statistics.median(lat), 3),
       "mean_s": round(statistics.mean(lat), 3), "p95_s": round(lat[int(0.95 * len(lat))], 3),
       "rerank_tokens_per_search_mean": round(statistics.mean(ntok)), "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu (%d threads)" % torch.get_num_threads())}
print(json.dumps(out)); json.dump(out, open(sys.argv[3], "w"), indent=1)
