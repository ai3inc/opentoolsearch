"""CPU timing on a fixed sample of the head-to-head requests (the full 200 would take hours
on CPU). Same end-to-end steps as v3_time_ours.py.  OTS_DEVICE=cpu python scripts/v3_time_cpu_sample.py data/v3/bench.json RERANKER N OUT.json"""
import json, os, statistics, sys, time, torch
from sentence_transformers import SentenceTransformer
from opentoolsearch.rerank import Reranker
B = json.load(open(sys.argv[1])); N = int(sys.argv[3]); out = sys.argv[4]
dev = os.environ.get("OTS_DEVICE") or None
m = SentenceTransformer("nvidia/Nemotron-3-Embed-1B-BF16", model_kwargs={"torch_dtype": torch.float32}, device=dev); m.eval(); m.max_seq_length = 512
rr = Reranker(sys.argv[2], dtype=torch.float32, device=dev); rr.model.eval()
texts = [json.dumps({"name": t["api_name"], "description": t["description"], "parameters": t["parameters"]}) for t in B["tools"]]
t0 = time.time()
with torch.no_grad():
    E = m.encode(["passage: " + x for x in texts], batch_size=64, convert_to_tensor=True, normalize_embeddings=True).float()
index_s = time.time() - t0
z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
lat, nq = [], []
with torch.no_grad():
    for c in B["questions"][:N]:
        t0 = time.time()
        qe = m.encode(["query: " + c["query"]], convert_to_tensor=True, normalize_embeddings=True).float()
        s, j = (qe @ E.T).topk(50, dim=1)
        pairs = [("", c["query"], texts[i]) for i in j[0].tolist()]
        r = rr.score(pairs); _ = (z(r) + 0.25 * z(s[0])).argsort(descending=True)[:10].tolist()
        lat.append(time.time() - t0); nq.append(sum(len(rr.tok.encode(p[1] + p[2], add_special_tokens=False)) for p in pairs))
lat.sort()
res = {"requests": N, "index_1000_tools_s": round(index_s, 1), "p50_s": round(statistics.median(lat), 2), "mean_s": round(statistics.mean(lat), 2),
       "p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2), "rerank_tokens_per_search_mean": round(statistics.mean(nq)),
       "cpu": "8 vCPU (cpu_8c_16g), fp32, 8 threads"}
print(json.dumps(res)); json.dump(res, open(out, "w"), indent=1)
