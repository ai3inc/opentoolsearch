"""Profile one CPU search: where the time goes (embed vs rerank), and how much the token
length of tool documents costs.  OTS_DEVICE=cpu python scripts/v3_time_cpu_profile.py data/v3/bench.json RERANKER N"""
import json, os, statistics, sys, time, torch
from sentence_transformers import SentenceTransformer
from opentoolsearch.rerank import Reranker
B = json.load(open(sys.argv[1])); N = int(sys.argv[4]) if len(sys.argv) > 4 else 5
dev = os.environ.get("OTS_DEVICE") or None
m = SentenceTransformer("nvidia/Nemotron-3-Embed-1B-BF16", model_kwargs={"torch_dtype": torch.float32}, device=dev); m.eval(); m.max_seq_length = 512
rr = Reranker(sys.argv[2], dtype=torch.float32, device=dev); rr.model.eval()
texts = [json.dumps({"name": t["api_name"], "description": t["description"], "parameters": t["parameters"]}) for t in B["tools"]]
t0 = time.time()
with torch.no_grad():
    E = m.encode(["passage: " + x for x in texts], batch_size=64, convert_to_tensor=True, normalize_embeddings=True).float()
print(json.dumps({"step": "index 1000 tools", "s": round(time.time() - t0, 1), "threads": torch.get_num_threads()}), flush=True)
q = B["questions"][0]["query"]
t0 = time.time(); qe = m.encode(["query: " + q], convert_to_tensor=True, normalize_embeddings=True).float(); te = time.time() - t0
s, j = (qe @ E.T).topk(50, dim=1)
print(json.dumps({"step": "embed one query", "s": round(te, 2)}), flush=True)
for k in (10, 25, 50):
    pairs = [("", q, texts[i]) for i in j[0].tolist()[:k]]
    t0 = time.time()
    with torch.no_grad():
        rr.score(pairs)
    print(json.dumps({"step": f"rerank {k}", "s": round(time.time() - t0, 2), "per_pair": round((time.time() - t0) / k, 3)}), flush=True)
for L in (2000, 6000):
    pairs = [("", q, texts[i][:L]) for i in j[0].tolist()[:N]]
    t0 = time.time()
    with torch.no_grad():
        rr.score(pairs)
    print(json.dumps({"step": f"rerank {N} pairs cut to {L} chars", "s": round(time.time() - t0, 2)}), flush=True)
