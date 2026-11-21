"""Throughput when requests are served concurrently (batched across requests).

Closed loop: K requests, served in waves of C (a wave = the requests waiting at once).
Per wave: embed the queries as one batch -> top 50 each -> rerank all C x 50 pairs in
chunks -> fuse. Reports requests/s and cost per 1,000 search? no: per 1,000 searches at a
given USD/GPU-hour, for C = 1, 2, 4, 8, 16, 32.

  OTS_DEVICE=cuda python scripts/v3_serve_throughput.py data/v3/bench.json RERANKER OUT.json [K]
"""
import json, os, statistics, sys, time, torch
from sentence_transformers import SentenceTransformer
from opentoolsearch.rerank import Reranker
B = json.load(open(sys.argv[1])); RR = sys.argv[2]; DST = sys.argv[3]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 200
dev = os.environ.get("OTS_DEVICE") or None
cuda = torch.cuda.is_available() and dev != "cpu"
m = SentenceTransformer("nvidia/Nemotron-3-Embed-1B-BF16", model_kwargs={"torch_dtype": torch.float32 if not cuda else torch.bfloat16}, device=dev); m.eval(); m.max_seq_length = 512
rr = Reranker(RR, dtype=torch.float32 if not cuda else torch.bfloat16, device=dev); rr.model.eval()
texts = [json.dumps({"name": t["api_name"], "description": t["description"], "parameters": t["parameters"]}) for t in B["tools"]]
sync = torch.cuda.synchronize if cuda else (lambda: None)
t0 = time.time()
with torch.no_grad():
    E = m.encode(["passage: " + x for x in texts], batch_size=128 if cuda else 64, convert_to_tensor=True, normalize_embeddings=True).float()
sync(); index_s = time.time() - t0
z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
qs = [c["query"] for c in B["questions"][:K]]
res = {"K": K, "index_1000_tools_s": round(index_s, 1), "waves": {}}
with torch.no_grad():
    for C in (1, 2, 4, 8, 16, 32):
        for _ in range(1):   # warm-up wave
            (m.encode(["query: " + q for q in qs[:C]], convert_to_tensor=True, normalize_embeddings=True).float())
        t0 = time.time()
        for a in range(0, K, C):
            wave = qs[a:a + C]
            Q = m.encode(["query: " + q for q in wave], convert_to_tensor=True, normalize_embeddings=True).float()
            S, J = (Q @ E.T).topk(50, dim=1)
            pairs = [(("", wave[i], texts[J[i, k].item()])) for i in range(len(wave)) for k in range(J.shape[1])]
            flat = []
            for b in range(0, len(pairs), 256):
                flat += rr.score(pairs[b:b + 256]).tolist()
            for i in range(len(wave)):
                r = torch.tensor(flat[i * 50:(i + 1) * 50], device=S.device)
                _ = (z(r) + 0.25 * z(S[i])).argsort(descending=True)[:10].tolist()
        sync(); el = time.time() - t0
        rps = K / el
        res["waves"][str(C)] = {"wall_s": round(el, 2), "requests_per_s": round(rps, 2),
                                "ms_per_search": round(1000 / rps, 1),
                                "usd_per_1k_at_6.94": round(1000 / rps / 3600 * 6.94, 3),
                                "usd_per_1k_at_7.09": round(1000 / rps / 3600 * 7.09, 3),
                                "usd_per_1k_at_10.76": round(1000 / rps / 3600 * 10.76, 3)}
        print(json.dumps({str(C): res["waves"][str(C)]}), flush=True)
json.dump(res, open(DST, "w"), indent=1)
