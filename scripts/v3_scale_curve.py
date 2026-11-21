"""How throughput scales with what we do per search: rerank depth (candidates scored) and
rerank on/off. One B300, 1,000-tool library, 200 requests served in waves of 8.

  OTS_DEVICE=cuda python scripts/v3_scale_curve.py data/v3/bench.json RERANKER OUT.json
"""
import json, os, sys, time, torch
from sentence_transformers import SentenceTransformer
from opentoolsearch.rerank import Reranker
B = json.load(open(sys.argv[1])); RR, DST = sys.argv[2], sys.argv[3]
K, C = 200, 8
cuda = torch.cuda.is_available()
m = SentenceTransformer("nvidia/Nemotron-3-Embed-1B-BF16", model_kwargs={"torch_dtype": torch.bfloat16 if cuda else torch.float32}, device="cuda" if cuda else "cpu"); m.eval()
rr = Reranker(RR, dtype=torch.bfloat16 if cuda else torch.float32)
texts = [json.dumps({"name": t["api_name"], "description": t["description"], "parameters": t["parameters"]}) for t in B["tools"]]
sync = torch.cuda.synchronize if cuda else (lambda: None)
with torch.no_grad():
    E = m.encode(["passage: " + x for x in texts], batch_size=128, convert_to_tensor=True, normalize_embeddings=True).float()
sync()
qs = [c["query"] for c in B["questions"][:K]]
z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
out = {"device": torch.cuda.get_device_name(0) if cuda else "cpu", "K": K, "C": C, "library": len(texts), "points": {}}
with torch.no_grad():
    for depth in (0, 5, 10, 25, 50):
        for a in range(0, C, C):                                    # warm-up wave
            Q = m.encode(["query: " + q for q in qs[a:a + C]], convert_to_tensor=True, normalize_embeddings=True).float()
            S, J = (Q @ E.T).topk(50, dim=1)
        t0 = time.time()
        for a in range(0, K, C):
            wave = qs[a:a + C]
            Q = m.encode(["query: " + q for q in wave], convert_to_tensor=True, normalize_embeddings=True).float()
            S, J = (Q @ E.T).topk(50, dim=1)
            if depth:
                pairs = [("", wave[i], texts[J[i, k].item()]) for i in range(len(wave)) for k in range(depth)]
                flat = []
                for b in range(0, len(pairs), 256):
                    flat += rr.score(pairs[b:b + 256]).tolist()
                for i in range(len(wave)):
                    r = torch.tensor(flat[i * depth:(i + 1) * depth], device=S.device)
                    _ = (z(r) + 0.25 * z(S[i][:depth])).argsort(descending=True)[:10].tolist()
            else:
                for i in range(len(wave)):
                    _ = S[i].argsort(descending=True)[:10].tolist()
        sync(); el = time.time() - t0
        rps = K / el
        out["points"][str(depth)] = {"rerank_depth": depth, "ms_per_search": round(1000 / rps, 1),
                                      "searches_per_s": round(rps, 2), "searches_per_hour": round(rps * 3600),
                                      "usd_per_1k_at_7.09": round(1000 / rps / 3600 * 7.09, 3),
                                      "usd_per_1k_at_6.94": round(1000 / rps / 3600 * 6.94, 3)}
        print(json.dumps({str(depth): out["points"][str(depth)]}), flush=True)
json.dump(out, open(DST, "w"), indent=1)
