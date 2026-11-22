"""The default OpenToolSearch pair, end to end through the library, on OTS-MCP-1k.

  python scripts/v10_default_speed.py OUT.json [EMBEDDER [RERANKER]]

1. Quality: `ToolIndex(...)` with library defaults (rerank 25, fusion 0.5), English and Korean requests,
   ranked ids saved for scoring with scripts/v8_score.py's metric (interchangeable tools credited).
2. Single-request latency: one request at a time, p50/p90 of `index.search`, and of the embedder alone.
3. Throughput: closed loop of 200 requests served in waves of C requests (queries embedded as one batch,
   top 25 each, all C x 25 pairs reranked, fused) - the same procedure as scripts/v3_serve_throughput.py.
"""
import json, math, os, statistics, sys, time
import torch
from opentoolsearch import ToolIndex
from opentoolsearch.index import tool_text

OUT = sys.argv[1]
EMB = sys.argv[2] if len(sys.argv) > 2 else None
RRK = sys.argv[3] if len(sys.argv) > 3 else None
B = json.load(open("data/v3/bench.json"))
KO = {r["id"]: r["ko"] for r in json.load(open("data/v6/ko_queries.json"))}
tools = [{"name": t["api_name"], "description": t["description"], "parameters": t["parameters"]} for t in B["tools"]]
ids = [t["id"] for t in B["tools"]]
EQ = {g: set(v) for g, v in B["equiv"].items()}
qs = {"own_en": [(q["id"], q["query"]) for q in B["questions"]], "own_ko": [(q["id"], KO[q["id"]]) for q in B["questions"]]}
kw = {}
if EMB: kw["model"] = EMB
if RRK: kw["reranker"] = RRK
REV, QP = os.environ.get("OTS_REVISION"), os.environ.get("OTS_QPREFIX")      # another first stage: a revision, a query format ("\\n" = newline)
if REV or QP:
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(EMB, revision=REV, model_kwargs={"torch_dtype": torch.bfloat16}); enc.max_seq_length = 512
    kw["encoder"] = enc
    if QP: kw["query_prefix"] = QP.encode().decode("unicode_escape")
sync = torch.cuda.synchronize
t0 = time.time(); index = ToolIndex(**kw); load_s = time.time() - t0
t0 = time.time(); index.add(tools, batch_size=128); sync(); index_s = time.time() - t0
by_obj = {id(t): i for i, t in enumerate(index.tools)}
res = {"embedder": EMB or "default", "reranker": RRK or "default", "load_s": round(load_s, 1), "index_1000_tools_s": round(index_s, 2),
       "rerank_top": index.rerank_top, "fuse": index.fuse, "max_doc_tokens": index.encoder.max_seq_length,
       "embedding_dim": int(index._emb.shape[1]), "query_prefix": index.q_prefix, "doc_prefix": index.d_prefix}


def fill(ranked, gold):
    open_slots, hits = list(gold), []
    for r, d in enumerate(ranked):
        g = next((g for g in open_slots if d == g or d in EQ.get(g, ())), None)
        if g: open_slots.remove(g); hits.append(r)
    return hits


# ---- 1. quality
ranks = {}
for key, lst in qs.items():
    ranks[key] = {}
    for qid, q in lst:
        ranks[key][qid] = [ids[by_obj[id(h["tool"])]] for h in index.search(q, k=10)]
    nd = 0
    for q in B["questions"]:
        gold = q["gold"]; idcg = sum(1 / math.log2(k + 2) for k in range(min(10, len(gold))))
        nd += sum(1 / math.log2(r + 2) for r in fill(ranks[key][q["id"]], gold)) / idcg
    res[key + "_ndcg10"] = round(100 * nd / len(B["questions"]), 2)
    print(key, res[key + "_ndcg10"], flush=True)
res["ranks"] = ranks

# ---- 2. single-request latency (warm)
sample = [q for _, q in qs["own_en"]]            # all 200 requests, warm
for q in sample[:10]: index.search(q, k=5)
lat, lat_e = [], []
for q in sample:
    sync(); t0 = time.perf_counter(); index.search(q, k=5); sync(); lat.append((time.perf_counter() - t0) * 1000)
    sync(); t0 = time.perf_counter(); index.encoder.encode([index._query_text(q, None)], normalize_embeddings=True); sync(); lat_e.append((time.perf_counter() - t0) * 1000)
pct = lambda v, p: round(sorted(v)[int(p * (len(v) - 1))], 1)
res["latency_ms"] = {"search_p50": pct(lat, .5), "search_p90": pct(lat, .9), "search_mean": round(statistics.mean(lat), 1), "embed_query_p50": pct(lat_e, .5), "embed_query_p90": pct(lat_e, .9)}
print(res["latency_ms"], flush=True)

# ---- 3. throughput in waves
z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
texts = [tool_text(t) for t in index.tools]
E = torch.tensor(index._emb, device="cuda", dtype=torch.float32)
Q = [q for _, q in qs["own_en"]]; K = len(Q); res["waves"] = {}
N = index.rerank_top
with torch.no_grad():
    for C in (1, 8, 32):
        index.encoder.encode([index._query_text(q, None) for q in Q[:C]], convert_to_tensor=True, normalize_embeddings=True)
        sync(); t0 = time.time()
        for a in range(0, K, C):
            wave = Q[a:a + C]
            qe = index.encoder.encode([index._query_text(q, None) for q in wave], convert_to_tensor=True, normalize_embeddings=True).float()
            S, J = (qe @ E.T).topk(N, dim=1)
            pairs = [("", wave[i], texts[J[i, k].item()]) for i in range(len(wave)) for k in range(N)]
            flat = []
            for b in range(0, len(pairs), 256):
                flat += index.reranker.score(pairs[b:b + 256]).tolist()
            for i in range(len(wave)):
                r = torch.tensor(flat[i * N:(i + 1) * N], device=S.device)
                _ = (z(r) + index.fuse * z(S[i])).argsort(descending=True)[:10].tolist()
        sync(); el = time.time() - t0; rps = K / el
        res["waves"][str(C)] = {"wall_s": round(el, 2), "requests_per_s": round(rps, 2), "ms_per_search": round(1000 / rps, 1),
                                "searches_per_hour": int(rps * 3600), "usd_per_1k_at_7.09": round(1000 / rps / 3600 * 7.09, 3)}
        print(C, res["waves"][str(C)], flush=True)
res["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
res["gpu"] = torch.cuda.get_device_name(0)
json.dump(res, open(OUT, "w"), indent=1)
print("done", flush=True)
