"""Three-way fusion: z(rerank) + w1 z(first stage) + w2 z(BM25), over the reranked candidates.
BM25 scores come from a whole-corpus BM25 run; candidates it did not return get its minimum.
  python scripts/fuse3.py FIRST.json RERANK.json BM25.json w1 w2 [w1 w2 ...]
"""
import json, sys, statistics
from opentoolsearch.data import load_all_queries
from opentoolsearch.evaluate import score
from opentoolsearch.metrics import fmt
first, rr, bm = (json.load(open(p)) for p in sys.argv[1:4])
pairs = [(float(sys.argv[i]), float(sys.argv[i + 1])) for i in range(4, len(sys.argv), 2)]
queries = load_all_queries()
def z(m):
    v = list(m.values()); mu = statistics.mean(v); sd = statistics.pstdev(v) or 1.0
    return {k: (x - mu) / sd for k, x in m.items()}
for w1, w2 in pairs:
    out = {}
    for t in rr:
        out[t] = {}
        for q, d in rr[t].items():
            b = bm[t].get(q, {}); lo = min(b.values()) if b else 0.0
            zr, zf, zb = z(d), z({k: first[t][q][k] for k in d}), z({k: b.get(k, lo) for k in d})
            out[t][q] = {k: zr[k] + w1 * zf[k] + w2 * zb[k] for k in d}
    _, aggs = score(out, queries)
    print(f"w_first={w1} w_bm25={w2}", fmt(aggs["NDCG@10"]), flush=True)
