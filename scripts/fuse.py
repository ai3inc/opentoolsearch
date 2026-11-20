"""Fuse a first-stage run with a reranker run.

score = z(rerank) + w * z(first stage), both z-normalised per query over the reranked
candidates. Prints the leaderboard aggregate for each w, and a *cross-fitted* figure:
the test questions are split in two halves by a hash of their id; w is chosen on one half
and scored on the other, both ways, so no question is scored with a weight that was
tuned on it.

  python scripts/fuse.py FIRST_RUN.json RERANK_RUN.json [w ...]     (env TOPN = rerank depth)
"""
import hashlib, json, os, statistics, sys
from opentoolsearch.data import load_all_queries
from opentoolsearch.metrics import per_query, aggregate, fmt

first, rr = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
TOPN = int(os.environ.get("TOPN", 0))
SAVE = os.environ.get("SAVE")              # optional path to write the fused run for the first w
ws = [float(x) for x in sys.argv[3:]] or [0, 0.25, 0.5, 1]
queries = load_all_queries()
qrels = {t: {q["id"]: q["qrels"] for q in qs} for t, qs in queries.items()}


def z(m):
    v = list(m.values()); mu = statistics.mean(v); sd = statistics.pstdev(v) or 1.0
    return {k: (x - mu) / sd for k, x in m.items()}


def fused(w):
    out = {}
    for t in rr:
        out[t] = {}
        for q, d in rr[t].items():
            if TOPN:
                keep = sorted(d, key=lambda k: -first[t][q][k])[:TOPN]; d = {k: d[k] for k in keep}
            zr, zf = z(d), z({k: first[t][q][k] for k in d})
            out[t][q] = {k: zr[k] + w * zf[k] for k in d}
    return out


fold = lambda qid: int(hashlib.md5(qid.encode()).hexdigest(), 16) % 2
pq = {}
for i, w in enumerate(ws):
    run = fused(w)
    if SAVE and i == 0:
        json.dump(run, open(SAVE, "w"))
    pq[w] = {t: per_query(qrels[t], run[t]) for t in run}
    print(f"w={w}", fmt(aggregate(pq[w], "NDCG@10")), flush=True)

if len(ws) > 1:
    half = lambda w, f: {t: {q: m for q, m in d.items() if fold(q) == f} for t, d in pq[w].items()}
    chosen, mixed = {}, {t: {} for t in rr}
    for f in (0, 1):
        best = max(ws, key=lambda w: aggregate(half(w, 1 - f), "NDCG@10")["avg"])   # choose on the other half
        chosen[f] = best
        for t, d in half(best, f).items():
            mixed[t].update(d)
    board = {}
    for m in ("NDCG@10", "Recall@10", "Comp@10", "P@10"):
        a = aggregate(mixed, m)
        print(f"cross-fitted (w per half = {chosen})", fmt(a), flush=True)
        board[m] = {"Avg": a["avg"], "API": a["category"]["web"], "Code": a["category"]["code"],
                    "Customized": a["category"]["customized"], "pooled": a["micro"]}
    if os.environ.get("OUTJSON"):          # leaderboard-style numbers: metric -> category -> value
        json.dump({"weights": chosen, "metrics": board}, open(os.environ["OUTJSON"], "w"), indent=1)
