"""Choose the fusion weight on held-out *training* questions, not on the test set.

Dev = data/toolret_dev.jsonl (2,000 ToolRet-train rows never trained on). Pool = every
distinct tool text in the given training files plus the dev rows' tools. First stage ->
top-N -> reranker -> nDCG@10 for a grid of weights w in  z(rerank) + w * z(first).

  python scripts/dev_weight.py --embed MODEL --rerank RERANKER [--template concat ...] data/mined_hard.jsonl
"""
import argparse, json, math, statistics
import torch
from sentence_transformers import SentenceTransformer
from opentoolsearch.evaluate import encode
from opentoolsearch.rerank import Reranker

ap = argparse.ArgumentParser()
ap.add_argument("--embed", required=True); ap.add_argument("--rerank", required=True)
ap.add_argument("--dev", default="data/toolret_dev.jsonl")
ap.add_argument("--template", default="concat"); ap.add_argument("--q_prefix", default="query: ")
ap.add_argument("--d_prefix", default="passage: "); ap.add_argument("--topn", type=int, default=50)
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--grid", default="0,0.1,0.25,0.5,0.75,1,2")
ap.add_argument("files", nargs="+")
a = ap.parse_args()

dev = [json.loads(l) for l in open(a.dev)][: a.n]
pool, idx = [], {}
def add(d):
    if d not in idx:
        idx[d] = len(pool); pool.append(d)
for f in a.files:
    for l in open(f):
        r = json.loads(l)
        for d in r["pos"] + r.get("neg", []):
            add(d)
for r in dev:
    for d in r["pos"] + r.get("neg", []):
        add(d)
print("dev", len(dev), "pool", len(pool), flush=True)
emb = SentenceTransformer(a.embed, model_kwargs={"torch_dtype": torch.bfloat16}); emb.eval()
D = encode(emb, [a.d_prefix + d for d in pool], 256, 512, "pool")
if a.template == "concat":
    qs = [a.q_prefix + f"{r['instruction']}\n{r['query']}" for r in dev]
else:
    qs = [f"Instruct: {r['instruction']}\nQuery: {r['query']}" for r in dev]
Q = encode(emb, qs, 256, 256, "dev queries")
top_s, top_i = (Q @ D.T).topk(a.topn, dim=1)
top_s, top_i = top_s.cpu().tolist(), top_i.cpu().tolist()
del emb, D, Q; torch.cuda.empty_cache()
rr = Reranker(a.rerank, dtype=torch.bfloat16); rr.model.eval()
pairs = [(r["instruction"], r["query"], pool[k]) for r, ks in zip(dev, top_i) for k in ks]
scores = []
with torch.no_grad():
    for b in range(0, len(pairs), 256):
        scores += rr.score(pairs[b:b + 256]).tolist()

def ndcg10(ranked, gold):
    dcg = sum(1 / math.log2(i + 2) for i, k in enumerate(ranked[:10]) if k in gold)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(10, len(gold))))
    return dcg / idcg if idcg else 0.0

def z(v):
    mu = statistics.mean(v); sd = statistics.pstdev(v) or 1.0
    return [(x - mu) / sd for x in v]

best = None
for w in [float(x) for x in a.grid.split(",")]:
    tot = 0.0
    for qi, r in enumerate(dev):
        gold = {idx[p] for p in r["pos"]}
        ks = top_i[qi]; zr = z(scores[qi * a.topn:(qi + 1) * a.topn]); zf = z(top_s[qi])
        ranked = [k for k, _ in sorted(zip(ks, [x + w * y for x, y in zip(zr, zf)]), key=lambda t: -t[1])]
        tot += ndcg10(ranked, gold)
    m = tot / len(dev)
    print(f"w={w} dev nDCG@10 {100*m:.2f}", flush=True)
    if best is None or m > best[1]:
        best = (w, m)
first_only = sum(ndcg10(top_i[qi], {idx[p] for p in r['pos']}) for qi, r in enumerate(dev)) / len(dev)
print(f"first stage only {100*first_only:.2f} | best w={best[0]} ({100*best[1]:.2f})", flush=True)
