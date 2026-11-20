"""Mine hard negatives for training rows with a retriever.

Pool = every distinct tool text in the given training files. For each row, take the
top-ranked pool tools that are not its positives and whose score is below
`perc` x (its weakest positive's score) - the "positive-aware" filter that removes
most unlabelled duplicates of the positive.

  python scripts/mine_negs.py --model runs/h1/final --out data/x_mined.jsonl data/a.jsonl data/b.jsonl
"""
import argparse, json, random
import torch
from sentence_transformers import SentenceTransformer
from opentoolsearch.evaluate import encode

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--perc", type=float, default=0.95)
ap.add_argument("--topk", type=int, default=60)
ap.add_argument("--keep", type=int, default=24)
ap.add_argument("--skip_top", type=int, default=0)
ap.add_argument("--template", default="instruct")
ap.add_argument("--q_prefix", default="")
ap.add_argument("--d_prefix", default="")
ap.add_argument("--only_src", default="", help="mine only for rows of this source (pool still uses all files)")
ap.add_argument("files", nargs="+")
a = ap.parse_args()

rows = [json.loads(l) for f in a.files for l in open(f)]
pool, idx = [], {}
for r in rows:
    for d in r["pos"] + r.get("neg", []):
        if d not in idx:
            idx[d] = len(pool); pool.append(d)
if a.only_src:
    rows = [r for r in rows if r.get("src") == a.only_src]
print("rows", len(rows), "pool", len(pool), flush=True)
model = SentenceTransformer(a.model, model_kwargs={"torch_dtype": torch.bfloat16}); model.eval()
D = encode(model, [a.d_prefix + d for d in pool], 256, 512, "pool")
if a.template == "concat":
    qs = [a.q_prefix + (f"{r['instruction']}\n{r['query']}" if r.get("instruction") else r["query"]) for r in rows]
else:
    qs = [f"Instruct: {r.get('instruction') or 'Given a web search query, retrieve relevant passages that answer the query'}\nQuery: {r['query']}" for r in rows]
Q = encode(model, qs, 512, 256, "queries")
random.seed(0)
n_kept = 0
with open(a.out, "w") as f:
    for i in range(0, len(rows), 4096):
        s = Q[i:i + 4096] @ D.T
        top_s, top_i = s.topk(a.topk, dim=1)
        for j, r in enumerate(rows[i:i + 4096]):
            pid = [idx[p] for p in r["pos"]]
            thr = a.perc * s[j, pid].min().item()
            pset = set(pid)
            cand = [(k, v) for k, v in zip(top_i[j].tolist()[a.skip_top:], top_s[j].tolist()[a.skip_top:])
                    if k not in pset and v < thr]
            mined = [pool[k] for k, _ in cand[: a.keep]]
            n_kept += len(mined)
            r2 = dict(r); r2["neg"] = mined + [d for d in r.get("neg", []) if d not in mined][: max(0, 8)]
            r2["n_mined"] = len(mined)
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")
print("avg mined per row", n_kept / len(rows), flush=True)
