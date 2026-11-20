"""Training files for round 2.

  train_v1h.jsonl  round-1 data minus every Toucan row that touches a held-out MCP server
  train_v2.jsonl   the same, with "negatives" removed when they are the same tool as a positive
                   under another name or provider (document-embedding cosine >= TAU)

  python scripts/v2_make_train.py data/v2 TAU
"""
import hashlib, json, sys, collections, torch
from sentence_transformers import SentenceTransformer
from opentoolsearch.evaluate import encode
D, TAU = sys.argv[1], float(sys.argv[2])
held = lambda name: int(hashlib.md5(name.encode()).hexdigest(), 16) % 5 == 0
u2s = json.load(open(f"{D}/uuid2servers.json"))
rows, dropped = [], 0
for l in open("data/mined_hard.jsonl"):
    r = json.loads(l)
    if r.get("src") == "toucan":
        s = u2s.get(r["id"][len("toucan_"):])
        if s is None or any(held(x) for x in s):
            dropped += 1; continue
    rows.append(r)
print("rows kept", len(rows), "| toucan rows dropped for held-out servers", dropped, flush=True)
with open(f"{D}/train_v1h.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
pool, ix = [], {}
for r in rows:
    for d in r["pos"] + r["neg"]:
        if d not in ix:
            ix[d] = len(pool); pool.append(d)
m = SentenceTransformer("nvidia/Nemotron-3-Embed-1B-BF16", model_kwargs={"torch_dtype": torch.bfloat16}); m.eval()
E = encode(m, ["passage: " + d for d in pool], 256, 512, "train pool")
stat = collections.defaultdict(lambda: [0, 0, 0])
with open(f"{D}/train_v2.jsonl", "w") as f:
    for r in rows:
        P = E[[ix[p] for p in r["pos"]]]; N = E[[ix[n] for n in r["neg"]]]
        mx = (N @ P.T).max(dim=1).values.tolist() if r["neg"] else []
        keep = [n for n, v in zip(r["neg"], mx) if v < TAU]
        st = stat[r.get("src", "?")]; st[0] += len(r["neg"]); st[1] += len(r["neg"]) - len(keep); st[2] += (len(keep) == 0)
        r2 = dict(r); r2["neg"] = keep
        if keep:
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")
for s, (a, b, c) in stat.items():
    print(f"{s}: negatives {a}, removed as same-tool {b} ({100*b/max(1,a):.1f}%), rows left with none {c}", flush=True)
