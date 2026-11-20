"""Evaluate rerankers on the OTS-MCP dev set (unseen MCP servers).

Two settings: the request alone, and the request with an instruction written from the
request only (the writer never saw the answer tools). Two scorings: strict (only the
labelled tool counts) and equivalence-credited (a tool an LLM judged interchangeable with
a labelled tool fills that tool's slot; each slot counts once).

  python scripts/v2_eval_dev.py data/v2 name=path [name=path ...]
"""
import json, math, os, statistics, sys, torch
from sentence_transformers import SentenceTransformer
from opentoolsearch.evaluate import encode
from opentoolsearch.rerank import Reranker

D, specs = sys.argv[1], [s.split("=", 1) for s in sys.argv[2:]]
TOPN = int(os.environ.get("TOPN", 50))
pool = [json.loads(l) for l in open(f"{D}/dev_pool.jsonl")]
dev = [json.loads(l) for l in open(f"{D}/dev_questions.jsonl")]
inst = json.load(open(f"{D}/dev_inst.json")) if os.path.exists(f"{D}/dev_inst.json") else {}
eq = {}
if os.path.exists(f"{D}/equiv.jsonl"):
    for l in open(f"{D}/equiv.jsonl"):
        r = json.loads(l)
        if r["equiv"]:
            eq.setdefault(r["gold"], set()).add(r["cand"])
SET = {"alone": lambda c: c["query"], "inst": lambda c: (inst.get(c["id"], "") + "\n" + c["query"]).strip()}

first_path = f"{D}/dev_first_top{TOPN}.json"
if os.path.exists(first_path):
    first = json.load(open(first_path))
else:
    m = SentenceTransformer("nvidia/Nemotron-3-Embed-1B-BF16", model_kwargs={"torch_dtype": torch.bfloat16}); m.eval()
    E = torch.load(f"{D}/dev_pool_emb.pt").float().cuda() if os.path.exists(f"{D}/dev_pool_emb.pt") else \
        encode(m, ["passage: " + p["text"] for p in pool], 256, 512, "pool")
    first = {}
    for k, fn in SET.items():
        Q = encode(m, ["query: " + fn(c) for c in dev], 256, 384, k)
        s, j = (Q @ E.T).topk(TOPN, dim=1)
        first[k] = {"idx": j.cpu().tolist(), "score": s.cpu().tolist()}
    json.dump(first, open(first_path, "w")); del m, E; torch.cuda.empty_cache()


def ndcg(ranked, gold, credit):
    open_slots, dcg = list(gold), 0.0
    for r, d in enumerate(ranked[:10]):
        hit = next((g for g in open_slots if d == g or (credit and d in eq.get(g, ()))), None)
        if hit:
            open_slots.remove(hit); dcg += 1 / math.log2(r + 2)
    return dcg / sum(1 / math.log2(i + 2) for i in range(min(10, len(gold))))


def z(v):
    mu = statistics.mean(v); sd = statistics.pstdev(v) or 1.0
    return [(x - mu) / sd for x in v]


def report(name, k, rr_scores):
    out = []
    for w in ([None] if rr_scores is None else [0, 0.25, 0.5]):
        tot = [0.0, 0.0]
        for qi, c in enumerate(dev):
            ids = [pool[i]["id"] for i in first[k]["idx"][qi]]; fs = first[k]["score"][qi]
            if rr_scores is not None:
                fused = [a + w * b for a, b in zip(z(rr_scores[qi]), z(fs))]
                ids = [i for i, _ in sorted(zip(ids, fused), key=lambda t: -t[1])]
            gold = [f"t:{g}" for g in c["gold"]]
            tot[0] += ndcg(ids, gold, False); tot[1] += ndcg(ids, gold, True)
        out.append(f"{'first' if w is None else 'w=' + str(w)}: strict {100*tot[0]/len(dev):.2f} equiv {100*tot[1]/len(dev):.2f}")
    print(f"[{k:5}] {name:22} " + " | ".join(out), flush=True)


print("dev", len(dev), "pool", len(pool), "gold tools with an equivalent:", len(eq), flush=True)
for k in SET:
    report("first stage", k, None)
for name, path in specs:
    cache = f"{D}/dev_rr_{name}_top{TOPN}.json"
    if os.path.exists(cache):
        sc = json.load(open(cache))
    else:
        rr = Reranker(path, dtype=torch.bfloat16); rr.model.eval(); sc = {}
        for k, fn in SET.items():
            pairs = [((inst.get(c["id"], "") if k == "inst" else ""), c["query"], pool[i]["text"])
                     for qi, c in enumerate(dev) for i in first[k]["idx"][qi]]
            order = sorted(range(len(pairs)), key=lambda i: -len(pairs[i][2])); flat = [0.0] * len(pairs)
            with torch.no_grad():
                for b in range(0, len(order), 384):
                    ix = order[b:b + 384]
                    for i, v in zip(ix, rr.score([pairs[i] for i in ix]).tolist()):
                        flat[i] = v
            sc[k] = [flat[q * TOPN:(q + 1) * TOPN] for q in range(len(dev))]
        json.dump(sc, open(cache, "w")); del rr; torch.cuda.empty_cache()
    for k in SET:
        report(name, k, sc[k])
