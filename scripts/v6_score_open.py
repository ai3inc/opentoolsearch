"""Score the open systems on English and Korean requests from the stored first-stage / reranker scores.

  python scripts/v6_score_open.py data/v3/bench.json data/v6/open_scores.json [OUT.json]
Each system is a ranking of the first-stage candidates: embedder alone, or reranker + w * first-stage, both z-scored over the
first `depth` candidates (the library defaults are depth 25, w 0.5). nDCG@10 / Recall@10 / Completeness@10 as in
scripts/v5_score_hosted.py: a tool the LLM judge found interchangeable with a labelled one counts.
"""
import json, math, statistics, sys
B = json.load(open(sys.argv[1])); S = json.load(open(sys.argv[2]))
ID = [t["id"] for t in B["tools"]]
EQ = {g: set(v) for g, v in B["equiv"].items()}


def fill(ranked, gold):
    open_slots, hits = list(gold), []
    for r, d in enumerate(ranked):
        g = next((g for g in open_slots if d == g or d in EQ.get(g, ())), None)
        if g: open_slots.remove(g); hits.append(r)
    return hits


def metrics(rankings):
    nd = rc = cp = 0
    for qi, q in enumerate(B["questions"]):
        gold, ranked = q["gold"], rankings[qi]
        idcg = sum(1 / math.log2(k + 2) for k in range(min(10, len(gold))))
        h = fill(ranked[:10], gold)
        nd += sum(1 / math.log2(r + 2) for r in h) / idcg
        rc += sum(r < 10 for r in h) / len(gold); cp += (sum(r < 10 for r in h) == len(gold))
    n = len(B["questions"])
    return {"ndcg10": round(100 * nd / n, 2), "recall10": round(100 * rc / n, 2), "comp10": round(100 * cp / n, 2)}


def z(v):
    mu = statistics.mean(v); sd = statistics.pstdev(v) or 1.0
    return [(x - mu) / sd for x in v]


def system(lg, reranker, depth, w):
    f = S[lg]["first"]; out = []
    for qi in range(len(B["questions"])):
        idx, fs = f["idx"][qi][:depth], f["score"][qi][:depth]
        if reranker is None:
            order = sorted(range(len(idx)), key=lambda k: -fs[k])
        else:
            sc = S[lg]["rerank"][reranker][qi][:depth]
            fused = [a + w * b for a, b in zip(z(sc), z(fs))]
            order = sorted(range(len(idx)), key=lambda k: -fused[k])
        out.append([ID[idx[k]] for k in order])
    return out


rows = []
for lg in ("en", "ko"):
    rows.append({"lang": lg, "system": "Nemotron-3-Embed-1B alone", **metrics(system(lg, None, 50, 0))})
    for rr, label in (("published", "OpenToolSearch"), ("base", "Qwen3-Reranker-0.6B as released")):
        if rr not in S[lg]["rerank"]: continue
        for depth, w in ((25, 0.5), (50, 0.25)):
            tag = " - default (depth 25)" if (depth, w) == (25, 0.5) and rr == "published" else f" - depth {depth}, fusion {w}"
            rows.append({"lang": lg, "system": label + tag, **metrics(system(lg, rr, depth, w))})
    # recall of the first stage: is a labelled or interchangeable tool inside the top 25 / 50 candidates at all?
    for k in (25, 50):
        hit = 0
        for qi, q in enumerate(B["questions"]):
            cand = {ID[i] for i in S[lg]["first"]["idx"][qi][:k]}
            hit += all(any(g == c or c in EQ.get(g, ()) for c in cand) for g in q["gold"])
        rows.append({"lang": lg, "system": f"first stage: all labelled tools inside top {k}", "ndcg10": None,
                     "share_of_questions": round(100 * hit / len(B["questions"]), 1)})
for r in rows:
    print(json.dumps(r, ensure_ascii=False))
if len(sys.argv) > 3:
    json.dump(rows, open(sys.argv[3], "w"), indent=1, ensure_ascii=False)
