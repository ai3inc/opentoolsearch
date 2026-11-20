"""Score every system on the public set: ToolRet, per-source pools, the benchmark's own labels.

nDCG@10 / Recall@10 / Completeness@10 per question (binary relevance, strict labels), averaged
per subset, then over the subsets (macro) and over all questions (pooled).

  python scripts/v4_score.py data/v4 [OUT.json]
"""
import json, math, os, statistics, sys, collections
D = sys.argv[1]
B = json.load(open(f"{D}/bench_public.json"))
CASES = {c["id"]: c for c in B["cases"] if c["query"].strip() and c["gold"]}     # one question has an empty request
POOL = B["pools"]
PRICE = {"gemini-3.6-flash": (0.75, 3.75), "gemini-4-hydrogen": (1.50, 7.50)}   # USD / 1M tokens, list, Nov 2026
KRW = 1400.0
BIG = {t for t, p in POOL.items() if len(p) > 200}


def per_q(ranked, gold):
    hits = [r for r, d in enumerate(ranked[:10]) if d in gold]
    idcg = sum(1 / math.log2(k + 2) for k in range(min(10, len(gold))))
    nd = sum(1 / math.log2(r + 2) for r in hits) / idcg if idcg else 0.0
    rec = len(set(ranked[:10]) & set(gold)) / len(gold) if gold else 0.0
    return nd, rec, float(rec == 1.0)


def agg(rankings, tasks=None):
    by = collections.defaultdict(list)
    for cid, c in CASES.items():
        if tasks is not None and c["task"] not in tasks:
            continue
        by[c["task"]].append(per_q(rankings.get(cid, []), c["gold"]))
    sub = {t: [statistics.mean(x[i] for x in v) for i in range(3)] for t, v in by.items()}
    allq = [x for v in by.values() for x in v]
    return {"ndcg10": round(100 * statistics.mean(s[0] for s in sub.values()), 2),
            "recall10": round(100 * statistics.mean(s[1] for s in sub.values()), 2),
            "comp10": round(100 * statistics.mean(s[2] for s in sub.values()), 2),
            "ndcg10_pooled": round(100 * statistics.mean(x[0] for x in allq), 2), "subsets": len(sub), "n": len(allq),
            "by_subset": {t: round(100 * s[0], 1) for t, s in sorted(sub.items())}}


def z(v):
    mu = statistics.mean(v); sd = statistics.pstdev(v) or 1.0
    return [(x - mu) / sd for x in v]

rows, RUNS = [], {}
def add(name, kind, rk, **extra):
    RUNS[name] = {cid: rk.get(cid, [])[:20] for cid in CASES}
    a = agg(rk)
    small = agg(rk, set(POOL) - BIG)
    rows.append({"system": name, "kind": kind, **{k: a[k] for k in ("ndcg10", "recall10", "comp10", "ndcg10_pooled", "n")},
                 "ndcg10_small_pools": small["ndcg10"], **extra, "by_subset": a["by_subset"]})

first = B["first"]
add("Nemotron-3-Embed-1B alone", "open baseline", {cid: [POOL[c["task"]][i]["id"] for i in first[cid]["idx"]] for cid, c in CASES.items()})
for name, label, kind in (("base", "Qwen3-Reranker-0.6B as released", "open baseline"), ("published", "OpenToolSearch", "open weights")):
    rk = {}
    for cid, c in CASES.items():
        sc = B["rerank"][name][cid]; idx = first[cid]["idx"][:len(sc)]; fs = first[cid]["score"][:len(sc)]
        fused = [a + 0.25 * b for a, b in zip(z(sc), z(fs))] if len(sc) > 1 else sc
        rk[cid] = [POOL[c["task"]][i]["id"] for i, _ in sorted(zip(idx, fused), key=lambda t: -t[1])]
    add(label, kind, rk)
for fn, label in (("jev_choice", "Jev (single choice)"), ("jev_noul", "Jev (yes/no per tool)"),
                  ("gdec_choice", "Gemini Decision (single choice)"), ("gdec_noul", "Gemini Decision (yes/no per tool)")):
    p = f"{D}/{fn}.jsonl"
    if not os.path.exists(p):
        continue
    rk, cost, lat = {}, [], []
    for l in open(p):
        r = json.loads(l); pool = POOL[r["task"]]
        order = sorted(range(len(r["scores"])), key=lambda k: -r["scores"][k])       # stable: ties keep the option order
        rk[r["id"]] = [pool[r["opt"][k]]["id"] for k in order]
        cost.append((r.get("usage") or {}).get("cost", 0)); lat.append(r["latency"])
    add(label, "hosted API", rk, answered=len(rk), usd_per_1k=round(1000 * statistics.mean(cost) / KRW, 2),
        latency_p50=round(statistics.median(lat), 2))
for model in ("gemini-3.6-flash", "gemini-4-hydrogen"):
    p = f"{D}/{model}.jsonl"
    if not os.path.exists(p):
        continue
    rk, n_ret, usd, lat, nosearch = {}, [], [], [], 0
    for l in open(p):
        r = json.loads(l); n2i = {x["api_name"]: x["id"] for x in POOL[r["task"]]}
        ids = [n2i[n.split(":")[-1]] for n in r["names"] if n.split(":")[-1] in n2i]
        rk[r["id"]] = ids; n_ret.append(len(ids)); nosearch += (r["searches"] == 0); lat.append(r["latency"])
        u = r["usage"] or {}; pi, po = PRICE[model]
        usd.append(((u.get("total_input_tokens", 0) + u.get("total_tool_use_tokens", 0)) * pi +
                    (u.get("total_output_tokens", 0) + u.get("total_thought_tokens", 0)) * po) / 1e6)
    add(f"Gemini API tool search ({model})", "hosted API", rk, answered=len(rk), avg_returned=round(statistics.mean(n_ret), 2),
        no_search=nosearch, usd_per_1k=round(1000 * statistics.mean(usd), 2), latency_p50=round(statistics.median(lat), 2))
for r in rows:
    print(json.dumps({k: v for k, v in r.items() if k != "by_subset"}, ensure_ascii=False))
if len(sys.argv) > 2:
    json.dump({"subsets": len(POOL), "questions": len(CASES), "big_pool_subsets": sorted(BIG), "rows": rows},
              open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
if len(sys.argv) > 3:      # ids only: questions, pools and each system's top-20, so anyone can rescore against the public labels
    json.dump({"cases": [{"task": c["task"], "id": c["id"]} for c in CASES.values()],
               "pools": {t: [x["id"] for x in p] for t, p in POOL.items()}, "runs": RUNS}, open(sys.argv[3], "w"))
