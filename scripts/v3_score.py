"""Score every system on the head-to-head set (200 questions, 1,000-tool library).

Credited scoring: a returned tool fills a labelled tool's slot if it is that tool or one an
LLM judged interchangeable with it; each slot counts once. Strict: only the labelled tool.

  python scripts/v3_score.py data/v3 [OUT.json]
"""
import json, math, os, statistics, sys
D = sys.argv[1]
B = json.load(open(f"{D}/bench.json"))
T, Q = B["tools"], B["questions"]
EQ = {g: set(v) for g, v in B["equiv"].items()}
name2id = {t["api_name"]: t["id"] for t in T}
qi_of = {q["id"]: i for i, q in enumerate(Q)}
# Gemini list prices, USD per 1M tokens (standard tier, Nov 2026): input, output
PRICE = {"gemini-3.6-flash": (0.75, 3.75), "gemini-4-hydrogen": (1.50, 7.50)}
KRW_PER_USD = 1400.0


def fill(ranked, gold, credit):
    """rank positions (0-based) at which each labelled slot gets filled"""
    open_slots, hits = list(gold), []
    for r, d in enumerate(ranked):
        g = next((g for g in open_slots if d == g or (credit and d in EQ.get(g, ()))), None)
        if g:
            open_slots.remove(g); hits.append(r)
    return hits


def metrics(rankings):
    """rankings: {question index: [tool ids, best first]} -> averaged metrics over ALL questions"""
    out = {"ndcg10_c": 0, "ndcg10_s": 0, "r5_c": 0, "r10_c": 0, "all_c": 0}
    for i, q in enumerate(Q):
        ranked, gold = rankings.get(i, []), q["gold"]
        idcg = sum(1 / math.log2(k + 2) for k in range(min(10, len(gold))))
        for credit, key in ((True, "ndcg10_c"), (False, "ndcg10_s")):
            out[key] += sum(1 / math.log2(r + 2) for r in fill(ranked[:10], gold, credit)) / idcg
        h = fill(ranked, gold, True)
        out["r5_c"] += sum(r < 5 for r in h) / len(gold)
        out["r10_c"] += sum(r < 10 for r in h) / len(gold)
        out["all_c"] += (sum(r < 10 for r in h) == len(gold))
    return {k: round(100 * v / len(Q), 2) for k, v in out.items()}


def z(v):
    mu = statistics.mean(v); sd = statistics.pstdev(v) or 1.0
    return [(x - mu) / sd for x in v]

rows = []
first = B["first"]
rows.append({"system": "Nemotron-3-Embed-1B alone (first stage)", **metrics({i: [T[k]["id"] for k in first["idx"][i]] for i in range(len(Q))})})
for name, w in (("base", 0.25), ("published", 0.25), ("clean", 0.25)):
    rk = {}
    for i in range(len(Q)):
        fused = [a + w * b for a, b in zip(z(B["rerank"][name][i]), z(first["score"][i]))]
        rk[i] = [T[k]["id"] for k, _ in sorted(zip(first["idx"][i], fused), key=lambda t: -t[1])]
    rows.append({"system": f"rerank:{name}", **metrics(rk)})
for fn, label in (("jev_noul", "Jev (yes/no per tool)"), ("jev_choice", "Jev (single choice)"),
                  ("gdec_noul", "Gemini Decision (yes/no per tool)"), ("gdec_choice", "Gemini Decision (single choice)")):
    p = f"{D}/{fn}.jsonl"
    if not os.path.exists(p):
        continue
    rk, cost, lat = {}, [], []
    for l in open(p):
        r = json.loads(l); i = qi_of[r["id"]]
        order = sorted(range(len(r["scores"])), key=lambda k: -r["scores"][k])      # stable: ties keep first-stage order
        rk[i] = [T[first["idx"][i][k]]["id"] for k in order]
        cost.append((r.get("usage") or {}).get("cost", 0)); lat.append(r["latency"])
    rows.append({"system": label, **metrics(rk), "answered": len(rk), "krw_per_1k": round(1000 * statistics.mean(cost)),
                 "latency_p50": round(statistics.median(lat), 2)})
for model in ("gemini-3.6-flash", "gemini-4-hydrogen"):
    p = f"{D}/{model}.jsonl"
    if not os.path.exists(p):
        continue
    rk, n_ret, usd, lat, nosearch, unknown = {}, [], [], [], 0, 0
    for l in open(p):
        r = json.loads(l); i = qi_of[r["id"]]
        ids = []
        for n in r["names"]:
            n = n.split(":")[-1]
            if n in name2id: ids.append(name2id[n])
            else: unknown += 1
        rk[i] = ids; n_ret.append(len(ids)); nosearch += (r["searches"] == 0); lat.append(r["latency"])
        u = r["usage"] or {}; pi, po = PRICE[model]
        usd.append(((u.get("total_input_tokens", 0) + u.get("total_tool_use_tokens", 0)) * pi +
                    (u.get("total_output_tokens", 0) + u.get("total_thought_tokens", 0)) * po) / 1e6)
    rows.append({"system": f"Gemini API tool search ({model})", **metrics(rk), "answered": len(rk),
                 "avg_returned": round(statistics.mean(n_ret), 2), "no_search": nosearch, "unknown_names": unknown,
                 "krw_per_1k": round(1000 * statistics.mean(usd) * KRW_PER_USD), "usd_per_1k": round(1000 * statistics.mean(usd), 2),
                 "latency_p50": round(statistics.median(lat), 2)})
for r in rows:
    print(json.dumps(r, ensure_ascii=False))
if len(sys.argv) > 2:
    json.dump({"questions": len(Q), "tools": len(T), "rows": rows}, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
