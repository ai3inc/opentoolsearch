"""Score hosted tool-search systems on the 1,000-tool set (same scoring as everything else).

  python scripts/v5_score_hosted.py data/v3 OUT.json NAME=path.jsonl ...   (price in env: PRICES="name:in:out,...")
"""
import json, math, statistics, sys, os
D, OUT = sys.argv[1], sys.argv[2]
specs = [s.split("=", 1) for s in sys.argv[3:]]
B = json.load(open(f"{D}/bench.json"))
tools = {t["api_name"]: t["id"] for t in B["tools"]}
qmap = {q["id"]: q for q in B["questions"]}
EQ = {g: set(v) for g, v in B["equiv"].items()}
PRICES = {}
for part in os.environ.get("PRICES", "").split(","):
    if part:
        n, i, o = part.split(":"); PRICES[n] = (float(i), float(o))
KRW = 1400.0
def fill(ranked, gold):
    open_slots, hits = list(gold), []
    for r, d in enumerate(ranked):
        g = next((g for g in open_slots if d == g or d in EQ.get(g, ())), None)
        if g: open_slots.remove(g); hits.append(r)
    return hits
def metrics(rankings):
    out = {"ndcg10": 0, "recall10": 0, "comp10": 0}
    for qid, q in qmap.items():
        ranked, gold = rankings.get(qid, []), q["gold"]
        idcg = sum(1 / math.log2(k + 2) for k in range(min(10, len(gold))))
        h = fill(ranked[:10], gold)
        out["ndcg10"] += sum(1 / math.log2(r + 2) for r in h) / idcg
        out["recall10"] += sum(r < 10 for r in h) / len(gold)
        out["comp10"] += (sum(r < 10 for r in h) == len(gold))
    return {k: round(100 * v / len(qmap), 2) for k, v in out.items()}
res = []
for name, path in specs:
    rows = [json.loads(l) for l in open(path) if '"error"' not in l]
    rk, ret, usd, lat, no_search = {}, [], [], [], 0
    for r in rows:
        q = qmap.get(r["id"])
        if not q: continue
        ids = [tools[n.split(":")[-1]] for n in r["names"] if n.split(":")[-1] in tools]
        rk[r["id"]] = ids; ret.append(len(ids)); lat.append(r["latency"]); no_search += (r["searches"] == 0)
        u = r["usage"] or {}
        if name in PRICES:
            pi, po = PRICES[name]
            tin = (u.get("total_input_tokens") or u.get("input_tokens") or 0) + (u.get("total_tool_use_tokens") or 0)
            tout = (u.get("total_output_tokens") or u.get("output_tokens") or 0) + (u.get("total_thought_tokens") or 0)
            usd.append((tin * pi + tout * po) / 1e6)
    m = metrics(rk)
    res.append({"system": name, **m, "answered": len(rows), "avg_returned": round(statistics.mean(ret), 2) if ret else 0,
                "no_search": no_search, "usd_per_1k": round(1000 * statistics.mean(usd), 2) if usd else None,
                "latency_p50": round(statistics.median(lat), 2) if lat else None})
    print(json.dumps(res[-1]), flush=True)
if OUT: json.dump(res, open(OUT, "w"), indent=1)
