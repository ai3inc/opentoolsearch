"""Score Jev and Gemini Decision (single-choice mode) on the Korean requests, over the first-stage candidates of data/v6/bench.json.

  python scripts/v6_score_decisions.py data/v6 results/korean_decisions.json
A candidate's score is the probability the decision model gave it; ties keep the first-stage order. Cost is what the gateway billed
(KRW, 1,400 per USD).
"""
import json, math, statistics, sys
D, OUT = sys.argv[1], sys.argv[2]
B = json.load(open(f"{D}/bench.json"))
EQ = {g: set(v) for g, v in B["equiv"].items()}
ID = [t["id"] for t in B["tools"]]
QI = {q["id"]: i for i, q in enumerate(B["questions"])}


def fill(ranked, gold):
    open_slots, hits = list(gold), []
    for r, d in enumerate(ranked):
        g = next((g for g in open_slots if d == g or d in EQ.get(g, ())), None)
        if g: open_slots.remove(g); hits.append(r)
    return hits


rows = []
for name, stem in (("Jev (single choice)", "jev_choice"), ("Gemini Decision (single choice)", "gdec_choice")):
    for k in (25, 50):
        rk, cost, lat = {}, [], []
        for l in open(f"{D}/{stem}_k{k}.jsonl"):
            r = json.loads(l); qi = QI[r["id"]]
            cand = B["first"]["idx"][qi][:k]
            order = sorted(range(len(cand)), key=lambda j: -r["scores"][j])
            rk[r["id"]] = [ID[cand[j]] for j in order]
            cost.append((r.get("usage") or {}).get("cost", 0)); lat.append(r["latency"])
        nd = rc = cp = 0
        for q in B["questions"]:
            gold, ranked = q["gold"], rk[q["id"]]
            idcg = sum(1 / math.log2(i + 2) for i in range(min(10, len(gold))))
            h = fill(ranked[:10], gold)
            nd += sum(1 / math.log2(x + 2) for x in h) / idcg; rc += sum(x < 10 for x in h) / len(gold); cp += (sum(x < 10 for x in h) == len(gold))
        n = len(B["questions"])
        rows.append({"system": name, "candidates": k, "answered": len(rk), "ndcg10": round(100 * nd / n, 2), "recall10": round(100 * rc / n, 2),
                     "comp10": round(100 * cp / n, 2), "usd_per_1k": round(1000 * statistics.mean(cost) / 1400, 3), "latency_p50": round(statistics.median(lat), 2)})
        print(json.dumps(rows[-1], ensure_ascii=False))
json.dump(rows, open(OUT, "w"), indent=1)
