"""Score the dense tool-retrieval models (scripts/v7_dense_models.py output) on the three benchmarks.

  python scripts/v7_score_dense.py DIR_WITH_dense_*.json results/dense_tool_retrievers.json

Own set (English, Korean): nDCG@10 with interchangeable tools credited, as scripts/v5_score_hosted.py.
Public set: nDCG@10 / Recall@10 / Completeness@10 per question, macro-averaged over the 32 subsets, strict labels, as scripts/v4_score.py.
Each model ran with every query format its card documents; the format with the best English own-set nDCG@10 is the one reported
everywhere (so no model gets a format chosen on the benchmark it is shown on, except that one, which favours the competitors).
"""
import glob, json, math, os, statistics, sys, collections
D, OUT = sys.argv[1], sys.argv[2]
B = json.load(open("data/v3/bench.json")); P = json.load(open("data/v4/bench_public.json"))
EQ = {g: set(v) for g, v in B["equiv"].items()}
CASES = {c["id"]: c for c in P["cases"] if c["query"].strip() and c["gold"]}


def fill(ranked, gold):
    open_slots, hits = list(gold), []
    for r, d in enumerate(ranked):
        g = next((g for g in open_slots if d == g or d in EQ.get(g, ())), None)
        if g: open_slots.remove(g); hits.append(r)
    return hits


def own_metrics(rk):
    nd = rc = cp = 0
    for q in B["questions"]:
        gold, ranked = q["gold"], rk[q["id"]]
        idcg = sum(1 / math.log2(k + 2) for k in range(min(10, len(gold))))
        h = fill(ranked[:10], gold)
        nd += sum(1 / math.log2(r + 2) for r in h) / idcg
        rc += sum(r < 10 for r in h) / len(gold); cp += (sum(r < 10 for r in h) == len(gold))
    n = len(B["questions"])
    return {"ndcg10": round(100 * nd / n, 2), "recall10": round(100 * rc / n, 2), "comp10": round(100 * cp / n, 2)}


def per_q(ranked, gold):
    hits = [r for r, d in enumerate(ranked[:10]) if d in gold]
    idcg = sum(1 / math.log2(k + 2) for k in range(min(10, len(gold))))
    nd = sum(1 / math.log2(r + 2) for r in hits) / idcg if idcg else 0.0
    rec = len(set(ranked[:10]) & set(gold)) / len(gold)
    return nd, rec, float(rec == 1.0)


def pub_metrics(rk, ids=None):
    by = collections.defaultdict(list)
    for cid, c in CASES.items():
        if ids is not None and cid not in ids:
            continue
        by[c["task"]].append(per_q(rk.get(cid, []), c["gold"]))
    sub = [[statistics.mean(x[i] for x in v) for i in range(3)] for v in by.values()]
    return {k: round(100 * statistics.mean(s[i] for s in sub), 2) for i, k in enumerate(("ndcg10", "recall10", "comp10"))}


SAMPLE = set(json.load(open("data/v4/sample_s8_ids.json"))) if os.path.exists("data/v4/sample_s8_ids.json") else None
rows = []
for f in sorted(glob.glob(f"{D}/dense_*.json")):
    r = json.load(open(f)); name = os.path.basename(f)[6:-5]
    variants = {}
    for vn, v in r["variants"].items():
        variants[vn] = {"own_en": own_metrics(v["own_en"]), "own_ko": own_metrics(v["own_ko"]), "public": pub_metrics(v["public"]),
                        "public_s8": pub_metrics(v["public"], SAMPLE) if SAMPLE else None}
    best = max(variants, key=lambda k: variants[k]["own_en"]["ndcg10"])
    rows.append({"system": name, "weights": r["model"], "revision": r.get("revision"), "params_m": r["params_m"],
                 "latency_ms_p50": r["latency_ms_p50"], "format": best, **{k: variants[best][k] for k in ("own_en", "own_ko", "public", "public_s8")},
                 "all_formats": variants})
for x in rows:
    print(f'{x["system"]:36s} [{x["format"]:20s}] own EN {x["own_en"]["ndcg10"]:6.2f}  own KO {x["own_ko"]["ndcg10"]:6.2f}  public {x["public"]["ndcg10"]:6.2f}  '
          f'{x["params_m"]}M  {x["latency_ms_p50"]} ms', flush=True)
json.dump(rows, open(OUT, "w"), indent=1, ensure_ascii=False)
