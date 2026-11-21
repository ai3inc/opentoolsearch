"""Score the first-stage + OpenToolSearch-reranker runs (scripts/v8_first_stage_rerank.py).

  python scripts/v8_score.py DIR_WITH_fs_*.json results/first_stage_plus_reranker.json

Per first stage: nDCG@10 of the first stage alone, of the reranker's default use on its top 25 (fusion 0.5), and of the
reranker alone on those 25; and the coverage of the candidate list - the share of questions whose every labelled tool
(or an interchangeable one on our own set) is inside the top 25 / top 50, which is the ceiling the reranker works under.
Metrics as scripts/v5_score_hosted.py (own set, interchangeable tools credited) and scripts/v4_score.py (public, macro over subsets).
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


def own_ndcg(rk):
    nd = 0
    for q in B["questions"]:
        gold = q["gold"]; idcg = sum(1 / math.log2(k + 2) for k in range(min(10, len(gold))))
        nd += sum(1 / math.log2(r + 2) for r in fill(rk[q["id"]][:10], gold)) / idcg
    return round(100 * nd / len(B["questions"]), 2)


def own_cover(rk, k):
    full = rec = 0
    for q in B["questions"]:
        h = len(fill(rk[q["id"]][:k], q["gold"])); full += (h == len(q["gold"])); rec += h / len(q["gold"])
    n = len(B["questions"]); return round(100 * full / n, 1), round(100 * rec / n, 1)


def pub_ndcg(rk):
    by = collections.defaultdict(list)
    for cid, c in CASES.items():
        gold = c["gold"]; r = rk.get(cid, [])
        hits = [i for i, d in enumerate(r[:10]) if d in gold]
        idcg = sum(1 / math.log2(k + 2) for k in range(min(10, len(gold))))
        by[c["task"]].append(sum(1 / math.log2(i + 2) for i in hits) / idcg)
    return round(100 * statistics.mean(statistics.mean(v) for v in by.values()), 2)


def pub_cover(rk, k):
    full, rec = collections.defaultdict(list), collections.defaultdict(list)
    for cid, c in CASES.items():
        g = set(c["gold"]); h = len(g & set(rk.get(cid, [])[:k]))
        full[c["task"]].append(float(h == len(g))); rec[c["task"]].append(h / len(g))
    m = lambda d: round(100 * statistics.mean(statistics.mean(v) for v in d.values()), 1)
    return m(full), m(rec)


rows = []
for f in sorted(glob.glob(f"{D}/fs_*.json")):
    r = json.load(open(f)); name = os.path.basename(f)[3:-5]
    row = {"first_stage": name, "weights": r["model"], "revision": r.get("revision"), "depth": r["depth"], "fusion": r["fusion"]}
    for label, part in (("alone", "first_top20"), ("with_reranker", "final"), ("reranker_only", "rerank_only")):
        row[label] = {"own_en": own_ndcg(r[part]["own_en"]), "own_ko": own_ndcg(r[part]["own_ko"]), "public": pub_ndcg(r[part]["public"])}
    cov = {}
    for k in (25, 50):
        cov[f"top{k}"] = {}
        for key in ("own_en", "own_ko"):
            full, rec = own_cover(r["first_top50"][key], k); cov[f"top{k}"][key] = {"all_found": full, "recall": rec}
        full, rec = pub_cover(r["first_top50"]["public"], k); cov[f"top{k}"]["public"] = {"all_found": full, "recall": rec}
    row["coverage"] = cov
    rows.append(row)
for x in sorted(rows, key=lambda x: -x["with_reranker"]["public"]):
    a, w, c = x["alone"], x["with_reranker"], x["coverage"]["top25"]
    print(f'{x["first_stage"]:22s} alone {a["public"]:5.1f}/{a["own_en"]:5.1f}/{a["own_ko"]:5.1f} | +reranker {w["public"]:5.1f}/{w["own_en"]:5.1f}/{w["own_ko"]:5.1f} | '
          f'all-in-top25 {c["public"]["all_found"]:5.1f}/{c["own_en"]["all_found"]:5.1f}/{c["own_ko"]["all_found"]:5.1f}', flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
