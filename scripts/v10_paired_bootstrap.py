"""Paired bootstrap of the nDCG@10 difference between two first stages behind the same reranker (scripts/v8_first_stage_rerank.py runs).

  python scripts/v10_paired_bootstrap.py data/v8 results/first_stage_paired_bootstrap.json

Own set: the 200 requests are resampled with replacement (same draw for both systems). Public: the questions of each of the 32
subsets are resampled and the subsets averaged (macro), as in the headline metric. 95% interval from 4,000 / 2,000 draws.
"""
import collections, json, math, random, statistics, sys
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


def own_pq(rk):
    out = []
    for q in B["questions"]:
        gold = q["gold"]; idcg = sum(1 / math.log2(k + 2) for k in range(min(10, len(gold))))
        out.append(sum(1 / math.log2(r + 2) for r in fill(rk[q["id"]][:10], gold)) / idcg)
    return out


def pub_pq(rk):
    by = collections.defaultdict(list)
    for cid, c in CASES.items():
        gold = c["gold"]; r = rk.get(cid, [])
        hits = [i for i, d in enumerate(r[:10]) if d in gold]
        idcg = sum(1 / math.log2(k + 2) for k in range(min(10, len(gold))))
        by[c["task"]].append(sum(1 / math.log2(i + 2) for i in hits) / idcg)
    return by


names = ["OpenToolSearch-Embed-0.6B", "Qwen3-Embedding-0.6B", "Nemotron-3-Embed-1B", "callbus-emb-8b"]
runs = {n: json.load(open(f"{D}/fs_{n}.json"))["final"] for n in names}
random.seed(0)


def own(a, b, key, n=4000):
    d = [i - j for i, j in zip(own_pq(runs[a][key]), own_pq(runs[b][key]))]
    ms = sorted(statistics.mean(random.choices(d, k=len(d))) for _ in range(n))
    return [round(100 * statistics.mean(d), 2), round(100 * ms[int(.025 * n)], 2), round(100 * ms[int(.975 * n)], 2)]


def pub(a, b, n=2000):
    x, y = pub_pq(runs[a]["public"]), pub_pq(runs[b]["public"])
    dd = {t: [i - j for i, j in zip(x[t], y[t])] for t in x}
    base = statistics.mean(statistics.mean(v) for v in dd.values())
    ms = sorted(statistics.mean(statistics.mean(random.choices(v, k=len(v))) for v in dd.values()) for _ in range(n))
    return [round(100 * base, 2), round(100 * ms[int(.025 * n)], 2), round(100 * ms[int(.975 * n)], 2)]


res = {}
for a, b in (("OpenToolSearch-Embed-0.6B", "Nemotron-3-Embed-1B"), ("OpenToolSearch-Embed-0.6B", "Qwen3-Embedding-0.6B"),
             ("callbus-emb-8b", "OpenToolSearch-Embed-0.6B"), ("Qwen3-Embedding-0.6B", "Nemotron-3-Embed-1B")):
    res[f"{a} minus {b}"] = {"public": pub(a, b), "own_en": own(a, b, "own_en"), "own_ko": own(a, b, "own_ko")}
    print(f"{a} - {b}:", {k: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]" for k, v in res[f'{a} minus {b}'].items()})
json.dump(res, open(OUT, "w"), indent=1)
