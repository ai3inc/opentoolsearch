"""Score the second-stage combinations of scripts/v11_second_stage_combos.py.

  python scripts/v11_score.py data/v11 results/second_stage_combos.json

For each first stage, the four pre-registered scorings (z-scores over the 25 candidates, first-stage weight 0.5):
  A ours + first | B Tool-Rank | C Tool-Rank + first | D ours + Tool-Rank + first
Metrics as scripts/v8_score.py: own set (interchangeable tools credited), ToolRet per-source macro over 32 subsets (strict).
"""
import collections, glob, json, math, os, random, statistics, sys
D, OUT = sys.argv[1], sys.argv[2]
B = json.load(open("data/v3/bench.json")); P = json.load(open("data/v4/bench_public.json"))
EQ = {g: set(v) for g, v in B["equiv"].items()}
CASES = {c["id"]: c for c in P["cases"] if c["query"].strip() and c["gold"]}
W = 0.5


def z(v):
    mu = statistics.mean(v); sd = statistics.pstdev(v) or 1.0
    return [(x - mu) / sd for x in v]


VARIANTS = {"A": ("ours + first", lambda o, t, f: [a + W * c for a, c in zip(z(o), z(f))]),
            "B": ("Tool-Rank alone", lambda o, t, f: z(t)),
            "C": ("Tool-Rank + first", lambda o, t, f: [b + W * c for b, c in zip(z(t), z(f))]),
            "D": ("ours + Tool-Rank + first", lambda o, t, f: [a + b + W * c for a, b, c in zip(z(o), z(t), z(f))])}


def ranked(row, fn):
    if len(row["ids"]) < 2:
        return row["ids"]
    s = fn(row["ours"], row["toolrank"], row["first"])
    return [row["ids"][j] for j in sorted(range(len(s)), key=lambda j: -s[j])]


def fill(rk, gold):
    open_slots, hits = list(gold), []
    for r, d in enumerate(rk):
        g = next((g for g in open_slots if d == g or d in EQ.get(g, ())), None)
        if g: open_slots.remove(g); hits.append(r)
    return hits


def own_ndcg(rows, fn):
    nd = 0
    for q in B["questions"]:
        gold = q["gold"]; idcg = sum(1 / math.log2(k + 2) for k in range(min(10, len(gold))))
        nd += sum(1 / math.log2(r + 2) for r in fill(ranked(rows[q["id"]], fn)[:10], gold)) / idcg
    return round(100 * nd / len(B["questions"]), 2)


def pub_ndcg(rows, fn):
    by = collections.defaultdict(list)
    for cid, c in CASES.items():
        gold = c["gold"]; rk = ranked(rows[cid], fn)[:10] if cid in rows else []
        hits = [i for i, d in enumerate(rk) if d in gold]
        idcg = sum(1 / math.log2(k + 2) for k in range(min(10, len(gold))))
        by[c["task"]].append(sum(1 / math.log2(i + 2) for i in hits) / idcg)
    return round(100 * statistics.mean(statistics.mean(v) for v in by.values()), 2)


out = {}
for f in sorted(glob.glob(f"{D}/combo_*.json")):
    r = json.load(open(f)); name = os.path.basename(f)[6:-5]
    out[name] = {"first_stage": r["first_stage"], "depth": r["depth"], "ms_per_search_waves_of_8": r.get("ms_per_search_waves_of_8"), "variants": {}}
    for k, (lab, fn) in VARIANTS.items():
        out[name]["variants"][k] = {"label": lab, "public": pub_ndcg(r["public"], fn), "own_en": own_ndcg(r["own_en"], fn), "own_ko": own_ndcg(r["own_ko"], fn)}
    print(name)
    for k, v in out[name]["variants"].items():
        print(f'  {k} {v["label"]:26s} {v["public"]:5.1f} / {v["own_en"]:5.1f} / {v["own_ko"]:5.1f}')
json.dump(out, open(OUT, "w"), indent=1)


# ---- paired bootstrap (95%) of chosen differences: requests resampled (own), questions within each subset (public)
random.seed(0)
RAW = {os.path.basename(f)[6:-5]: json.load(open(f)) for f in sorted(glob.glob(f"{D}/combo_*.json"))}


def own_pq(rows, fn):
    res = []
    for q in B["questions"]:
        gold = q["gold"]; idcg = sum(1 / math.log2(k + 2) for k in range(min(10, len(gold))))
        res.append(sum(1 / math.log2(r + 2) for r in fill(ranked(rows[q["id"]], fn)[:10], gold)) / idcg)
    return res


def pub_pq(rows, fn):
    by = collections.defaultdict(list)
    for cid, c in CASES.items():
        gold = c["gold"]; rk = ranked(rows[cid], fn)[:10]
        hits = [i for i, d in enumerate(rk) if d in gold]
        idcg = sum(1 / math.log2(k + 2) for k in range(min(10, len(gold))))
        by[c["task"]].append(sum(1 / math.log2(i + 2) for i in hits) / idcg)
    return by


def diff(a, b):
    (fa, va), (fb, vb) = a, b; res = {}
    for key in ("own_en", "own_ko"):
        d = [x - y for x, y in zip(own_pq(RAW[fa][key], VARIANTS[va][1]), own_pq(RAW[fb][key], VARIANTS[vb][1]))]
        ms = sorted(statistics.mean(random.choices(d, k=len(d))) for _ in range(4000))
        res[key] = [round(100 * statistics.mean(d), 2), round(100 * ms[100], 2), round(100 * ms[3899], 2)]
    x, y = pub_pq(RAW[fa]["public"], VARIANTS[va][1]), pub_pq(RAW[fb]["public"], VARIANTS[vb][1])
    dd = {t: [i - j for i, j in zip(x[t], y[t])] for t in x}
    base = statistics.mean(statistics.mean(v) for v in dd.values())
    ms = sorted(statistics.mean(statistics.mean(random.choices(v, k=len(v))) for v in dd.values()) for _ in range(2000))
    res["public"] = [round(100 * base, 2), round(100 * ms[50], 2), round(100 * ms[1949], 2)]
    return res


O, T4 = "OpenToolSearch-Embed-0.6B", "Tool-Embed-4B"
PAIRS = {"Embed 0.6B: Tool-Rank alone minus ours": ((O, "B"), (O, "A")), "Embed 0.6B: ours + Tool-Rank minus ours": ((O, "D"), (O, "A")),
         "Embed 0.6B: ours + Tool-Rank minus Tool-Rank alone": ((O, "D"), (O, "B")),
         "Tool-Rank alone: Embed 0.6B minus Tool-Embed-4B": ((O, "B"), (T4, "B")),
         "Tool-DE (Tool-Embed-4B, Tool-Rank alone) minus OpenToolSearch default": ((T4, "B"), (O, "A"))}
out["paired_bootstrap"] = {}
for lab, (a, b) in PAIRS.items():
    out["paired_bootstrap"][lab] = diff(a, b)
    print(lab, {k: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]" for k, v in out["paired_bootstrap"][lab].items()})
json.dump(out, open(OUT, "w"), indent=1)
