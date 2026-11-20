"""Public-benchmark head-to-head: ToolRet in its per-source setting.

Each question is searched only among the tools of its own source dataset (the setting of
Tables 10-11 of the ToolRet paper), with the benchmark's own labels. Subsets: every one whose
own tool pool has at most 2,000 tools (32 of 35; ToolBench, APIGen and ToolACE have 3,605 to
16,072). Questions: at most N_Q per subset, sampled with a fixed seed. Request only.

  python scripts/v4_public_prep.py OUT.json name=reranker_path [...]
"""
import json, random, re, sys, collections, torch
from sentence_transformers import SentenceTransformer
from opentoolsearch.data import load_tools, load_all_queries
from opentoolsearch.evaluate import encode
from opentoolsearch.rerank import Reranker

OUT, specs = sys.argv[1], [s.split("=", 1) for s in sys.argv[2:]]
MAX_POOL, N_Q, TOPF, TOPR = 2000, 25, 100, 50
random.seed(11)
pre = lambda i: re.sub(r"_tool_\d+$", "", str(i)) if re.search(r"_tool_\d+$", str(i)) else re.sub(r"_?\d+$", "", str(i))
by = collections.defaultdict(list)
for t in load_tools():
    by[pre(t["id"])].append(t)
NAME_KEYS = ("name", "api_name", "tool_name", "func_name", "function_name", "title")
DESC_KEYS = ("description", "func_description", "api_description", "tool_description", "desc", "functionality", "summary")


def declare(pool):
    out, used = [], set()
    for t in pool:
        doc = t["documentation"]
        try:
            d = json.loads(doc)
        except Exception:
            d = {}
        d = d if isinstance(d, dict) else {}
        nm = next((str(d[k]) for k in NAME_KEYS if isinstance(d.get(k), (str, int))), str(t["id"]))
        nm = re.sub(r"[^A-Za-z0-9_-]", "_", nm)[:56].strip("_-") or "tool"
        if nm[0].isdigit():
            nm = "t_" + nm
        base, k = nm, 2
        while nm in used:
            nm = f"{base[:52]}_{k}"; k += 1
        used.add(nm)
        ds = next((str(d[k]) for k in DESC_KEYS if isinstance(d.get(k), str) and d[k].strip()), doc)
        ds = " ".join(ds.split())
        out.append({"id": str(t["id"]), "api_name": nm, "short": ds[:240], "text": doc})
    return out


pools, cases = {}, []
for task, qs in load_all_queries().items():
    ps = sorted({pre(g) for q in qs for g in q["qrels"]})
    pool = [t for p in ps for t in by[p]]
    if len(pool) > MAX_POOL:
        print("skip", task, len(pool)); continue
    pools[task] = declare(pool)
    ids = {t["id"] for t in pools[task]}
    for q in random.sample(qs, min(N_Q, len(qs))):
        gold = [g for g in q["qrels"] if g in ids and q["qrels"][g] > 0]
        cases.append({"task": task, "id": q["id"], "query": q["query"], "gold": gold})
print("subsets", len(pools), "questions", len(cases), "tools", sum(len(p) for p in pools.values()), flush=True)

m = SentenceTransformer("nvidia/Nemotron-3-Embed-1B-BF16", model_kwargs={"torch_dtype": torch.bfloat16}); m.eval()
first = {}
for task, pool in pools.items():
    E = encode(m, ["passage: " + t["text"] for t in pool], 256, 512, task)
    cs = [c for c in cases if c["task"] == task]
    Q = encode(m, ["query: " + c["query"] for c in cs], 256, 384, task)
    s, j = (Q @ E.T).topk(min(TOPF, len(pool)), dim=1)
    for c, ss, jj in zip(cs, s.cpu().tolist(), j.cpu().tolist()):
        first[c["id"]] = {"idx": jj, "score": ss}
del m; torch.cuda.empty_cache()
rerank = {}
for name, path in specs:
    rr = Reranker(path, dtype=torch.bfloat16); rr.model.eval(); rerank[name] = {}
    pairs, span = [], []
    for c in cases:
        idx = first[c["id"]]["idx"][:TOPR]
        span.append((c["id"], len(idx)))
        pairs += [("", c["query"], pools[c["task"]][i]["text"]) for i in idx]
    order = sorted(range(len(pairs)), key=lambda i: -len(pairs[i][2])); flat = [0.0] * len(pairs)
    with torch.no_grad():
        for b in range(0, len(order), 256):
            ix = order[b:b + 256]
            for i, v in zip(ix, rr.score([pairs[i] for i in ix]).tolist()):
                flat[i] = v
    p = 0
    for cid, n in span:
        rerank[name][cid] = flat[p:p + n]; p += n
    del rr; torch.cuda.empty_cache(); print("scored", name, len(pairs), flush=True)
json.dump({"cases": cases, "pools": {t: [{k: x[k] for k in ("id", "api_name", "short")} for x in p] for t, p in pools.items()},
           "first": first, "rerank": rerank, "topr": TOPR}, open(OUT, "w"))
print("wrote", OUT)
