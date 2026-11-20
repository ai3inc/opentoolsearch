"""Head-to-head set for API comparisons: a 1,000-tool library and 200 dev questions.

Library = the labelled tools of the sampled OTS-MCP dev questions, every Toucan tool an LLM
judged interchangeable with one of them, and random other Toucan tools up to 1,000. Runs the
local systems (first stage, rerankers) and writes what the API runners need.

  python scripts/v3_bench_prep.py data/v2 data/v3 name=reranker_path [...]
"""
import json, os, random, re, sys, torch
from sentence_transformers import SentenceTransformer
from opentoolsearch.evaluate import encode
from opentoolsearch.rerank import Reranker

D, OUT, specs = sys.argv[1], sys.argv[2], [s.split("=", 1) for s in sys.argv[3:]]
os.makedirs(OUT, exist_ok=True)
random.seed(7)
N_Q, N_POOL, TOPN = 200, 1000, 50
pool_all = {p["id"]: p for p in (json.loads(l) for l in open(f"{D}/dev_pool.jsonl")) if p["src"] == "toucan"}
dev = [json.loads(l) for l in open(f"{D}/dev_questions.jsonl")]
eq = {}
for l in open(f"{D}/equiv.jsonl"):
    r = json.loads(l)
    if r["equiv"] and r["cand"] in pool_all:
        eq.setdefault(r["gold"], set()).add(r["cand"])
qs = random.sample(dev, N_Q)
keep = set()
for c in qs:
    for g in c["gold"]:
        keep.add(f"t:{g}"); keep |= eq.get(f"t:{g}", set())
rest = [i for i in pool_all if i not in keep]
random.shuffle(rest)
ids = sorted(keep) + rest[: max(0, N_POOL - len(keep))]
random.shuffle(ids)


def clean_schema(s, depth=0):
    """A conservative JSON-Schema subset that function-calling APIs accept."""
    if not isinstance(s, dict) or depth > 4:
        return {"type": "string"}
    t = s.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    if t not in ("object", "array", "string", "number", "integer", "boolean"):
        t = "object" if "properties" in s else "string"
    out = {"type": t}
    if isinstance(s.get("description"), str):
        out["description"] = s["description"][:300]
    if t == "object":
        props = s.get("properties") if isinstance(s.get("properties"), dict) else {}
        out["properties"] = {re.sub(r"[^A-Za-z0-9_]", "_", k)[:60] or "arg": clean_schema(v, depth + 1) for k, v in list(props.items())[:20]}
        req = [re.sub(r"[^A-Za-z0-9_]", "_", k)[:60] for k in s.get("required", []) if isinstance(k, str)]
        req = [k for k in req if k in out["properties"]]
        if req:
            out["required"] = req
    if t == "array":
        out["items"] = clean_schema(s.get("items", {}), depth + 1)
    return out


tools, used = [], set()
for i in ids:
    f = json.loads(pool_all[i]["text"])
    name = re.sub(r"[^A-Za-z0-9_-]", "_", f["name"])[:60].strip("_-") or "tool"
    if name[0].isdigit():
        name = "t_" + name
    base, k = name, 2
    while name in used:
        name = f"{base[:56]}_{k}"; k += 1
    used.add(name)
    desc = " ".join((f.get("description") or "").split())[:1000] or name
    tools.append({"id": i, "api_name": name, "description": desc, "parameters": clean_schema(f.get("parameters") or {}),
                  "text": pool_all[i]["text"]})
ix = {t["id"]: n for n, t in enumerate(tools)}
print("questions", len(qs), "library", len(tools), "labelled+interchangeable", len(keep), flush=True)

m = SentenceTransformer("nvidia/Nemotron-3-Embed-1B-BF16", model_kwargs={"torch_dtype": torch.bfloat16}); m.eval()
E = encode(m, ["passage: " + t["text"] for t in tools], 256, 512, "library")
Q = encode(m, ["query: " + c["query"] for c in qs], 256, 384, "questions")
s, j = (Q @ E.T).topk(TOPN, dim=1)
first = {"idx": j.cpu().tolist(), "score": s.cpu().tolist()}
del m; torch.cuda.empty_cache()
rr_scores = {}
for name, path in specs:
    rr = Reranker(path, dtype=torch.bfloat16); rr.model.eval()
    pairs = [("", c["query"], tools[i]["text"]) for qi, c in enumerate(qs) for i in first["idx"][qi]]
    flat = []
    with torch.no_grad():
        for b in range(0, len(pairs), 256):
            flat += rr.score(pairs[b:b + 256]).tolist()
    rr_scores[name] = [flat[q * TOPN:(q + 1) * TOPN] for q in range(len(qs))]
    del rr; torch.cuda.empty_cache(); print("scored", name, flush=True)
json.dump({"questions": [{"id": c["id"], "query": c["query"], "gold": [f"t:{g}" for g in c["gold"]]} for c in qs],
           "tools": [{k: t[k] for k in ("id", "api_name", "description", "parameters")} for t in tools],
           "equiv": {g: sorted(v) for g, v in eq.items()},
           "first": first, "rerank": rr_scores}, open(f"{OUT}/bench.json", "w"))
print("wrote", f"{OUT}/bench.json")
