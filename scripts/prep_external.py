"""External (non-ToolRet) training pairs: real MCP-server tools from Toucan-1.5M and
function-calling dialogues from Glaive v2. Neither is one of ToolRet's 35 sources.
Rows that copy or nearly copy a ToolRet test question are dropped (same rule as
prep_train.py). Output: data/ext_raw.jsonl (no instruction yet).
"""
import json, os, random, re, sys
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from opentoolsearch.data import load_all_queries

OUT = sys.argv[1] if len(sys.argv) > 1 else "data"
N_TOUCAN = int(os.environ.get("N_TOUCAN", 80000))
N_GLAIVE = int(os.environ.get("N_GLAIVE", 40000))
random.seed(0)
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def render(fn, style):
    """One tool definition as JSON text, in one of several common layouts."""
    name, desc = fn.get("name", ""), fn.get("description", "")
    params = fn.get("parameters") or fn.get("input_schema") or {}
    if style == "openai":
        d = {"name": name, "description": desc, "parameters": params}
    elif style == "mcp":
        d = {"name": name, "description": desc, "input_schema": params}
    elif style == "flat":
        props = params.get("properties", {}) if isinstance(params, dict) else {}
        d = {"name": name, "description": desc,
             "parameters": {k: {kk: vv for kk, vv in v.items() if kk in ("type", "description", "default")}
                            for k, v in props.items() if isinstance(v, dict)}}
    else:  # wrapped
        d = {"type": "function", "function": {"name": name, "description": desc, "parameters": params}}
    return json.dumps(d, ensure_ascii=False)


STYLES = ["openai", "openai", "mcp", "flat", "wrapped"]
rows = []

# ---------- Toucan
seen_q = set()
for cfg, shards, n in (("Kimi-K2", 40, 4), ("OSS", 47, 4), ("Qwen3", 44, 4)):
    for i in random.sample(range(shards), n):
        ds = load_dataset("Agent-Ark/Toucan-1.5M", data_files=f"{cfg}/train-{i:05d}-of-{shards:05d}.parquet", split="train")
        for r in ds:
            q = (r["question"] or "").strip()
            if not q or q in seen_q or not r["target_tools"]:
                continue
            try:
                tools = [t["function"] for t in json.loads(r["available_tools"])]
            except Exception:
                continue
            targets = [t.strip() for t in r["target_tools"].split(",") if t.strip()]
            pos = [t for t in tools if any(t["name"] == x or t["name"].endswith("-" + x) or t["name"].endswith("_" + x) for x in targets)]
            if not pos or len(pos) != len(targets):
                continue
            neg = [t for t in tools if t not in pos]
            style = random.choice(STYLES)
            seen_q.add(q)
            rows.append({"id": f"toucan_{r['uuid']}", "query": q, "pos": [render(t, style) for t in pos],
                         "neg": [render(t, style) for t in random.sample(neg, min(15, len(neg)))],
                         "src": "toucan", "brief": "; ".join(f"{t['name']}: {(t.get('description') or '')[:120]}" for t in pos)})
        print(cfg, i, len(rows), flush=True)
random.shuffle(rows)
rows = rows[:N_TOUCAN]

# ---------- Glaive
g = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
dec = json.JSONDecoder()
grows = []
for k, r in enumerate(g):
    s = r["system"]
    fns, i = [], s.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(s[i:])
            if isinstance(obj, dict) and "name" in obj:
                fns.append(obj)
            i = s.find("{", i + end)
        except Exception:
            i = s.find("{", i + 1)
    m = re.search(r"USER:\s*(.*?)\n\n\nASSISTANT:\s*<functioncall>\s*\{\s*\"name\":\s*\"([^\"]+)\"", r["chat"], re.S)
    if not fns or not m:
        continue
    q, called = m.group(1).strip(), m.group(2)
    pos = [f for f in fns if f["name"] == called]
    if not pos or len(q) < 15 or q in seen_q:
        continue
    seen_q.add(q)
    style = random.choice(STYLES)
    grows.append({"id": f"glaive_{k}", "query": q, "pos": [render(pos[0], style)],
                  "neg": [render(f, style) for f in fns if f["name"] != called],
                  "src": "glaive", "brief": f"{pos[0]['name']}: {(pos[0].get('description') or '')[:120]}"})
random.shuffle(grows)
rows += grows[:N_GLAIVE]
print("before dedup", len(rows), flush=True)

# ---------- drop copies of test questions
test = [norm(q["query"]) for qs in load_all_queries().values() for q in qs]
tset, twords = set(test), [set(t.split()) for t in test]
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", model_kwargs={"torch_dtype": torch.bfloat16})
model.max_seq_length = 256
qn = [norm(r["query"]) for r in rows]
with torch.no_grad():
    Et = model.encode(test, batch_size=512, convert_to_tensor=True, normalize_embeddings=True).float()
    Eq = model.encode(qn, batch_size=1024, convert_to_tensor=True, normalize_embeddings=True).float()
    s, j = (Eq @ Et.T).topk(5, dim=1)
s, j = s.cpu(), j.cpu()
keep, dropped = [], 0
for i, r in enumerate(rows):
    w = set(qn[i].split())
    bad = qn[i] in tset or s[i, 0].item() >= 0.92 or any(
        len(w & twords[x]) / max(1, len(w | twords[x])) >= 0.60 for x in j[i].tolist())
    if bad:
        dropped += 1
    else:
        keep.append(r)
print("dropped as test near-copies", dropped, "kept", len(keep), flush=True)
with open(f"{OUT}/ext_raw.jsonl", "w") as f:
    for r in keep:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
import collections
print(collections.Counter(r["src"] for r in keep))
print("avg pos", sum(len(r["pos"]) for r in keep) / len(keep), "avg neg", sum(len(r["neg"]) for r in keep) / len(keep))
