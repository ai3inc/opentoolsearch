"""What it costs to keep the index current: update / add / remove / sync / save / load, measured with the real models.

  python scripts/v12_live_index.py OUT.json
Library of the 1,000 MCP tools of OTS-MCP-1k and its 200 requests (benchmarks/ots-mcp-1k/bench.json, or data/v3/bench.json).
"""
import json, os, random, statistics, sys, tempfile, time
import numpy as np
import torch
from opentoolsearch import ToolIndex
OUT = sys.argv[1]
B = json.load(open("benchmarks/ots-mcp-1k/bench.json" if os.path.exists("benchmarks/ots-mcp-1k/bench.json") else "data/v3/bench.json"))
tools = [{"name": t["api_name"], "description": t["description"], "parameters": t["parameters"]} for t in B["tools"]]
qs = [q["query"] for q in B["questions"]]
assert len({t["name"] for t in tools}) == len(tools)
sync = torch.cuda.synchronize
res = {"tools": len(tools), "gpu": torch.cuda.get_device_name(0)}
idx = ToolIndex(); idx.add(tools[:50])                       # warm the models
idx = ToolIndex.__new__(ToolIndex); idx.__dict__.update(ToolIndex(reranker=None).__dict__)   # embedder only is enough for index timings
calls = {"n": 0}; enc = idx.encoder; orig = enc.encode
def counted(texts, *a, **k):
    calls["n"] += len(texts); return orig(texts, *a, **k)
enc.encode = counted
def timed(fn):
    sync(); t0 = time.perf_counter(); r = fn(); sync(); return (time.perf_counter() - t0) * 1000, r

ms, _ = timed(lambda: idx.add(tools, batch_size=128)); res["build_1000_ms"] = round(ms)
calls["n"] = 0
rnd = random.Random(0)
# ---- change one tool (new description) / add one new tool / remove one, 30 times each
upd, add, rem = [], [], []
for i in range(30):
    t = dict(rnd.choice(tools)); t["description"] = t["description"] + " Also handles batch jobs."
    ms, _ = timed(lambda: idx.update([t])); upd.append(ms)
    n = {"name": f"new_tool_{i}", "description": "Convert a PDF invoice to structured JSON line items."}
    ms, _ = timed(lambda: idx.add([n])); add.append(ms)
    ms, _ = timed(lambda: idx.remove(f"new_tool_{i}")); rem.append(ms)
med = lambda v: round(statistics.median(v), 1)
res["update_one_ms_p50"], res["add_one_ms_p50"], res["remove_one_ms_p50"] = med(upd), med(add), med(rem)
res["texts_embedded_by_90_single_operations"] = calls["n"]          # 30 updates + 30 adds = 60
assert len(idx) == len(tools)
# ---- sync: 5 changed, 5 new, 5 gone, rest unchanged
now = [dict(t) for t in tools[5:]]
for t in now[:5]: t["description"] += " (v2)"
now += [{"name": f"fresh_{i}", "description": f"Brand new capability number {i} for scheduling."} for i in range(5)]
calls["n"] = 0; ms, r = timed(lambda: idx.sync(now)); res["sync_ms"], res["sync_counts"], res["sync_texts_embedded"] = round(ms), r, calls["n"]
# ---- the same library from scratch, for comparison
calls["n"] = 0; fresh = ToolIndex(reranker=None, encoder=enc); ms, _ = timed(lambda: fresh.add(now, batch_size=128)); res["rebuild_all_ms"] = round(ms)
# ---- incremental equals from-scratch (bi-encoder: each vector depends on its own text only)
order = {t["name"]: i for i, t in enumerate(fresh.tools)}
cos = [float(idx._emb[i] @ fresh._emb[order[t["name"]]]) for i, t in enumerate(idx.tools)]
res["incremental_vs_scratch_min_cosine"] = round(min(cos), 5)
agree = 0
for q in qs:
    a = [h["name"] for h in idx.search(q, k=5)]; b = [h["name"] for h in fresh.search(q, k=5)]; agree += (a == b)
res["incremental_vs_scratch_identical_top5_requests"] = f"{agree}/{len(qs)}"
# ---- save / load
with tempfile.TemporaryDirectory() as d:
    ms, _ = timed(lambda: idx.save(d)); res["save_ms"] = round(ms)
    res["saved_mb"] = round(sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 1e6, 1)
    calls["n"] = 0
    ms, back = timed(lambda: ToolIndex.load(d, reranker=None, encoder=enc)); res["load_ms"] = round(ms)
    same = sum([h["name"] for h in back.search(q, k=5)] == [h["name"] for h in idx.search(q, k=5)] for q in qs)
    res["load_same_top5_requests"] = f"{same}/{len(qs)}"
    res["texts_embedded_by_load"] = calls["n"] - len(qs) * 2          # minus the 400 query embeddings of the comparison
# ---- an edited tool is found by its new description at once
probe = {"name": "fresh_0", "description": "Translate a legal contract from Korean to English and flag risky clauses."}
idx.update([probe]); top = idx.search("Korean legal contract translation with risk flags", k=1)[0]["name"]
res["edited_tool_found_first_after_update"] = (top == "fresh_0")
print(json.dumps(res, indent=1)); json.dump(res, open(OUT, "w"), indent=1); print("done")
