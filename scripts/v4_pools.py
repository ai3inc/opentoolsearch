"""Per-source tool pools in ToolRet: for each subset, the tools that share an id prefix with its labelled tools."""
import json, re, collections
from opentoolsearch.data import load_tools, load_all_queries
tools = load_tools()
pre = lambda i: re.sub(r"_tool_\d+$", "", str(i)) if re.search(r"_tool_\d+$", str(i)) else re.sub(r"_?\d+$", "", str(i))
by = collections.defaultdict(list)
for t in tools:
    by[pre(t["id"])].append(str(t["id"]))
print("prefixes", len(by))
out = {}
for task, qs in load_all_queries().items():
    ps = collections.Counter(pre(g) for q in qs for g in q["qrels"])
    pool = sum(len(by[p]) for p in ps)
    gold = [len(q["qrels"]) for q in qs]
    out[task] = {"n_q": len(qs), "prefixes": dict(ps), "pool": pool, "avg_gold": round(sum(gold) / len(gold), 2)}
    print(f"{task:24} q={len(qs):5} pool={pool:6} avg_gold={out[task]['avg_gold']:5} prefixes={list(ps)[:3]}")
json.dump(out, open("data/v4_pools.json", "w"), indent=1)
