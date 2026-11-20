"""OTS-MCP dev set: questions about MCP servers that training never sees.

Toucan-1.5M rows carry the MCP servers they were generated from. One fifth of the servers
(by a hash of the server name) are held out. Dev questions are rows whose servers are all
held out, taken from shards round 1 never read; the search pool is every Toucan tool we
have seen (held-out and not) plus 20,000 distractor tools from the training pool.

Also writes uuid -> servers for the round-1 rows so that training rows from held-out
servers can be removed.
"""
import hashlib, json, os, random, re, sys
from datasets import load_dataset

OUT = sys.argv[1] if len(sys.argv) > 1 else "data/v2"
os.makedirs(OUT, exist_ok=True)
R1 = {"Kimi-K2": (40, [24, 26, 2, 16]), "OSS": (47, [18, 44, 12, 17]), "Qwen3": (44, [37, 0, 16, 3])}
DEV = {"Kimi-K2": (40, [5, 30]), "OSS": (47, [1, 29]), "Qwen3": (44, [10])}
held = lambda name: int(hashlib.md5(name.encode()).hexdigest(), 16) % 5 == 0
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()
random.seed(20261120)


def servers_of(r):
    md = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
    return sorted({s["server_name"] for s in md.get("mcp_servers", []) if s.get("server_name")})


def load(cfg, shards, i, cols):
    return load_dataset("Agent-Ark/Toucan-1.5M", data_files=f"{cfg}/train-{i:05d}-of-{shards:05d}.parquet",
                        split="train", columns=cols)


tools, uuid2srv, all_servers = {}, {}, set()

def add_tools(r):
    try:
        fs = [t["function"] for t in json.loads(r["available_tools"])]
    except Exception:
        return []
    for f in fs:
        tools.setdefault(f["name"], f)
    return fs

# round-1 shards: which servers each training row came from (+ their tools for the pool)
for cfg, (n, idx) in R1.items():
    for i in idx:
        for r in load(cfg, n, i, ["uuid", "metadata", "available_tools"]):
            s = servers_of(r); uuid2srv[r["uuid"]] = s; all_servers.update(s); add_tools(r)
        print("r1", cfg, i, len(uuid2srv), "tools", len(tools), flush=True)
json.dump(uuid2srv, open(f"{OUT}/uuid2servers.json", "w"))

test_q = set()
try:
    from opentoolsearch.data import load_all_queries
    test_q = {norm(q["query"]) for qs in load_all_queries().values() for q in qs}
except Exception as e:
    print("no ToolRet test check:", e)

cands, seen_q = [], set()
for cfg, (n, idx) in DEV.items():
    for i in idx:
        for r in load(cfg, n, i, ["uuid", "metadata", "available_tools", "question", "target_tools", "subset_name"]):
            s = servers_of(r); all_servers.update(s); fs = add_tools(r)
            q = (r["question"] or "").strip()
            if not s or not all(held(x) for x in s) or len(q) < 20 or norm(q) in seen_q or norm(q) in test_q:
                continue
            targets = [t.strip() for t in (r["target_tools"] or "").split(",") if t.strip()]
            gold = [f["name"] for f in fs if any(f["name"] == x or f["name"].endswith("-" + x) or f["name"].endswith("_" + x) for x in targets)]
            if not gold or len(gold) != len(targets):
                continue
            seen_q.add(norm(q))
            cands.append({"id": f"dev_{r['uuid']}", "query": q, "gold": gold, "servers": s})
        print("dev", cfg, i, len(cands), flush=True)

random.shuffle(cands)
per, dev = {}, []
for c in cands:                                   # at most 12 questions per server combination
    k = "|".join(c["servers"])
    if per.get(k, 0) < 12 and len(dev) < 1500:
        per[k] = per.get(k, 0) + 1; dev.append(c)
print("dev questions", len(dev), "from", len(per), "server sets;", "held-out servers",
      sum(held(s) for s in all_servers), "of", len(all_servers), flush=True)

pool = [{"id": f"t:{n}", "name": n, "src": "toucan",
         "text": json.dumps({"name": n, "description": f.get("description", ""), "parameters": f.get("parameters", {})}, ensure_ascii=False)}
        for n, f in tools.items()]
distract, seen = [], set(p["text"] for p in pool)
for l in open("data/mined_hard.jsonl"):
    r = json.loads(l)
    if r.get("src") == "toucan":
        continue
    for d in r["pos"] + r["neg"][:3]:
        if d not in seen:
            seen.add(d); distract.append(d)
random.shuffle(distract)
pool += [{"id": f"d:{i}", "name": "", "src": "train-pool", "text": d} for i, d in enumerate(distract[:20000])]
with open(f"{OUT}/dev_questions.jsonl", "w") as f:
    for c in dev:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
with open(f"{OUT}/dev_pool.jsonl", "w") as f:
    for p in pool:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
json.dump(sorted(s for s in all_servers if held(s)), open(f"{OUT}/heldout_servers.json", "w"), ensure_ascii=False)
gold_tools = {g for c in dev for g in c["gold"]}
print("pool", len(pool), "(toucan tools", len(tools), ") distinct gold tools", len(gold_tools),
      "avg gold per question", sum(len(c["gold"]) for c in dev) / max(1, len(dev)))
