import json, collections
from datasets import load_dataset
ds = load_dataset("Agent-Ark/Toucan-1.5M", data_files="Kimi-K2/train-00000-of-00040.parquet", split="train")
srv = collections.Counter(); ntools = {}
for r in ds.select(range(4000)):
    md = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
    for s in md.get("mcp_servers", []):
        srv[s["server_name"]] += 1
        ntools[s["server_name"]] = len((s.get("remote_server_response") or {}).get("tools", []))
print("servers in 4000 rows:", len(srv)); print(srv.most_common(12))
print("tools per server (sample):", list(ntools.items())[:8])
r = ds[0]; md = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
print({k: str(v)[:150] for k, v in md.items() if k != "mcp_servers"})
print("row servers:", [s["server_name"] for s in md["mcp_servers"]], "| target:", r["target_tools"], "| subset:", r["subset_name"])
