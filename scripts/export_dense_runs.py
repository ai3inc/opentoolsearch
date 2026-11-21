"""Write the dense retrievers' top-20 rankings (ids only) next to the other raw runs, using the query format picked in
results/dense_tool_retrievers.json.   python scripts/export_dense_runs.py data/v7/dense
"""
import glob, json, os, sys
D = sys.argv[1]
chosen = {r["system"]: r["format"] for r in json.load(open("results/dense_tool_retrievers.json"))}
pub, own = {}, {}
for f in sorted(glob.glob(f"{D}/dense_*.json")):
    name = os.path.basename(f)[6:-5]; r = json.load(open(f)); v = r["variants"][chosen[name]]
    meta = {"weights": r["model"], "revision": r.get("revision"), "format": chosen[name], "pooling": r["pooling"]}
    pub[name] = {**meta, "public": v["public"]}
    own[name] = {**meta, "own_en": v["own_en"], "own_ko": v["own_ko"]}
json.dump(pub, open("benchmarks/toolret-per-source/dense_runs_ids.json", "w"))
json.dump(own, open("benchmarks/ots-mcp-1k/dense_runs.json", "w"))
print("wrote", len(pub), "models")
