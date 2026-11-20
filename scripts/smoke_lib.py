"""Smoke test of the library path: index a small tool list, search with and without reranker."""
import json, sys, time
from opentoolsearch import ToolIndex
tools = json.load(open("examples/tools.json"))
emb, rr = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
t0 = time.time()
index = ToolIndex(model=emb, reranker=rr).add(tools)
print("indexed", len(index), "tools in", round(time.time() - t0, 1), "s; prefix-style:", index.q_prefix is not None)
for q in ["What's the weather in Seoul tomorrow?", "email the Q3 numbers to finance", "how many USD is 5,000,000 KRW",
          "open a bug for the login crash in our repo"]:
    hits = index.search(q, k=3)
    print(q, "->", [(h["name"], round(h["score"], 2)) for h in hits])
print("search_tool:", json.dumps(index.search_tool())[:160])
print("handle:", [t["name"] for t in index.handle({"query": "book a place to stay in Busan", "k": 2})])
print("allowed filter:", [h["name"] for h in index.search("send a message", k=3, allowed=["translate_text", "send_email"])])
print("catalogue line:", index.catalogue().splitlines()[0])
