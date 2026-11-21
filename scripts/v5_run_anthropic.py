"""Anthropic Messages API, the built-in tool search tool (bm25), over the 1,000-tool library.
  ANTHROPIC_API_KEY=... python scripts/v5_run_anthropic.py data/v3/bench.json OUT.jsonl MODEL [limit]
"""
import json, os, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
B = json.load(open(sys.argv[1])); DST, MODEL = sys.argv[2], sys.argv[3]
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
KEY = os.environ["ANTHROPIC_API_KEY"]
T = B["tools"]
TOOLS = [{"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}] + \
        [{"name": t["api_name"], "description": t["description"], "input_schema": t["parameters"], "defer_loading": True} for t in T]

def call(q):
    body = json.dumps({"model": MODEL, "max_tokens": 1024, "system": "You are an assistant with a large tool library. Use the tools to fulfil the request.",
                       "messages": [{"role": "user", "content": q["query"]}], "tools": TOOLS}).encode()
    err = None
    for attempt in range(4):
        t0 = time.time()
        try:
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body,
                                         headers={"x-api-key": KEY, "anthropic-version": "2026-10-21", "Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            names, calls = [], 0
            for blk in r.get("content", []):
                if blk.get("type") == "server_tool_use" and "tool_search" in str(blk.get("name")):
                    calls += 1
                if blk.get("type") == "tool_search_tool_result":
                    for ref in (blk.get("content") or {}).get("tool_references", []) or []:
                        if ref.get("tool_name"): names.append(ref["tool_name"])
            return {"id": q["id"], "names": names, "searches": calls, "usage": r.get("usage"), "latency": round(time.time() - t0, 2)}
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code} {e.read().decode()[:300]}"
            if e.code in (400, 403, 404): break
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            err = str(e)[:200]; time.sleep(5 * (attempt + 1))
    return {"id": q["id"], "error": err}

done = {json.loads(l)["id"] for l in open(DST)} if os.path.exists(DST) else set()
todo = [q for q in B["questions"] if q["id"] not in done][:LIMIT]
print("todo", len(todo), "tools", len(T), flush=True)
lock, out, n = threading.Lock(), open(DST, "a"), [0, 0]
def work(q):
    r = call(q)
    with lock:
        n[0] += 1
        if "error" in r: n[1] += 1; print("ERR", r["error"][:250], flush=True)
        else: out.write(json.dumps(r) + "\n"); out.flush()
        if n[0] % 25 == 0: print(n[0], "fail", n[1], flush=True)
with ThreadPoolExecutor(int(os.environ.get("WORKERS", 4))) as ex:
    list(ex.map(work, todo))
print("finished", n[0], "fail", n[1], flush=True)
