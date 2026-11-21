"""OpenAI Responses API, hosted tool search, over the 1,000-tool library.
All tools deferred inside namespaces (10 functions each); the model searches itself.
  OPENAI_API_KEY=... python scripts/v5_run_openai.py data/v3/bench.json OUT.jsonl MODEL [limit]
"""
import json, os, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
B = json.load(open(sys.argv[1])); DST, MODEL = sys.argv[2], sys.argv[3]
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
KEY = os.environ["OPENAI_API_KEY"]
T = B["tools"]
NS = [{"type": "namespace", "name": f"tools_{i}", "description": f"Tool set {i}: MCP tools, alphabetically grouped.",
       "tools": [{"type": "function", "name": t["api_name"], "description": t["description"], "defer_loading": True,
                  "parameters": t["parameters"]} for t in T[i:i + 10]]} for i in range(0, len(T), 10)]
TOOLS = NS + [{"type": "tool_search"}]

def call(q):
    body = json.dumps({"model": MODEL, "input": q["query"], "tools": TOOLS, "parallel_tool_calls": False, "tool_choice": "required",
                       "instructions": "You are an assistant with a large tool library. Use the tools to fulfil the request."}).encode()
    err = None
    for attempt in range(4):
        t0 = time.time()
        try:
            req = urllib.request.Request("https://api.openai.com/v1/responses", data=body,
                                         headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            names, calls = [], 0
            for item in r.get("output", []):
                t = item.get("type")
                if t == "tool_search_call":
                    calls += 1
                elif t == "tool_search_output":
                    def walk(x):
                        if isinstance(x, dict):
                            if x.get("type") == "function" and x.get("name"): names.append(x["name"])
                            for v in x.values(): walk(v)
                        elif isinstance(x, list):
                            for v in x: walk(v)
                    walk(item.get("tools"))
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
