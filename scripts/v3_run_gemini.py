"""Run Gemini API server-side tool search over the head-to-head library.

Every tool is sent with defer_loading; the tools the model loads (tool_search_call
function_names, in order) are recorded with usage and latency. Resumable.

  GEMINI_API_KEY=... python scripts/v3_run_gemini.py data/v3/bench.json OUT.jsonl MODEL [limit]
"""
import json, os, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
B = json.load(open(sys.argv[1])); DST, MODEL = sys.argv[2], sys.argv[3]
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
KEY = os.environ["GEMINI_API_KEY"]
TOOLS = [{"type": "tool_search"}] + [{"type": "function", "name": t["api_name"], "description": t["description"],
                                      "parameters": t["parameters"], "defer_loading": True} for t in B["tools"]]
SYS = "You are an assistant with a large tool library. Use the tools to fulfil the user's request."

def call(q):
    body = json.dumps({"model": MODEL, "store": False, "system_instruction": SYS, "input": q["query"], "tools": TOOLS}).encode()
    err = None
    for attempt in range(4):
        t0 = time.time()
        try:
            req = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/interactions", data=body,
                                         headers={"x-goog-api-key": KEY, "Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            names, searches, called = [], 0, []
            for s in r.get("steps", []):
                if s.get("type") == "tool_search_call":
                    searches += 1
                    for n in s.get("arguments", {}).get("function_names", []):
                        if n not in names:
                            names.append(n)
                if s.get("type") == "function_call":
                    called.append(s.get("name"))
            return {"id": q["id"], "names": names, "searches": searches, "called": called, "status": r.get("status"),
                    "usage": r.get("usage"), "latency": round(time.time() - t0, 2)}
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code} {e.read().decode()[:300]}"
            if e.code in (400, 403, 404):
                break
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            err = str(e)[:200]; time.sleep(5 * (attempt + 1))
    return {"id": q["id"], "error": err}

done = {json.loads(l)["id"] for l in open(DST)} if os.path.exists(DST) else set()
todo = [q for q in B["questions"] if q["id"] not in done][:LIMIT]
print("todo", len(todo), "tools", len(TOOLS) - 1, flush=True)
lock, out, n = threading.Lock(), open(DST, "a"), [0, 0]
def work(q):
    r = call(q)
    with lock:
        n[0] += 1
        if "error" in r:
            n[1] += 1; print("ERR", r["error"][:200], flush=True)
        else:
            out.write(json.dumps(r) + "\n"); out.flush()
        if n[0] % 20 == 0: print(n[0], "fail", n[1], flush=True)
with ThreadPoolExecutor(int(os.environ.get("WORKERS", 4))) as ex:
    list(ex.map(work, todo))
print("finished", n[0], "fail", n[1], flush=True)
