"""Gemini API server-side tool search on the public set (ToolRet, per-source pools).
Every tool of the question's source dataset is sent with defer_loading; name + a 240-character
short_description are what the model reads. Records the tools it loads. Resumable.

  GEMINI_API_KEY=... python scripts/v4_run_gemini.py data/v4/bench_public.json OUT.jsonl MODEL [limit]
"""
import json, os, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
B = json.load(open(sys.argv[1])); DST, MODEL = sys.argv[2], sys.argv[3]
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
KEY = os.environ["GEMINI_API_KEY"]
TOOLS = {t: [{"type": "tool_search"}] + [{"type": "function", "name": x["api_name"], "description": x["short"] or x["api_name"],
                                         "short_description": x["short"] or x["api_name"], "defer_loading": True,
                                         "parameters": {"type": "object", "properties": {}}} for x in p]
         for t, p in B["pools"].items()}
SYS = "You are an assistant with a large tool library. Use the tools to fulfil the user's request."

def call(c):
    body = json.dumps({"model": MODEL, "store": False, "system_instruction": SYS, "input": c["query"], "tools": TOOLS[c["task"]]}).encode()
    err = None
    for attempt in range(4):
        t0 = time.time()
        try:
            req = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/interactions", data=body,
                                         headers={"x-goog-api-key": KEY, "Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            names, searches = [], 0
            for s in r.get("steps", []):
                if s.get("type") == "tool_search_call":
                    searches += 1
                    for n in s.get("arguments", {}).get("function_names", []):
                        if n not in names:
                            names.append(n)
            return {"id": c["id"], "task": c["task"], "names": names, "searches": searches, "usage": r.get("usage"),
                    "latency": round(time.time() - t0, 2)}
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code} {e.read().decode()[:400]}"
            if e.code in (400, 403, 404):
                break
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            err = str(e)[:200]; time.sleep(5 * (attempt + 1))
    return {"id": c["id"], "task": c["task"], "error": err}

done = {json.loads(l)["id"] for l in open(DST)} if os.path.exists(DST) else set()
todo = [c for c in B["cases"] if c["id"] not in done][:LIMIT]
print("todo", len(todo), flush=True)
lock, out, n = threading.Lock(), open(DST, "a"), [0, 0]
def work(c):
    r = call(c)
    with lock:
        n[0] += 1
        if "error" in r:
            n[1] += 1; print("ERR", r["task"], r["error"][:260], flush=True)
        else:
            out.write(json.dumps(r) + "\n"); out.flush()
        if n[0] % 50 == 0: print(n[0], "fail", n[1], flush=True)
with ThreadPoolExecutor(int(os.environ.get("WORKERS", 5))) as ex:
    list(ex.map(work, todo))
print("finished", n[0], "fail", n[1], flush=True)
