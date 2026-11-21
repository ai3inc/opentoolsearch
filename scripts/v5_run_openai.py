"""OpenAI Responses API, hosted tool search; the model reads the library and searches it itself.

Own set (data/v3/bench.json): all 1,000 tools as one flat list of deferred functions (full descriptions + schemas).
Public set (data/v4/bench_public.json): every tool of the question's source dataset, deferred, name + 240-character
short description, empty parameters, the same conditions as scripts/v4_run_gemini.py.
tool_choice is "required": without it the model skips the search for ~40% of the requests (measured).
  OPENAI_API_KEY=... python scripts/v5_run_openai.py BENCH.json OUT.jsonl MODEL [limit]
"""
import json, os, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
B = json.load(open(sys.argv[1])); DST, MODEL = sys.argv[2], sys.argv[3]
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
KEY = os.environ["OPENAI_API_KEY"]
PUBLIC = "pools" in B
# Flat list: the docs recommend namespaces of <=10, but with 1,000 tools in 100 namespaces the
# hosted search returns nothing for most questions (measured); flat deferred functions work.
if PUBLIC:
    # OpenAI rejects a function named like one of its reserved namespaces (rotbench has a tool called "history":
    # "Function 'history.history' is not allowed in reserved namespace 'history'"). Send that one as "history_" and map
    # the names the search returns back to the benchmark's own.
    safe = lambda n: n + "_" if n == "history" else n
    SAFE = {t: {safe(x["api_name"]): x["api_name"] for x in p} for t, p in B["pools"].items()}
    TOOLS_BY = {t: [{"type": "function", "name": safe(x["api_name"]), "description": x["short"] or x["api_name"], "defer_loading": True,
                     "parameters": {"type": "object", "properties": {}}} for x in p] + [{"type": "tool_search"}]
                for t, p in B["pools"].items()}
    QS = [c for c in B["cases"] if c["query"].strip() and c["gold"]]
else:
    TOOLS = [{"type": "function", "name": t["api_name"], "description": t["description"], "defer_loading": True,
              "parameters": t["parameters"]} for t in B["tools"]] + [{"type": "tool_search"}]
    QS = B["questions"]

def call(q):
    body = json.dumps({"model": MODEL, "input": q["query"], "tools": TOOLS_BY[q["task"]] if PUBLIC else TOOLS, "parallel_tool_calls": False, "tool_choice": "required",
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
            if PUBLIC: names = [SAFE[q["task"]].get(n, n) for n in names]
            return {"id": q["id"], **({"task": q["task"]} if PUBLIC else {}), "names": names, "searches": calls, "usage": r.get("usage"), "latency": round(time.time() - t0, 2)}
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code} {e.read().decode()[:300]}"
            if e.code in (400, 403, 404): break
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            err = str(e)[:200]; time.sleep(5 * (attempt + 1))
    return {"id": q["id"], "error": err}

done = {json.loads(l)["id"] for l in open(DST)} if os.path.exists(DST) else set()
todo = [q for q in QS if q["id"] not in done][:LIMIT]
print("todo", len(todo), "tools", sum(len(v) - 1 for v in TOOLS_BY.values()) if PUBLIC else len(TOOLS) - 1, flush=True)
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
