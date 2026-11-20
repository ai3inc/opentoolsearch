"""Decision models (BizRouter /v1/decisions: Jev, Gemini Decision) on the public set.

Options = every tool of the question's source dataset when it has at most 200 tools; otherwise
the top 100 of our first stage (noted in the output). Modes: choice (one question, pick the
tool) or noul (one yes/no question per tool).

  BIZROUTER_API_KEY=... python scripts/v4_run_decisions.py data/v4/bench_public.json OUT.jsonl MODEL MODE [limit]
"""
import json, os, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
B = json.load(open(sys.argv[1])); DST, MODEL, MODE = sys.argv[2], sys.argv[3], sys.argv[4]
LIMIT = int(sys.argv[5]) if len(sys.argv) > 5 else None
KEY = os.environ["BIZROUTER_API_KEY"]; FULL = 200

def options(c):
    pool = B["pools"][c["task"]]
    return list(range(len(pool))) if len(pool) <= FULL else B["first"][c["id"]]["idx"][:100]

def call(c):
    pool, opt = B["pools"][c["task"]], options(c)
    if MODE == "choice":
        qs = {"tool": {"type": "choice", "instructions": "Select the tool that is needed to fulfil the user's request.",
                       "criteria": {pool[i]["api_name"]: (pool[i]["short"] or pool[i]["api_name"]) for i in opt}}}
    else:
        qs = {f"t{k}": {"type": "noul", "instructions": "Is the following tool needed to fulfil the user's request? "
                        f"Tool `{pool[i]['api_name']}`: {pool[i]['short']}"} for k, i in enumerate(opt)}
    body = json.dumps({"model": MODEL, "state": c["query"], "questions": qs}).encode()
    err = None
    for attempt in range(4):
        t0 = time.time()
        try:
            req = urllib.request.Request("https://api.bizrouter.ai/v1/decisions", data=body,
                                         headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json",
                                                  "User-Agent": "opentoolsearch-bench/0.1"})
            a = json.loads(urllib.request.urlopen(req, timeout=300).read()); ans = a["answers"]
            if MODE == "choice":
                p = ans["tool"].get("probabilities", {})
                scores = [float(p.get(pool[i]["api_name"], 0.0)) for i in opt]
            else:
                scores = [float(ans[f"t{k}"]["noul"]) for k in range(len(opt))]
            return {"id": c["id"], "task": c["task"], "opt": opt, "scores": scores, "full_pool": len(pool) <= FULL,
                    "usage": a.get("usage"), "latency": round(time.time() - t0, 2)}
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code} {e.read().decode()[:300]}"
            if e.code in (400, 401, 403, 404, 413, 422):
                break
            time.sleep(4 * (attempt + 1))
        except Exception as e:
            err = str(e)[:200]; time.sleep(4 * (attempt + 1))
    return {"id": c["id"], "task": c["task"], "error": err}

done = {json.loads(l)["id"] for l in open(DST)} if os.path.exists(DST) else set()
todo = [c for c in B["cases"] if c["id"] not in done][:LIMIT]
print("todo", len(todo), MODEL, MODE, flush=True)
lock, out, n = threading.Lock(), open(DST, "a"), [0, 0]
def work(c):
    r = call(c)
    with lock:
        n[0] += 1
        if "error" in r:
            n[1] += 1; print("ERR", r["task"], r["error"][:240], flush=True)
        else:
            out.write(json.dumps(r) + "\n"); out.flush()
        if n[0] % 100 == 0: print(n[0], "fail", n[1], flush=True)
with ThreadPoolExecutor(int(os.environ.get("WORKERS", 6))) as ex:
    list(ex.map(work, todo))
print("finished", n[0], "fail", n[1], flush=True)
