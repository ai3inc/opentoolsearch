"""Use a decision model (BizRouter /v1/decisions: Jev, Gemini Decision) as the second stage
over the same 50 first-stage candidates the rerankers read.

  mode noul    one yes/no question per candidate tool ("is this tool needed?") -> p(yes) per tool
  mode choice  one question: choose the tool among the 50 -> probability per tool

  BIZROUTER_API_KEY=... python scripts/v3_run_decisions.py data/v3/bench.json OUT.jsonl MODEL MODE [limit]
"""
import json, os, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
B = json.load(open(sys.argv[1])); DST, MODEL, MODE = sys.argv[2], sys.argv[3], sys.argv[4]
LIMIT = int(sys.argv[5]) if len(sys.argv) > 5 else None
KEY = os.environ["BIZROUTER_API_KEY"]
TOPK = int(os.environ.get("TOPK", "50"))
T = B["tools"]

def body(qi, q):
    cand = B["first"]["idx"][qi][:TOPK]
    if MODE == "choice":
        qs = {"tool": {"type": "choice", "instructions": "Select the tool that is needed to fulfil the user's request.",
                       "criteria": {T[i]["api_name"]: T[i]["description"][:400] for i in cand}}}
    else:
        qs = {f"t{k}": {"type": "noul", "instructions": "Is the following tool needed to fulfil the user's request? "
                        f"Tool `{T[i]['api_name']}`: {T[i]['description'][:400]}"} for k, i in enumerate(cand)}
    return json.dumps({"model": MODEL, "state": q["query"], "questions": qs}).encode()

def call(qi, q):
    err = None
    for attempt in range(4):
        t0 = time.time()
        try:
            req = urllib.request.Request("https://api.bizrouter.ai/v1/decisions", data=body(qi, q),
                                         headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json",
                                                  "User-Agent": "opentoolsearch-bench/0.1"})
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            cand = B["first"]["idx"][qi][:TOPK]; a = r["answers"]
            if MODE == "choice":
                p = a["tool"].get("probabilities", {})
                scores = [float(p.get(T[i]["api_name"], 0.0)) for i in cand]
            else:
                scores = [float(a[f"t{k}"]["noul"]) for k in range(len(cand))]
            return {"id": q["id"], "scores": scores, "usage": r.get("usage"), "latency": round(time.time() - t0, 2)}
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code} {e.read().decode()[:300]}"
            if e.code in (400, 401, 403, 404, 422):
                break
            time.sleep(4 * (attempt + 1))
        except Exception as e:
            err = str(e)[:200]; time.sleep(4 * (attempt + 1))
    return {"id": q["id"], "error": err}

done = {json.loads(l)["id"] for l in open(DST)} if os.path.exists(DST) else set()
todo = [(qi, q) for qi, q in enumerate(B["questions"]) if q["id"] not in done][:LIMIT]
print("todo", len(todo), MODEL, MODE, flush=True)
lock, out, n = threading.Lock(), open(DST, "a"), [0, 0]
def work(x):
    r = call(*x)
    with lock:
        n[0] += 1
        if "error" in r:
            n[1] += 1; print("ERR", r["error"][:220], flush=True)
        else:
            out.write(json.dumps(r) + "\n"); out.flush()
        if n[0] % 25 == 0: print(n[0], "fail", n[1], flush=True)
with ThreadPoolExecutor(int(os.environ.get("WORKERS", 6))) as ex:
    list(ex.map(work, todo))
print("finished", n[0], "fail", n[1], flush=True)
