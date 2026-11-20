"""Write a ToolRet-style retrieval instruction for every external training row.

  BIZROUTER_API_KEY=... python scripts/gen_instructions.py in.jsonl out.jsonl [model]

Calls an OpenAI-compatible chat endpoint; resumable (skips ids already in out.jsonl).
"""
import json, os, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

SRC, DST = sys.argv[1], sys.argv[2]
MODEL = sys.argv[3] if len(sys.argv) > 3 else "bizrouter/deepseek-v4.1-flash"
BASE = os.environ.get("LLM_BASE_URL", "https://api.bizrouter.ai/v1")
KEY = os.environ["BIZROUTER_API_KEY"]
WORKERS = int(os.environ.get("WORKERS", 32))

SYSTEM = """You write one-sentence retrieval instructions for a tool search engine.
You are given a user request and a short note on the tools that solve it.
Write ONE sentence in exactly this style:
Given a `<short task type>` task, retrieve tools that <what the tools must do: the inputs they process and the results they return>.
Rules: describe the needed capability, do not copy tool names, do not answer the request, no quotes around the sentence, at most 55 words."""

SHOTS = [
    ("What is the straddle data for Alphabet Inc. (ticker: GOOGL) and what are the latest popular ideas on Trading View for the US stock market?",
     "straddle: retrieves the straddle data for a given stock ticker; ideas_list: fetches a list of ideas/posts with filtering and sorting options",
     "Given a `stock market data` and `popular ideas retrieval` task, retrieve tools that can provide straddle data for a given stock ticker and fetch the latest popular ideas on Trading View for the US stock market."),
    ("I'm planning to take my kids to the park tomorrow. Can you provide the air quality forecast for zip code 98101?",
     "air_quality_forecast: returns the air quality forecast for a zip code",
     "Given an `air quality forecast` task, retrieve tools that provide environmental data by processing geographic details like zip codes to return forecasts for outdoor planning."),
    ("I want to check the status of my recent order. Can you fetch the details of my order, including the order status and delivery date?",
     "get_order: get an order by its id",
     "Given an `order status inquiry` task, retrieve tools that facilitate order management by processing order-related parameters to provide detailed information about the order status and delivery date."),
]


SYSTEM_Q = """You write one-sentence retrieval instructions for a tool search engine.
You are given only a user request. Write ONE sentence in exactly this style:
Given a `<short task type>` task, retrieve tools that <what the tools must do: the inputs they process and the results they return>.
Rules: describe the capability the request needs, do not answer the request, no quotes, at most 55 words."""


def messages(row):
    if "brief" not in row:                      # request-only mode: the writer does not see the answer tools
        m = [{"role": "system", "content": SYSTEM_Q}]
        for q, _, a in SHOTS:
            m += [{"role": "user", "content": f"Request: {q}"}, {"role": "assistant", "content": a}]
        m.append({"role": "user", "content": f"Request: {row['query'][:1500]}"})
        return m
    m = [{"role": "system", "content": SYSTEM}]
    for q, b, a in SHOTS:
        m += [{"role": "user", "content": f"Request: {q}\nTools: {b}"}, {"role": "assistant", "content": a}]
    m.append({"role": "user", "content": f"Request: {row['query'][:1500]}\nTools: {row['brief'][:600]}"})
    return m


def call(row):
    body = json.dumps({"model": MODEL, "messages": messages(row), "max_tokens": 120, "temperature": 0.4}).encode()
    for attempt in range(5):
        try:
            req = urllib.request.Request(f"{BASE}/chat/completions", data=body,
                                         headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                out = json.loads(r.read())
            text = (out["choices"][0]["message"].get("content") or "").strip().split("\n")[0].strip().strip('"')
            if text.lower().startswith("given a"):
                return text
        except Exception as e:
            err = str(e)[:100]
            time.sleep(2 * (attempt + 1))
    return None


done = set()
if os.path.exists(DST):
    done = {json.loads(l)["id"] for l in open(DST)}
rows = [json.loads(l) for l in open(SRC)]
todo = [r for r in rows if r["id"] not in done]
if os.environ.get("LIMIT"):
    todo = todo[: int(os.environ["LIMIT"])]
print(f"rows {len(rows)} done {len(done)} todo {len(todo)}", flush=True)
lock, n, fail, t0 = threading.Lock(), 0, 0, time.time()
out = open(DST, "a")


def work(row):
    global n, fail
    inst = call(row)
    with lock:
        n += 1
        if inst:
            out.write(json.dumps({"id": row["id"], "instruction": inst}, ensure_ascii=False) + "\n"); out.flush()
        else:
            fail += 1
        if n % 500 == 0:
            print(f"{n}/{len(todo)} fail {fail} {n/(time.time()-t0):.1f}/s", flush=True)


with ThreadPoolExecutor(WORKERS) as ex:
    list(ex.map(work, todo))
print(f"finished {n} fail {fail} in {time.time()-t0:.0f}s", flush=True)
