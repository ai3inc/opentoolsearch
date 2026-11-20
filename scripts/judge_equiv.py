"""LLM judge: can tool B stand in for tool A?  BIZROUTER_API_KEY=... python scripts/judge_equiv.py in.jsonl out.jsonl"""
import json, os, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
SRC, DST = sys.argv[1], sys.argv[2]
MODEL = sys.argv[3] if len(sys.argv) > 3 else "bizrouter/deepseek-v4.1-flash"
KEY = os.environ["BIZROUTER_API_KEY"]; BASE = os.environ.get("LLM_BASE_URL", "https://api.bizrouter.ai/v1")
SYSTEM = """You compare two tool definitions from different providers.
Answer "yes" if tool B offers the same capability as tool A, so that an agent holding only tool B could
fulfil the requests tool A is meant for (inputs may be named differently; the provider may differ).
Answer "no" if B does something different, narrower in a way that matters, or only related.
Answer with one word: yes or no."""

def call(r):
    body = json.dumps({"model": MODEL, "max_tokens": 8, "temperature": 0,
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": f"Tool A:\n{r['a']}\n\nTool B:\n{r['b']}"}]}).encode()
    for attempt in range(5):
        try:
            req = urllib.request.Request(f"{BASE}/chat/completions", data=body,
                                         headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                t = (json.loads(resp.read())["choices"][0]["message"].get("content") or "").strip().lower()
            if t.startswith("yes") or t.startswith("no"):
                return t.startswith("yes")
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None

done = {json.loads(l)["id"] for l in open(DST)} if os.path.exists(DST) else set()
todo = [r for r in (json.loads(l) for l in open(SRC)) if r["id"] not in done]
print("todo", len(todo), flush=True)
lock, out, n = threading.Lock(), open(DST, "a"), [0, 0]
def work(r):
    v = call(r)
    with lock:
        n[0] += 1
        if v is None: n[1] += 1
        else:
            out.write(json.dumps({"id": r["id"], "gold": r["gold"], "cand": r["cand"], "cos": r["cos"], "equiv": v}) + "\n"); out.flush()
        if n[0] % 1000 == 0: print(n[0], "fail", n[1], flush=True)
with ThreadPoolExecutor(int(os.environ.get("WORKERS", 40))) as ex:
    list(ex.map(work, todo))
print("finished", n[0], "fail", n[1], flush=True)
