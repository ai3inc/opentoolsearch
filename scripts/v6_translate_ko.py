"""Translate the 200 English requests of OTS-MCP-1k into natural spoken Korean, then check each translation.

Translation: anthropic/claude-sonnet-5.5; check: google/gemini-3.6-flash (a different model) reads English + Korean and
rates whether the Korean keeps every requirement. Both through BizRouter. The tool library stays English.

  BIZROUTER_API_KEY=... python scripts/v6_translate_ko.py data/v3/bench.json data/v6/ko_queries.json [limit]
"""
import json, os, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
B = json.load(open(sys.argv[1])); DST = sys.argv[2]
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else None
KEY = os.environ["BIZROUTER_API_KEY"]
TRANSLATE, CHECK = "anthropic/claude-sonnet-5.5", "google/gemini-3.6-flash"
SYS_T = ("You translate a user's request to an AI assistant from English into Korean. Write it the way a Korean speaker would "
         "naturally type or say it to an assistant (polite conversational style, e.g. 해요체 or 합니다체; not stiff translationese). "
         "Keep file paths, URLs, code, identifiers, numbers, units, proper nouns and product or API names exactly as written. "
         "Do not add, drop or soften any requirement. Output only the Korean request.")
SYS_C = ("You compare an English request with its Korean translation. Reply with JSON only: "
         '{"score": 1-5, "missing": [details of the English that the Korean lacks], "added": [details the Korean adds]}. '
         "5 = every requirement and detail preserved, nothing added; 1 = the meaning changed.")


def chat(model, system, user):
    body = json.dumps({"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
    err = None
    for attempt in range(4):
        try:
            req = urllib.request.Request("https://api.bizrouter.ai/v1/chat/completions", data=body,
                                         headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json", "User-Agent": "opentoolsearch-bench/0.1"})
            r = json.loads(urllib.request.urlopen(req, timeout=180).read())
            return r["choices"][0]["message"]["content"].strip(), r.get("usage")
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code} {e.read().decode()[:300]}"
            if e.code in (400, 401, 403, 404, 422): break
            time.sleep(4 * (attempt + 1))
        except Exception as e:
            err = str(e)[:200]; time.sleep(4 * (attempt + 1))
    raise RuntimeError(err)


def one(q):
    ko, u1 = chat(TRANSLATE, SYS_T, q["query"])
    raw, u2 = chat(CHECK, SYS_C, f"English:\n{q['query']}\n\nKorean:\n{ko}")
    try:
        s, e = raw.find("{"), raw.rfind("}")
        chk = json.loads(raw[s:e + 1])
    except Exception:
        chk = {"score": None, "raw": raw[:300]}
    return {"id": q["id"], "en": q["query"], "ko": ko, "check": chk, "usage": [u1, u2]}


done = {}
if os.path.exists(DST):
    done = {r["id"]: r for r in json.load(open(DST))}
todo = [q for q in B["questions"] if q["id"] not in done][:LIMIT]
print("todo", len(todo), flush=True)
lock = threading.Lock()
def work(q):
    try:
        r = one(q)
    except Exception as e:
        print("ERR", q["id"][:12], str(e)[:200], flush=True); return
    with lock:
        done[r["id"]] = r
        json.dump(list(done.values()), open(DST, "w"), ensure_ascii=False)
with ThreadPoolExecutor(int(os.environ.get("WORKERS", 6))) as ex:
    list(ex.map(work, todo))
print("done", len(done), flush=True)
