import json
from huggingface_hub import hf_hub_download
for f in ("huggingface_train.json", "torchhub_train.json", "tensorflow_train.json"):
    p = hf_hub_download("gorilla-llm/APIBench", f, repo_type="dataset")
    txt = open(p).read()
    try:
        rows = json.loads(txt)
    except Exception:
        rows = [json.loads(l) for l in txt.splitlines() if l.strip()]
    print(f, len(rows), type(rows[0]), list(rows[0].keys()) if isinstance(rows[0], dict) else "")
    r = rows[0]
    for k, v in r.items():
        print("  ##", k, ":", str(v)[:500])
from opentoolsearch.data import load_tools
t = load_tools()
for pre in ("gorilla_huggingface", "gorilla_pytorch", "gorilla_tensor"):
    d = [x for x in t if str(x["id"]).startswith(pre)]
    print(pre, len(d), d[0]["documentation"][:400])
