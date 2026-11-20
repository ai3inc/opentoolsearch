"""Track B data: the *training* split of Gorilla APIBench (HF / TorchHub / TensorFlow Hub).
ToolRet's gorilla-* subsets are built from the same API pool, so this is in-domain data for
3 of the 35 subsets; it is kept as a separately reported track. Rows that copy or nearly
copy a ToolRet test question are dropped with the same rule as prep_train.py.
"""
import json, re, sys, random
import torch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from opentoolsearch.data import load_all_queries

OUT = sys.argv[1] if len(sys.argv) > 1 else "data"
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()
rows, seen = [], set()
for f in ("huggingface_train.json", "torchhub_train.json", "tensorflow_train.json"):
    p = hf_hub_download("gorilla-llm/APIBench", f, repo_type="dataset")
    txt = open(p).read()
    try:
        data = json.loads(txt)
    except Exception:
        data = [json.loads(l) for l in txt.splitlines() if l.strip()]
    n0 = len(rows)
    for i, r in enumerate(data):
        code = r["code"] if isinstance(r["code"], str) else json.dumps(r["code"])
        m = re.search(r"###\s*Instruction:\s*(.*?)\s*###\s*Output", code, re.S) or \
            re.search(r"'Instruction':\s*(.*?),\s*'Output'", code, re.S) or \
            re.search(r"\"Instruction\":\s*\"(.*?)\",\s*\"Output\"", code, re.S)
        if not m or not isinstance(r.get("api_data"), dict):
            continue
        q = m.group(1).strip().strip("'\"")
        if len(q) < 15 or q in seen:
            continue
        seen.add(q)
        d = dict(r["api_data"]); name = d.pop("api_name", None)
        if name is not None:
            d["name"] = name                                   # corpus layout: name last
        rows.append({"id": f"gorilla_{f.split('_')[0]}_{i}", "query": q, "pos": [json.dumps(d, ensure_ascii=False)],
                     "neg": [], "src": "gorilla-train",
                     "brief": f"{name}: {d.get('functionality','')} ({d.get('domain','')}, {d.get('framework','')})"})
    print(f, len(data), "->", len(rows) - n0, flush=True)

test = [norm(q["query"]) for qs in load_all_queries().values() for q in qs]
tset, twords = set(test), [set(t.split()) for t in test]
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", model_kwargs={"torch_dtype": torch.bfloat16})
model.max_seq_length = 256
qn = [norm(r["query"]) for r in rows]
with torch.no_grad():
    Et = model.encode(test, batch_size=512, convert_to_tensor=True, normalize_embeddings=True).float()
    Eq = model.encode(qn, batch_size=1024, convert_to_tensor=True, normalize_embeddings=True).float()
    s, j = (Eq @ Et.T).topk(5, dim=1)
s, j = s.cpu(), j.cpu()
keep, why = [], {"exact": 0, "jaccard": 0, "cosine": 0}
for i, r in enumerate(rows):
    w = set(qn[i].split())
    if qn[i] in tset:
        why["exact"] += 1
    elif any(len(w & twords[x]) / max(1, len(w | twords[x])) >= 0.60 for x in j[i].tolist()):
        why["jaccard"] += 1
    elif s[i, 0].item() >= 0.92:
        why["cosine"] += 1
    else:
        keep.append(r)
print("removed as test near-copies", why, "kept", len(keep), flush=True)
json.dump({"rows": len(rows), "removed": why, "kept": len(keep)}, open(f"{OUT}/gorilla_dedup_report.json", "w"))
with open(f"{OUT}/gorilla_raw.jsonl", "w") as f, open(f"{OUT}/gorilla_slim.jsonl", "w") as g:
    for r in keep:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        g.write(json.dumps({k: r[k] for k in ("id", "query", "brief", "src")}, ensure_ascii=False) + "\n")
