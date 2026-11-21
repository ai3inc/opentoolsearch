"""Open dense tool-retrieval models on our benchmarks, same texts and protocol for all.

Models (weights on Hugging Face): the four ToolRet-trained retrievers of the ToolRet authors (bge-base/large, e5-base/large),
callbus-emb-8b (Qwen3-Embedding-8B further trained on ToolRet-Training), and the two Qwen3-Embedding models as general
references. Each model is run with every query format its card documents; the format is picked afterwards on one benchmark.

Per model it stores the top-20 ranking for
  own_en   the 200 English requests over the 1,000 MCP tools of OTS-MCP-1k
  own_ko   the same requests in Korean
  public   the 769 ToolRet per-source questions, each over its own source dataset's tools (request only)
and the median time to embed one request (one at a time on this GPU; the tool vectors are precomputed).

  python scripts/v7_dense_models.py OUT_DIR [model-name ...]
Run from the repository root of a GPU machine with the packages of this repository plus `datasets`.
"""
import json, os, random, re, sys, time, collections, statistics
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from opentoolsearch.data import load_tools, load_all_queries

OUT = sys.argv[1]; ONLY = set(sys.argv[2:])
os.makedirs(OUT, exist_ok=True)
GEN = "Given a task, retrieve tools that can accomplish it."
CB = "Instruct: Given an agent's request for a tool, retrieve the MCP tool that fulfills it.\nQuery: {q}"
BGE_I = "Represent this sentence for searching relevant passages: {q}"
MODELS = {
    "ToolRet-trained-bge-base-en-v1.5": dict(path="mangopy/ToolRet-trained-bge-base-en-v1.5", pool="cls", dp="", maxlen=512,
        variants={"plain": "{q}", "bge-instruction": BGE_I, "tool-prompt": GEN + " {q}"}),
    "ToolRet-trained-bge-large-en-v1.5": dict(path="mangopy/ToolRet-trained-bge-large-en-v1.5", pool="cls", dp="", maxlen=512,
        variants={"plain": "{q}", "bge-instruction": BGE_I, "tool-prompt": GEN + " {q}"}),
    "ToolRet-trained-e5-base-v2": dict(path="mangopy/ToolRet-trained-e5-base-v2", pool="mean", dp="passage: ", maxlen=512,
        variants={"query": "query: {q}", "query+tool-prompt": "query: " + GEN + " {q}"}),
    "ToolRet-trained-e5-large-v2": dict(path="mangopy/ToolRet-trained-e5-large-v2", pool="mean", dp="passage: ", maxlen=512,
        variants={"query": "query: {q}", "query+tool-prompt": "query: " + GEN + " {q}"}),
    "callbus-emb-8b": dict(path="yasinyaman/callbus-emb-8b", revision="v0.2", pool="st", dp="", maxlen=512, variants={"callbus-instruction": CB}),
    "Qwen3-Embedding-8B": dict(path="Qwen/Qwen3-Embedding-8B", pool="st", dp="", maxlen=512, variants={"callbus-instruction": CB}),
    "Qwen3-Embedding-0.6B": dict(path="Qwen/Qwen3-Embedding-0.6B", pool="st", dp="", maxlen=512, variants={"callbus-instruction": CB}),
}

# ---------------------------------------------------------------- the three benchmarks
B = json.load(open("data/v3/bench.json"))
KO = {r["id"]: r["ko"] for r in json.load(open("data/v6/ko_queries.json"))}
own_tools = [json.dumps({"name": t["api_name"], "description": t["description"], "parameters": t["parameters"]}, ensure_ascii=False) for t in B["tools"]]
own_ids = [t["id"] for t in B["tools"]]
own_q = {"own_en": [q["query"] for q in B["questions"]], "own_ko": [KO[q["id"]] for q in B["questions"]]}
own_qid = [q["id"] for q in B["questions"]]

# public: rebuilt exactly as scripts/v4_public_prep.py does (same code, same seed), then checked against bench_public.json
random.seed(11)
pre = lambda i: re.sub(r"_tool_\d+$", "", str(i)) if re.search(r"_tool_\d+$", str(i)) else re.sub(r"_?\d+$", "", str(i))
by = collections.defaultdict(list)
for t in load_tools():
    by[pre(t["id"])].append(t)
pools, cases = {}, []
for task, qs in load_all_queries().items():
    ps = sorted({pre(g) for q in qs for g in q["qrels"]})
    pool = [t for p in ps for t in by[p]]
    if len(pool) > 2000:
        continue
    pools[task] = [{"id": str(t["id"]), "text": t["documentation"]} for t in pool]
    ids = {t["id"] for t in pools[task]}
    for q in random.sample(qs, min(25, len(qs))):
        gold = [g for g in q["qrels"] if g in ids and q["qrels"][g] > 0]
        cases.append({"task": task, "id": q["id"], "query": q["query"], "gold": gold})
ref = json.load(open("data/v4/bench_public.json"))
assert [c["id"] for c in cases] == [c["id"] for c in ref["cases"]], "public cases differ from bench_public.json"
assert {t: [x["id"] for x in p] for t, p in pools.items()} == {t: [x["id"] for x in p] for t, p in ref["pools"].items()}, "pools differ"
cases = [c for c in cases if c["query"].strip() and c["gold"]]
print("own", len(own_tools), "tools,", len(own_qid), "requests; public", len(pools), "subsets,", len(cases), "questions", flush=True)


# ---------------------------------------------------------------- encoders
class Enc:
    def __init__(self, spec):
        self.spec = spec; self.pool = spec["pool"]
        kw = {"revision": spec["revision"]} if "revision" in spec else {}
        if self.pool == "st":
            from sentence_transformers import SentenceTransformer
            self.m = SentenceTransformer(spec["path"], model_kwargs={"torch_dtype": torch.bfloat16}, **kw); self.m.eval()
            self.m.max_seq_length = spec["maxlen"]
        else:
            self.tok = AutoTokenizer.from_pretrained(spec["path"], **kw)
            self.m = AutoModel.from_pretrained(spec["path"], torch_dtype=torch.bfloat16, **kw).cuda().eval()

    @torch.no_grad()
    def encode(self, texts, bs=128):
        order = np.argsort([-len(t) for t in texts]); out = [None] * len(texts)
        for i in range(0, len(texts), bs):
            b = order[i:i + bs]; batch = [texts[j] for j in b]
            if self.pool == "st":
                e = self.m.encode(batch, batch_size=len(batch), convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
            else:
                x = self.tok(batch, padding=True, truncation=True, max_length=self.spec["maxlen"], return_tensors="pt").to("cuda")
                h = self.m(**x).last_hidden_state
                if self.pool == "cls":
                    e = h[:, 0]
                else:
                    msk = x["attention_mask"].unsqueeze(-1).to(h.dtype)
                    e = (h * msk).sum(1) / msk.sum(1)
                e = F.normalize(e.float(), dim=-1)
            for j, v in zip(b, e.float()):
                out[j] = v
        return torch.stack(out)


def rank(Q, E, ids, k=20):
    s = Q @ E.T
    top = s.topk(min(k, E.shape[0]), dim=1).indices.cpu().tolist()
    return [[ids[j] for j in row] for row in top]


for name, spec in MODELS.items():
    if ONLY and name not in ONLY:
        continue
    dst = f"{OUT}/dense_{name}.json"
    if os.path.exists(dst):
        print("skip", name, flush=True); continue
    t0 = time.time(); enc = Enc(spec)
    res = {"model": spec["path"], "revision": spec.get("revision"), "pooling": spec["pool"], "variants": {}}
    own_E = enc.encode([spec["dp"] + t for t in own_tools])
    pub_E = {task: enc.encode([spec["dp"] + x["text"] for x in pool]) for task, pool in pools.items()}
    for vn, tmpl in spec["variants"].items():
        v = {}
        for key, qs in own_q.items():
            Q = enc.encode([tmpl.format(q=q) for q in qs])
            v[key] = dict(zip(own_qid, rank(Q, own_E, own_ids)))
        v["public"] = {}
        for task, pool in pools.items():
            cs = [c for c in cases if c["task"] == task]
            if not cs:
                continue
            Q = enc.encode([tmpl.format(q=c["query"]) for c in cs])
            for c, r in zip(cs, rank(Q, pub_E[task], [x["id"] for x in pool])):
                v["public"][c["id"]] = r
        res["variants"][vn] = v
    # one request at a time: embed the request, score it against the 1,000 precomputed tool vectors
    vn0 = next(iter(spec["variants"])); tm = spec["variants"][vn0]; lat = []
    for i, q in enumerate(own_q["own_en"][:60]):
        torch.cuda.synchronize(); t1 = time.perf_counter()
        Q = enc.encode([tm.format(q=q)]); (Q @ own_E.T).topk(20); torch.cuda.synchronize()
        if i >= 8:
            lat.append((time.perf_counter() - t1) * 1000)
    res["latency_ms_p50"] = round(statistics.median(lat), 2)
    res["params_m"] = round(sum(p.numel() for p in (enc.m.parameters())) / 1e6)
    json.dump(res, open(dst, "w"))
    print(f"done {name}: {time.time()-t0:.0f}s, p50 {res['latency_ms_p50']} ms, {res['params_m']}M params", flush=True)
    del enc, own_E, pub_E; torch.cuda.empty_cache()
