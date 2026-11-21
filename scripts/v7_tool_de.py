"""Tool-DE (Tool-Embed dense retriever, Tool-Rank LLM reranker) on our benchmarks, same texts and protocol as scripts/v7_dense_models.py.

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
from opentoolsearch.rerank import Reranker

OUT = sys.argv[1]; ONLY = set(sys.argv[2:])
os.makedirs(OUT, exist_ok=True)
GEN = "Given a task, retrieve tools that can accomplish it."
CB = "Instruct: Given an agent's request for a tool, retrieve the MCP tool that fulfills it.\nQuery: {q}"
BGE_I = "Represent this sentence for searching relevant passages: {q}"
MODELS_UNUSED = {
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



TD_DEFAULT = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {q}"
TD_TOOL = "Instruct: " + GEN + "\nQuery: {q}"
EMB = {
    "Tool-Embed-0.6B": dict(path="Lux1997/Tool-Embed-0.6B", pool="st", dp="", maxlen=512, variants={"callbus-instruction": CB, "qwen3-default-instruction": TD_DEFAULT, "tool-prompt": TD_TOOL}),
    "Tool-Embed-4B": dict(path="Lux1997/Tool-Embed-4B", pool="st", dp="", maxlen=512, variants={"callbus-instruction": CB, "qwen3-default-instruction": TD_DEFAULT, "tool-prompt": TD_TOOL}),
}
RERANK_DEPTH = 25
for name, spec in EMB.items():
    dst = f"{OUT}/dense_{name}.json"
    if os.path.exists(dst):
        continue
    t0 = time.time(); enc = Enc(spec)
    res = {"model": spec["path"], "revision": None, "pooling": spec["pool"], "variants": {}}
    own_E = enc.encode([spec["dp"] + t for t in own_tools])
    pub_E = {task: enc.encode([spec["dp"] + x["text"] for x in pool]) for task, pool in pools.items()}
    cand = {}          # (variant, key) -> candidate lists kept for the reranker
    for vn, tmpl in spec["variants"].items():
        v = {}
        for key, qs in own_q.items():
            Q = enc.encode([tmpl.format(q=q) for q in qs])
            s_, j_ = (Q @ own_E.T).topk(RERANK_DEPTH, dim=1)
            v[key] = dict(zip(own_qid, [[own_ids[j] for j in row[:20]] for row in j_.cpu().tolist()]))
            cand[(name, vn, key)] = j_.cpu().tolist()
        v["public"] = {}
        for task, pool in pools.items():
            cs = [c for c in cases if c["task"] == task]
            if not cs: continue
            Q = enc.encode([tmpl.format(q=c["query"]) for c in cs])
            s_, j_ = (Q @ pub_E[task].T).topk(min(RERANK_DEPTH, len(pool)), dim=1)
            for c, row in zip(cs, j_.cpu().tolist()):
                v["public"][c["id"]] = [pool[j]["id"] for j in row[:20]]
                cand[(name, vn, "public", c["id"])] = row
        res["variants"][vn] = v
    vn0 = next(iter(spec["variants"])); tm = spec["variants"][vn0]; lat = []
    for i, q in enumerate(own_q["own_en"][:60]):
        torch.cuda.synchronize(); t1 = time.perf_counter()
        Q = enc.encode([tm.format(q=q)]); (Q @ own_E.T).topk(20); torch.cuda.synchronize()
        if i >= 8: lat.append((time.perf_counter() - t1) * 1000)
    res["latency_ms_p50"] = round(statistics.median(lat), 2)
    res["params_m"] = round(sum(p.numel() for p in enc.m.parameters()) / 1e6)
    json.dump(res, open(dst, "w"))
    print(f"done {name}: {time.time()-t0:.0f}s p50 {res['latency_ms_p50']} ms", flush=True)
    if name == "Tool-Embed-4B":
        # the Tool-DE pipeline: Tool-Embed-4B -> top 25 -> Tool-Rank-4B (pointwise yes/no, Qwen3-Reranker prompt)
        best = max(res["variants"], key=lambda k: 0)           # all formats are reranked; the format is picked afterwards on the English own set
        rr = Reranker("Lux1997/Tool-Rank-4B", dtype=torch.bfloat16); rr.model.eval()
        pres = {"model": "Lux1997/Tool-Embed-4B + Lux1997/Tool-Rank-4B", "revision": None, "pooling": "st+rerank", "variants": {}}

        def rerank(queries, docs_idx, texts, ids):
            pairs = [("", q, texts[i]) for q, row in zip(queries, docs_idx) for i in row]
            flat = []
            with torch.no_grad():
                for b in range(0, len(pairs), 128):
                    flat += rr.score(pairs[b:b + 128]).tolist()
            out, k = [], 0
            for row in docs_idx:
                sc = flat[k:k + len(row)]; k += len(row)
                order = sorted(range(len(row)), key=lambda j: -sc[j])
                out.append([ids[row[j]] for j in order][:20])
            return out
        for vn in spec["variants"]:
            v = {}
            for key, qs in own_q.items():
                v[key] = dict(zip(own_qid, rerank(qs, cand[(name, vn, key)], own_tools, own_ids)))
            v["public"] = {}
            for task, pool in pools.items():
                cs = [c for c in cases if c["task"] == task]
                if not cs: continue
                rows = [cand[(name, vn, "public", c["id"])] for c in cs]
                for c, r_ in zip(cs, rerank([c["query"] for c in cs], rows, [x["text"] for x in pool], [x["id"] for x in pool])):
                    v["public"][c["id"]] = r_
            pres["variants"][vn] = v
            print("reranked", vn, flush=True)
        lat2 = []
        for i, q in enumerate(own_q["own_en"][:30]):
            row = cand[(name, vn0, "own_en")][i]
            torch.cuda.synchronize(); t1 = time.perf_counter()
            Q = enc.encode([tm.format(q=q)]); (Q @ own_E.T).topk(RERANK_DEPTH)
            with torch.no_grad(): rr.score([("", q, own_tools[j]) for j in row])
            torch.cuda.synchronize()
            if i >= 5: lat2.append((time.perf_counter() - t1) * 1000)
        pres["latency_ms_p50"] = round(statistics.median(lat2), 2)
        pres["params_m"] = res["params_m"] + round(sum(p.numel() for p in rr.model.parameters()) / 1e6)
        json.dump(pres, open(f"{OUT}/dense_Tool-DE (Tool-Embed-4B + Tool-Rank-4B).json", "w"))
        print("done Tool-DE pipeline", pres["latency_ms_p50"], "ms", flush=True)
        del rr
    del enc, own_E, pub_E; torch.cuda.empty_cache()
