"""Other first stages in front of the OpenToolSearch reranker, on our three benchmarks.

For each first-stage retriever: its top-50 candidates (ids and scores) and what the published reranker
(ai3inc/opentoolsearch-rerank-0.6b) makes of the top 25 - the library default: rerank 25, fuse
z(rerank) + 0.5 z(first stage). Own set in English and Korean, ToolRet per-source (769 questions).

  python scripts/v8_first_stage_rerank.py OUT_DIR [first-stage-name ...]
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



TD_TOOL = "Instruct: " + GEN + "\nQuery: {q}"
FIRST = {
    "callbus-emb-8b": dict(path="yasinyaman/callbus-emb-8b", revision="v0.2", pool="st", dp="", maxlen=512, q=CB),
    "Tool-Embed-4B": dict(path="Lux1997/Tool-Embed-4B", pool="st", dp="", maxlen=512, q=TD_TOOL),
    "Qwen3-Embedding-8B": dict(path="Qwen/Qwen3-Embedding-8B", pool="st", dp="", maxlen=512, q=CB),
    "Nemotron-3-Embed-1B": dict(path="nvidia/Nemotron-3-Embed-1B-BF16", pool="st", dp="passage: ", maxlen=512, q="query: {q}"),
}
DEPTH, W, TOPN = 25, 0.5, 50
rr = Reranker("ai3inc/opentoolsearch-rerank-0.6b", dtype=torch.bfloat16); rr.model.eval()


def zs(v):
    mu = statistics.mean(v); sd = statistics.pstdev(v) or 1.0
    return [(x - mu) / sd for x in v]


def rerank_rows(queries, cand, scores, texts, ids):
    """cand: top-50 index lists, scores: their first-stage scores. -> (fused top 20 ids, reranker-only top 20 ids)"""
    pairs = [("", q, texts[j]) for q, row in zip(queries, cand) for j in row[:DEPTH]]
    flat = []
    with torch.no_grad():
        for b in range(0, len(pairs), 128):
            flat += rr.score(pairs[b:b + 128]).tolist()
    fused, alone, k = [], [], 0
    for row, fs in zip(cand, scores):
        n = min(DEPTH, len(row)); sc = flat[k:k + n]; k += n; f = fs[:n]
        z = [a + W * b for a, b in zip(zs(sc), zs(f))] if n > 1 else sc
        fused.append([ids[row[j]] for j in sorted(range(n), key=lambda j: -z[j])][:20])
        alone.append([ids[row[j]] for j in sorted(range(n), key=lambda j: -sc[j])][:20])
    return fused, alone


for name, spec in FIRST.items():
    if ONLY and name not in ONLY:
        continue
    dst = f"{OUT}/fs_{name}.json"
    if os.path.exists(dst):
        print("skip", name, flush=True); continue
    t0 = time.time(); spec = {**spec, "variants": {"x": spec["q"]}}; enc = Enc(spec); tmpl = spec["q"]
    own_E = enc.encode([spec["dp"] + t for t in own_tools])
    pub_E = {task: enc.encode([spec["dp"] + x["text"] for x in pool]) for task, pool in pools.items()}
    res = {"model": spec["path"], "revision": spec.get("revision"), "depth": DEPTH, "fusion": W, "first_top50": {}, "final": {}, "rerank_only": {}, "first_top20": {}}
    for key, qs in own_q.items():
        Q = enc.encode([tmpl.format(q=q) for q in qs])
        s_, j_ = (Q @ own_E.T).topk(TOPN, dim=1); cand, sc = j_.cpu().tolist(), s_.cpu().tolist()
        res["first_top50"][key] = dict(zip(own_qid, [[own_ids[j] for j in row] for row in cand]))
        res["first_top20"][key] = {qid: v[:20] for qid, v in res["first_top50"][key].items()}
        f, a = rerank_rows(qs, cand, sc, own_tools, own_ids)
        res["final"][key] = dict(zip(own_qid, f)); res["rerank_only"][key] = dict(zip(own_qid, a))
    for part in ("first_top50", "final", "rerank_only", "first_top20"):
        res[part].setdefault("public", {})
    for task, pool in pools.items():
        cs = [c for c in cases if c["task"] == task]
        if not cs:
            continue
        Q = enc.encode([tmpl.format(q=c["query"]) for c in cs])
        s_, j_ = (Q @ pub_E[task].T).topk(min(TOPN, len(pool)), dim=1); cand, sc = j_.cpu().tolist(), s_.cpu().tolist()
        pids = [x["id"] for x in pool]
        f, a = rerank_rows([c["query"] for c in cs], cand, sc, [x["text"] for x in pool], pids)
        for c, row, ff, aa in zip(cs, cand, f, a):
            res["first_top50"]["public"][c["id"]] = [pids[j] for j in row]
            res["first_top20"]["public"][c["id"]] = [pids[j] for j in row][:20]
            res["final"]["public"][c["id"]] = ff; res["rerank_only"]["public"][c["id"]] = aa
    json.dump(res, open(dst, "w"))
    print(f"done {name}: {time.time()-t0:.0f}s", flush=True)
    del enc, own_E, pub_E; torch.cuda.empty_cache()
