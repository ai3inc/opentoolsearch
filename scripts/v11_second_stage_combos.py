"""Second-stage combinations behind three first stages: our reranker, Tool-Rank-4B (the reranker of Tool-DE), and both.

Pre-registered before the run (nothing below is chosen on the results): first stages OpenToolSearch Embed 0.6B (CB query format),
Tool-Embed-4B and Tool-Embed-0.6B (the tool prompt, as in the dedicated-retriever runs); each hands its top 25 to
  ours     = ai3inc/opentoolsearch-rerank-0.6b
  toolrank = Lux1997/Tool-Rank-4B, pointwise yes/no with the Qwen3-Reranker prompt (as in scripts/v7_tool_de.py)
and four scorings are compared, all z-scored over the 25 candidates of a request, first-stage weight 0.5 as in the library:
  A  z(ours) + 0.5 z(first)                 the library default
  B  z(toolrank)                            Tool-DE's ordering
  C  z(toolrank) + 0.5 z(first)
  D  z(ours) + z(toolrank) + 0.5 z(first)
The raw scores are stored, so scripts/v11_score.py can compute these (and nothing else is reported as a result).
Own set in English and Korean, ToolRet per-source (769 questions). Also times the rerankers (waves of 8) with the first stage OpenToolSearch Embed.

  python scripts/v11_second_stage_combos.py OUT_DIR
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
    "Nemotron-3-Embed-8B": dict(path="nvidia/Nemotron-3-Embed-8B-BF16", pool="st", dp="passage: ", maxlen=512, q="query: {q}"),
    "Qwen3-Embedding-0.6B": dict(path="Qwen/Qwen3-Embedding-0.6B", pool="st", dp="", maxlen=512, q=CB),
    "OpenToolSearch-Embed-0.6B": dict(path="ai3inc/opentoolsearch-embed-0.6b", pool="st", dp="", maxlen=512, q=CB),
}

DEPTH, W, TOPN = 25, 0.5, 25
COMBO_FIRST = {
    "OpenToolSearch-Embed-0.6B": dict(path="ai3inc/opentoolsearch-embed-0.6b", pool="st", dp="", maxlen=512, q=CB),
    "Tool-Embed-4B": dict(path="Lux1997/Tool-Embed-4B", pool="st", dp="", maxlen=512, q=TD_TOOL),
    "Tool-Embed-0.6B": dict(path="Lux1997/Tool-Embed-0.6B", pool="st", dp="", maxlen=512, q=TD_TOOL),
}
ours = Reranker("ai3inc/opentoolsearch-rerank-0.6b", dtype=torch.bfloat16); ours.model.eval()
toolrank = Reranker("Lux1997/Tool-Rank-4B", dtype=torch.bfloat16); toolrank.model.eval()
CACHE = {}          # (benchmark, question id, tool id) -> (ours, toolrank), shared by the first stages (their candidates overlap)


def score_pairs(items):
    """items: [(bench, qid, tid, query, text)] -> fills CACHE for the missing ones, longest texts first."""
    todo = [it for it in items if (it[0], it[1], it[2]) not in CACHE]
    todo = sorted({(a, b, c): (d, e) for a, b, c, d, e in todo}.items(), key=lambda kv: -len(kv[1][1]))
    for i in range(0, len(todo), 96):
        chunk = todo[i:i + 96]; pairs = [("", q, t) for _, (q, t) in chunk]
        with torch.no_grad():
            a = ours.score(pairs).tolist(); b = toolrank.score(pairs).tolist()
        for (k, _), x, y in zip(chunk, a, b):
            CACHE[k] = (x, y)


def collect(bench, qids, queries, cand, scores, texts, ids, out):
    items = [(bench, qid, ids[j], q, texts[j]) for qid, q, row in zip(qids, queries, cand) for j in row[:DEPTH]]
    score_pairs(items)
    for qid, row, fs in zip(qids, cand, scores):
        row = row[:DEPTH]
        out[qid] = {"ids": [ids[j] for j in row], "first": [round(x, 5) for x in fs[:len(row)]],
                    "ours": [round(CACHE[(bench, qid, ids[j])][0], 4) for j in row], "toolrank": [round(CACHE[(bench, qid, ids[j])][1], 4) for j in row]}


for name, spec in COMBO_FIRST.items():
    dst = f"{OUT}/combo_{name}.json"
    if os.path.exists(dst):
        print("skip", name, flush=True); continue
    t0 = time.time(); spec = {**spec, "variants": {"x": spec["q"]}}; enc = Enc(spec); tmpl = spec["q"]
    own_E = enc.encode([spec["dp"] + t for t in own_tools])
    pub_E = {task: enc.encode([spec["dp"] + x["text"] for x in pool]) for task, pool in pools.items()}
    res = {"first_stage": spec["path"], "second_stages": {"ours": "ai3inc/opentoolsearch-rerank-0.6b", "toolrank": "Lux1997/Tool-Rank-4B"}, "depth": DEPTH,
           "own_en": {}, "own_ko": {}, "public": {}}
    for key, qs in own_q.items():
        Q = enc.encode([tmpl.format(q=q) for q in qs])
        s_, j_ = (Q @ own_E.T).topk(TOPN, dim=1)
        collect(key, own_qid, qs, j_.cpu().tolist(), s_.cpu().tolist(), own_tools, own_ids, res[key])
        print(name, key, f"{time.time()-t0:.0f}s", flush=True)
    for task, pool in pools.items():
        cs = [c for c in cases if c["task"] == task]
        if not cs:
            continue
        Q = enc.encode([tmpl.format(q=c["query"]) for c in cs])
        s_, j_ = (Q @ pub_E[task].T).topk(min(TOPN, len(pool)), dim=1)
        collect("public", [c["id"] for c in cs], [c["query"] for c in cs], j_.cpu().tolist(), s_.cpu().tolist(), [x["text"] for x in pool], [x["id"] for x in pool], res["public"])
    if name == "OpenToolSearch-Embed-0.6B":
        # time of the second stages on our own set, 25 candidates, waves of 8 (embedding time measured the same way)
        qs = own_q["own_en"]; sync = torch.cuda.synchronize; timing = {}
        cand = {qid: [own_ids.index(i) for i in res["own_en"][qid]["ids"]] for qid in own_qid}
        for lab, models in (("embed", []), ("ours", [ours]), ("toolrank", [toolrank]), ("ours+toolrank", [ours, toolrank])):
            for rep in range(2):
                sync(); t1 = time.time()
                for a in range(0, len(qs), 8):
                    wave = qs[a:a + 8]
                    Q = enc.encode([tmpl.format(q=q) for q in wave])
                    if models:
                        pairs = [("", q, own_tools[j]) for q, qid in zip(wave, own_qid[a:a + 8]) for j in cand[qid]]
                        with torch.no_grad():
                            for m in models:
                                for b in range(0, len(pairs), 256): m.score(pairs[b:b + 256])
                sync(); el = time.time() - t1
            timing[lab] = round(1000 * el / len(qs), 1)
        res["ms_per_search_waves_of_8"] = timing; print("timing", timing, flush=True)
    json.dump(res, open(dst, "w"))
    print(f"done {name}: {time.time()-t0:.0f}s, cache {len(CACHE)}", flush=True)
    del enc, own_E, pub_E; torch.cuda.empty_cache()
print("done all", flush=True)
