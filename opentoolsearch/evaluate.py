"""Evaluate a retriever on ToolRet under the released protocol:
every query is searched against the whole 44,453-tool corpus, top-100 kept.

  python -m opentoolsearch.evaluate --model Qwen/Qwen3-Embedding-0.6B --inst --out results/q3-0.6b
  python -m opentoolsearch.evaluate --bm25 --inst --out results/bm25
"""
import argparse, json, os, time
import numpy as np
import torch
from .data import load_tools, load_all_queries
from .metrics import per_query, aggregate, fmt

DEFAULT_TASK = "Given a web search query, retrieve relevant passages that answer the query"


def format_query(q, inst, template):
    task = q["instruction"] if (inst and q["instruction"]) else None
    if template == "instruct":          # Qwen3 / e5-mistral / gte-Qwen2 style
        return f"Instruct: {task or DEFAULT_TASK}\nQuery: {q['query']}"
    if template == "instruct_opt":      # instruction only when present
        return f"Instruct: {task}\nQuery: {q['query']}" if task else q["query"]
    if template == "concat":
        return f"{task}\n{q['query']}" if task else q["query"]
    if template == "inst_only":         # the instruction alone is the search text
        return f"Instruct: {DEFAULT_TASK}\nQuery: {task or q['query']}"
    if template == "inst_first":        # instruction as task and again in front of the request
        return f"Instruct: {task or DEFAULT_TASK}\nQuery: {(task + chr(10)) if task else ''}{q['query']}"
    if template == "plain":
        return q["query"]
    raise ValueError(template)


def run_bm25(tools, queries, inst, topk):
    import bm25s, Stemmer
    stemmer = Stemmer.Stemmer("english")
    corpus = [t["documentation"] for t in tools]
    ids = [str(t["id"]) for t in tools]
    retr = bm25s.BM25()
    retr.index(bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer))
    results = {}
    for task, qs in queries.items():
        texts = [format_query(q, inst, "concat") for q in qs]
        idx, sc = retr.retrieve(bm25s.tokenize(texts, stopwords="en", stemmer=stemmer), k=topk)
        results[task] = {q["id"]: {ids[int(i)]: float(s) for i, s in zip(ii, ss)}
                         for q, ii, ss in zip(qs, idx, sc)}
    return results


@torch.no_grad()
def encode(model, texts, bs, max_len, desc=""):
    model.max_seq_length = max_len
    order = np.argsort([-len(t) for t in texts])
    out = None
    t0 = time.time()
    for i in range(0, len(texts), bs):
        b = order[i:i + bs]
        e = model.encode([texts[j] for j in b], batch_size=len(b), convert_to_tensor=True,
                         normalize_embeddings=True, show_progress_bar=False).float()
        if out is None:
            out = torch.empty(len(texts), e.shape[1], device=e.device)
        out[torch.as_tensor(b.copy(), device=e.device)] = e
    print(f"  encoded {len(texts)} {desc} in {time.time()-t0:.0f}s", flush=True)
    return out


def run_dense(args, tools, queries):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, trust_remote_code=True,
                                model_kwargs={"torch_dtype": torch.bfloat16},
                                tokenizer_kwargs={"padding_side": "left"} if args.left_pad else {})
    model.eval()
    docs = [t["documentation"] for t in tools]
    if args.doc_prefix:
        docs = [args.doc_prefix + d for d in docs]
    ids = [str(t["id"]) for t in tools]
    D = encode(model, docs, args.bs, args.doc_len, "tools")
    results = {}
    for task, qs in queries.items():
        Q = encode(model, [args.q_prefix + format_query(q, args.inst, args.template) for q in qs],
                   args.bs, args.query_len, f"queries[{task}]")
        sc, idx = (Q @ D.T).topk(args.topk, dim=1)
        sc, idx = sc.cpu().numpy(), idx.cpu().numpy()
        results[task] = {q["id"]: {ids[int(i)]: float(s) for i, s in zip(ii, ss)}
                         for q, ii, ss in zip(qs, idx, sc)}
    return results


def score(results, queries, drop=None):
    pq = {t: per_query({q["id"]: q["qrels"] for q in qs}, results[t]) for t, qs in queries.items()}
    return pq, {m: aggregate(pq, m, drop) for m in ("NDCG@10", "Recall@10", "Comp@10", "P@10", "Recall@100")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--bm25", action="store_true")
    ap.add_argument("--inst", action="store_true", help="w/ instruction setting")
    ap.add_argument("--template", default="instruct")
    ap.add_argument("--doc_prefix", default="")
    ap.add_argument("--q_prefix", default="")
    ap.add_argument("--left_pad", action="store_true")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--doc_len", type=int, default=2048)
    ap.add_argument("--query_len", type=int, default=1024)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tools = load_tools()
    queries = load_all_queries()
    print(f"tools {len(tools)} | queries {sum(len(v) for v in queries.values())} | tasks {len(queries)}", flush=True)
    results = run_bm25(tools, queries, args.inst, args.topk) if args.bm25 else run_dense(args, tools, queries)
    pq, aggs = score(results, queries)
    os.makedirs(args.out, exist_ok=True)
    tag = ("inst" if args.inst else "noinst") + ("" if args.template == "instruct" else "_" + args.template)
    with open(f"{args.out}/run_{tag}.json", "w") as f:
        json.dump(results, f)
    with open(f"{args.out}/metrics_{tag}.json", "w") as f:
        json.dump({"args": vars(args), "agg": aggs}, f, indent=1)
    for m in aggs.values():
        print(fmt(m), flush=True)


if __name__ == "__main__":
    main()
