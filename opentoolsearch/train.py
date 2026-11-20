"""Contrastive fine-tuning for tool retrieval with GradCache (large batches on one GPU).

Each training row: {"query", "instruction", "pos": [doc,...], "neg": [doc,...], "src"}.
A batch holds B queries; each brings one sampled positive and `--negs` hard negatives.
Every other document in the batch is an in-batch negative unless its text is one of the
query's own positives (masked out).

  python -m opentoolsearch.train --base Qwen/Qwen3-Embedding-0.6B --data data/a.jsonl:1 data/b.jsonl:0.5 \
      --out runs/v1 --bs 512 --negs 3 --steps 3000
"""
import argparse, json, math, os, random, time
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from .augment import augment_doc

DEFAULT_TASK = "Given a web search query, retrieve relevant passages that answer the query"


def load_rows(specs):
    rows = []
    for spec in specs:
        path, _, w = spec.partition(":")
        w = float(w or 1)
        with open(path) as f:
            r = [json.loads(l) for l in f]
        if w < 1:
            random.shuffle(r); r = r[: int(len(r) * w)]
        elif w > 1:
            r = r * int(w)
        print(f"  {path}: {len(r)} rows", flush=True)
        rows += r
    return rows


def fmt_query(r, p_drop, template="instruct", prefix=""):
    inst = r.get("instruction") or ""
    use = bool(inst) and random.random() >= p_drop
    if template == "concat":            # models with a fixed query prefix (e.g. "query: ")
        return prefix + (f"{inst}\n{r['query']}" if use else r["query"])
    return prefix + f"Instruct: {inst if use else DEFAULT_TASK}\nQuery: {r['query']}"


class Encoder(torch.nn.Module):
    def __init__(self, st):
        super().__init__()
        self.st = st

    def forward(self, texts, max_len):
        self.st.max_seq_length = max_len
        feats = self.st.tokenize(texts)
        feats = {k: (v.to(self.st.device) if torch.is_tensor(v) else v) for k, v in feats.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.st(feats)["sentence_embedding"]
        return F.normalize(out.float(), dim=-1)


def chunks(n, size, lens):
    """Index chunks sorted by length so padding is small."""
    order = sorted(range(n), key=lambda i: -lens[i])
    return [order[i:i + size] for i in range(0, n, size)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--negs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.02)
    ap.add_argument("--q_len", type=int, default=384)
    ap.add_argument("--d_len", type=int, default=768)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--p_drop_inst", type=float, default=0.1)
    ap.add_argument("--p_aug", type=float, default=0.0, help="prob. of re-rendering a tool in a random layout")
    ap.add_argument("--template", default="instruct")
    ap.add_argument("--q_prefix", default="")
    ap.add_argument("--d_prefix", default="")
    ap.add_argument("--same_src", action="store_true", help="draw each batch from one source")
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(f"{args.out}/train_args.json", "w"), indent=1)

    rows = load_rows(args.data)
    by_src = {}
    for r in rows:
        by_src.setdefault(r.get("src", "x"), []).append(r)
    srcs = list(by_src); src_w = [len(by_src[s]) for s in srcs]
    print(f"rows {len(rows)} | sources {dict(zip(srcs, src_w))}", flush=True)

    st = SentenceTransformer(args.base, trust_remote_code=True)
    st.train()
    enc = Encoder(st)
    opt = torch.optim.AdamW([p for p in st.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup) *
                                              max(0.0, 0.5 * (1 + math.cos(math.pi * s / args.steps))))
    log = open(f"{args.out}/train_log.jsonl", "a")
    t0 = time.time()
    for step in range(args.steps):
        if args.same_src:
            pool = by_src[random.choices(srcs, weights=src_w)[0]]
            batch = random.sample(pool, min(args.bs, len(pool)))
        else:
            batch = random.sample(rows, args.bs)
        qs, docs, pos_sets = [], [], []
        for r in batch:
            qs.append(fmt_query(r, args.p_drop_inst, args.template, args.q_prefix))
            p = random.choice(r["pos"])
            ng = random.sample(r["neg"], min(args.negs, len(r["neg"]))) if r.get("neg") else []
            ng = [d for d in ng if d not in r["pos"]]
            docs.append([p] + ng)
            pos_sets.append(set(r["pos"]))
        flat, tgt = [], []                    # flat = original texts (identity for masking)
        for d in docs:
            tgt.append(len(flat)); flat += d
        tgt = torch.tensor(tgt, device=st.device)
        # mask in-batch documents that are also positives of the query
        idx_of = {}
        for j, d in enumerate(flat):
            idx_of.setdefault(d, []).append(j)
        mask = torch.zeros(len(qs), len(flat), dtype=torch.bool, device=st.device)
        for i, ps in enumerate(pos_sets):
            for p in ps:
                for j in idx_of.get(p, ()):
                    mask[i, j] = True
            mask[i, tgt[i]] = False
        # what the encoder reads: optional layout augmentation + prefix
        flat = [args.d_prefix + (augment_doc(x, random, args.p_aug) if args.p_aug > 0 else x) for x in flat]

        # ---- GradCache: 1) embed without grad
        q_ch = chunks(len(qs), args.chunk, [len(x) for x in qs])
        d_ch = chunks(len(flat), args.chunk, [len(x) for x in flat])
        with torch.no_grad():
            Q = torch.empty(len(qs), st.get_sentence_embedding_dimension(), device=st.device)
            D = torch.empty(len(flat), Q.shape[1], device=st.device)
            for c in q_ch:
                Q[c] = enc([qs[i] for i in c], args.q_len)
            for c in d_ch:
                D[c] = enc([flat[i] for i in c], args.d_len)
        # ---- 2) loss + grads w.r.t. embeddings
        Q.requires_grad_(True); D.requires_grad_(True)
        logits = (Q @ D.T) / args.temp
        logits = logits.masked_fill(mask, -1e4)
        loss = F.cross_entropy(logits, tgt)
        loss.backward()
        gQ, gD = Q.grad.detach(), D.grad.detach()
        acc = (logits.argmax(1) == tgt).float().mean().item()
        # ---- 3) re-embed with grad, chunk by chunk
        opt.zero_grad(set_to_none=True)
        for c in q_ch:
            (enc([qs[i] for i in c], args.q_len) * gQ[c]).sum().backward()
        for c in d_ch:
            (enc([flat[i] for i in c], args.d_len) * gD[c]).sum().backward()
        torch.nn.utils.clip_grad_norm_(st.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 10 == 0 or step == args.steps - 1:
            rec = {"step": step, "loss": round(loss.item(), 4), "acc": round(acc, 4),
                   "lr": sched.get_last_lr()[0], "docs": len(flat), "elapsed": round(time.time() - t0)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, flush=True)
        if args.save_every and (step + 1) % args.save_every == 0 and step + 1 < args.steps:
            st.save(f"{args.out}/step{step+1}")
    st.save(f"{args.out}/final")
    print("saved", f"{args.out}/final", flush=True)


if __name__ == "__main__":
    main()
