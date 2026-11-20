"""Cross-encoder reranker (yes/no head of a causal LM, Qwen3-Reranker format).

Train:  python -m opentoolsearch.rerank train --base Qwen/Qwen3-Reranker-0.6B --data a.jsonl --out runs/r1
Eval :  python -m opentoolsearch.rerank eval --model runs/r1/final --run results/h1/run_inst.json --out results/h1_r1
"""
import argparse, json, math, os, random, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from .augment import augment_doc

SYS = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
       'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_TASK = "Given a user request, retrieve the tools that are needed to fulfil it"


class Reranker:
    def __init__(self, path, q_len=384, d_len=640, dtype=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype or torch.float32).to(self.device)
        self.yes = self.tok.convert_tokens_to_ids("yes"); self.no = self.tok.convert_tokens_to_ids("no")
        self.pre = self.tok.encode(SYS, add_special_tokens=False)
        self.suf = self.tok.encode(SUF, add_special_tokens=False)
        self.q_len, self.d_len = q_len, d_len

    def build(self, pairs):
        """pairs: [(instruction, query, doc)] -> padded batch on GPU"""
        ids = []
        for inst, q, d in pairs:
            a = self.tok.encode(f"<Instruct>: {inst or DEFAULT_TASK}\n<Query>: {q}", add_special_tokens=False)[: self.q_len]
            b = self.tok.encode(f"\n<Document>: {d}", add_special_tokens=False)[: self.d_len]
            ids.append(self.pre + a + b + self.suf)
        L = max(len(x) for x in ids)
        pad = self.tok.pad_token_id
        inp = torch.tensor([[pad] * (L - len(x)) + x for x in ids]).to(self.device)
        att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids]).to(self.device)
        return inp, att

    def score(self, pairs):
        inp, att = self.build(pairs)
        with torch.autocast(self.device if self.device != "mps" else "cpu", dtype=torch.bfloat16,
                            enabled=self.device == "cuda"):
            h = self.model.model(input_ids=inp, attention_mask=att).last_hidden_state[:, -1]
            w = self.model.lm_head.weight[[self.yes, self.no]]
            lg = h.to(w.dtype) @ w.T
        return (lg[:, 0] - lg[:, 1]).float()


def train(a):
    random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/train_args.json", "w"), indent=1)
    rows = []
    for spec in a.data:
        path, _, w = spec.partition(":")
        r = [json.loads(l) for l in open(path)]
        r = [x for x in r if x.get("neg")]
        if w and float(w) < 1:
            random.shuffle(r); r = r[: int(len(r) * float(w))]
        print(path, len(r), flush=True); rows += r
    rr = Reranker(a.base, a.q_len, a.d_len)
    rr.model.train()
    rr.model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(rr.model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup) *
                                              max(0.0, 0.5 * (1 + math.cos(math.pi * s / a.steps))))
    t0 = time.time(); G = 1 + a.negs
    log = open(f"{a.out}/train_log.jsonl", "a")
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        tot, acc = 0.0, 0.0
        for _ in range(a.accum):
            batch = random.sample(rows, a.bs)
            pairs = []
            for r in batch:
                inst = r.get("instruction") if random.random() >= a.p_drop_inst else ""
                negs = random.sample(r["neg"], min(a.negs, len(r["neg"])))
                while len(negs) < a.negs:           # pad with random docs from other rows
                    o = random.choice(rows); negs.append(random.choice(o["pos"]))
                negs = [d for d in negs]
                for d in [random.choice(r["pos"])] + negs:
                    pairs.append((inst, r["query"], augment_doc(d, random, a.p_aug) if a.p_aug > 0 else d))
            s = rr.score(pairs).view(a.bs, G)
            # a padded "negative" may coincide with a positive: mask exact matches
            tgt = torch.zeros(a.bs, dtype=torch.long, device=s.device)
            loss_list = F.cross_entropy(s, tgt)
            lab = torch.zeros_like(s); lab[:, 0] = 1
            loss_point = F.binary_cross_entropy_with_logits(s, lab)
            loss = (loss_list + a.point_w * loss_point) / a.accum
            loss.backward()
            tot += loss.item(); acc += (s.argmax(1) == 0).float().mean().item() / a.accum
        torch.nn.utils.clip_grad_norm_(rr.model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 10 == 0 or step == a.steps - 1:
            rec = {"step": step, "loss": round(tot, 4), "acc": round(acc, 4), "elapsed": round(time.time() - t0)}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if a.save_every and (step + 1) % a.save_every == 0 and step + 1 < a.steps:
            rr.model.save_pretrained(f"{a.out}/step{step+1}"); rr.tok.save_pretrained(f"{a.out}/step{step+1}")
    rr.model.save_pretrained(f"{a.out}/final"); rr.tok.save_pretrained(f"{a.out}/final")
    print("saved", flush=True)


@torch.no_grad()
def evaluate(a):
    from .data import load_tools, load_all_queries
    from .evaluate import score as score_run
    from .metrics import fmt
    run = json.load(open(a.run))
    tools = load_tools(); doc = {str(t["id"]): t["documentation"] for t in tools}
    queries = load_all_queries()
    rr = Reranker(a.model, a.q_len, a.d_len, dtype=torch.bfloat16); rr.model.eval()
    items = []   # (task, qid, docid, inst, query, doc)
    for task, qs in queries.items():
        for q in qs:
            cands = sorted(run[task][q["id"]].items(), key=lambda x: -x[1])[: a.topn]
            for did, _ in cands:
                items.append((task, q["id"], did, q["instruction"] if a.inst else "", q["query"], doc[did]))
    print("pairs", len(items), flush=True)
    order = sorted(range(len(items)), key=lambda i: -(len(items[i][5]) + len(items[i][4])))
    scores = [0.0] * len(items); t0 = time.time()
    for b in range(0, len(order), a.bs):
        idx = order[b:b + a.bs]
        s = rr.score([(items[i][3], items[i][4], items[i][5]) for i in idx]).tolist()
        for i, v in zip(idx, s):
            scores[i] = v
        if (b // a.bs) % 200 == 0:
            print(f"  {b}/{len(order)} {time.time()-t0:.0f}s", flush=True)
    out = {t: {} for t in queries}
    first = {}
    for (task, qid, did, *_), s in zip(items, scores):
        out[task].setdefault(qid, {})[did] = s
    # optional fusion with first-stage score (z-normalised per query)
    if a.fuse > 0:
        import statistics
        for task in out:
            for qid, d in out[task].items():
                fs = {k: run[task][qid][k] for k in d}
                def z(m):
                    v = list(m.values()); mu = statistics.mean(v); sd = statistics.pstdev(v) or 1.0
                    return {k: (x - mu) / sd for k, x in m.items()}
                zr, zf = z(d), z(fs)
                out[task][qid] = {k: zr[k] + a.fuse * zf[k] for k in d}
    pq, aggs = score_run(out, queries)
    os.makedirs(a.out, exist_ok=True)
    tag = ("inst" if a.inst else "noinst") + (f"_fuse{a.fuse}" if a.fuse else "")
    json.dump(out, open(f"{a.out}/run_{tag}.json", "w"))
    json.dump({"args": vars(a), "agg": aggs}, open(f"{a.out}/metrics_{tag}.json", "w"), indent=1)
    for m in aggs.values():
        print(fmt(m), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("train")
    t.add_argument("--base", required=True); t.add_argument("--data", nargs="+", required=True)
    t.add_argument("--out", required=True); t.add_argument("--bs", type=int, default=24)
    t.add_argument("--accum", type=int, default=2); t.add_argument("--negs", type=int, default=7)
    t.add_argument("--steps", type=int, default=2000); t.add_argument("--lr", type=float, default=6e-6)
    t.add_argument("--warmup", type=int, default=100); t.add_argument("--q_len", type=int, default=320)
    t.add_argument("--d_len", type=int, default=512); t.add_argument("--p_drop_inst", type=float, default=0.1)
    t.add_argument("--point_w", type=float, default=0.5); t.add_argument("--save_every", type=int, default=0)
    t.add_argument("--seed", type=int, default=0); t.add_argument("--p_aug", type=float, default=0.0)
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True); e.add_argument("--run", required=True); e.add_argument("--out", required=True)
    e.add_argument("--inst", action="store_true"); e.add_argument("--topn", type=int, default=50)
    e.add_argument("--bs", type=int, default=256); e.add_argument("--q_len", type=int, default=384)
    e.add_argument("--d_len", type=int, default=768); e.add_argument("--fuse", type=float, default=0.0)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)
