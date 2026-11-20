"""Where does reranker time go? tokenise vs forward, at a few batch sizes."""
import json, time, torch, sys
from opentoolsearch.rerank import Reranker
rr = Reranker(sys.argv[1], dtype=torch.bfloat16); rr.model.eval()
rows = [json.loads(l) for _, l in zip(range(400), open("data/mined_hard.jsonl"))]
pairs = [(r["instruction"], r["query"], d) for r in rows for d in (r["pos"][:1] + r["neg"][:7])]
pairs.sort(key=lambda p: -len(p[2]))
print("pairs", len(pairs), "attn impl:", getattr(rr.model.config, "_attn_implementation", None))
for bs in (256, 1024):
    t_tok = t_fwd = 0.0; ntok = 0
    with torch.no_grad():
        for b in range(0, len(pairs), bs):
            t0 = time.time(); inp, att = rr.build(pairs[b:b + bs]); torch.cuda.synchronize(); t1 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = rr.model.model(input_ids=inp, attention_mask=att).last_hidden_state[:, -1]
            torch.cuda.synchronize(); t2 = time.time()
            t_tok += t1 - t0; t_fwd += t2 - t1; ntok += int(att.sum())
    print(f"bs={bs}: tokenise {t_tok:.1f}s forward {t_fwd:.1f}s | {len(pairs)/(t_tok+t_fwd):.0f} pairs/s | {ntok/len(pairs):.0f} tok/pair")
