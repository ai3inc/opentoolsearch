"""Candidate equivalents of each dev gold tool: its nearest pool tools by document embedding.
Writes pairs for an LLM judge, and caches pool embeddings.
  python scripts/v2_equiv_candidates.py data/v2
"""
import json, sys, torch
from sentence_transformers import SentenceTransformer
from opentoolsearch.evaluate import encode
D = sys.argv[1]
pool = [json.loads(l) for l in open(f"{D}/dev_pool.jsonl")]
dev = [json.loads(l) for l in open(f"{D}/dev_questions.jsonl")]
ix = {p["id"]: i for i, p in enumerate(pool)}
m = SentenceTransformer("nvidia/Nemotron-3-Embed-1B-BF16", model_kwargs={"torch_dtype": torch.bfloat16}); m.eval()
E = encode(m, ["passage: " + p["text"] for p in pool], 256, 512, "dev pool")
torch.save(E.half().cpu(), f"{D}/dev_pool_emb.pt")
gold = sorted({f"t:{g}" for c in dev for g in c["gold"]})
G = E[[ix[g] for g in gold]]
s, j = (G @ E.T).topk(13, dim=1)
n = 0
with open(f"{D}/equiv_candidates.jsonl", "w") as f:
    for gi, g in enumerate(gold):
        for sc, k in zip(s[gi].tolist(), j[gi].tolist()):
            if pool[k]["id"] == g or sc < 0.70:
                continue
            f.write(json.dumps({"id": f"{g}||{pool[k]['id']}", "gold": g, "cand": pool[k]["id"], "cos": round(sc, 4),
                                "a": pool[ix[g]]["text"][:900], "b": pool[k]["text"][:900]}, ensure_ascii=False) + "\n"); n += 1
print("gold tools", len(gold), "candidate pairs", n)
