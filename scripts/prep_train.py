"""Build the clean training file from ToolRet-Training-20w.

Removes every training row whose question is a copy or near-copy of a ToolRet test
question (normalised exact match, word-set Jaccard >= 0.60 against its nearest test
questions, or embedding cosine >= 0.92), then writes jsonl + a removal report.
"""
import json, re, sys, os, random
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from opentoolsearch.data import load_all_queries, TRAIN_REPO, TRAIN_REV

OUT = sys.argv[1] if len(sys.argv) > 1 else "data"
os.makedirs(OUT, exist_ok=True)
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()

train = load_dataset(TRAIN_REPO, split="train", revision=TRAIN_REV)
test = [q for qs in load_all_queries().values() for q in qs]
test_norm = [norm(q["query"]) for q in test]
test_set = set(test_norm)
test_words = [set(t.split()) for t in test_norm]
tq = [r for r in train["query"]]
tn = [norm(q) for q in tq]
print("train", len(tq), "test", len(test), flush=True)

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", model_kwargs={"torch_dtype": torch.bfloat16})
model.max_seq_length = 256
with torch.no_grad():
    E_test = model.encode(test_norm, batch_size=512, convert_to_tensor=True, normalize_embeddings=True).float()
    E_tr = model.encode(tn, batch_size=1024, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True).float()
    cos, nn = [], []
    for i in range(0, len(tn), 8192):
        s, j = (E_tr[i:i + 8192] @ E_test.T).topk(5, dim=1)
        cos.append(s.cpu()); nn.append(j.cpu())
cos = torch.cat(cos); nn = torch.cat(nn)

reasons, keep = {"exact": 0, "jaccard": 0, "cosine": 0}, []
removed = []
for i in range(len(tn)):
    why = None
    if tn[i] in test_set:
        why = "exact"
    else:
        w = set(tn[i].split())
        for j in nn[i].tolist():
            u = len(w | test_words[j])
            if u and len(w & test_words[j]) / u >= 0.60:
                why = "jaccard"; break
        if why is None and cos[i, 0].item() >= 0.92:
            why = "cosine"
    if why:
        reasons[why] += 1
        removed.append({"train_id": train[i]["id"], "why": why, "test_id": test[nn[i, 0].item()]["id"],
                        "cos": round(cos[i, 0].item(), 4)})
    else:
        keep.append(i)
print("removed", reasons, "kept", len(keep), flush=True)

# How many distinct test questions had a copy in train (for the report)
hit_test = {r["test_id"] for r in removed}
json.dump({"train_rows": len(tn), "removed": reasons, "kept": len(keep),
           "test_questions_with_a_train_copy": len(hit_test)}, open(f"{OUT}/dedup_report.json", "w"), indent=1)
with open(f"{OUT}/removed_train_ids.jsonl", "w") as f:
    for r in removed:
        f.write(json.dumps(r) + "\n")

random.seed(0)
random.shuffle(keep)
dev = set(keep[:2000])
tok = model.tokenizer
with open(f"{OUT}/toolret_train_clean.jsonl", "w") as ft, open(f"{OUT}/toolret_dev.jsonl", "w") as fd:
    for i in keep:
        r = train[i]
        row = {"id": r["id"], "query": r["query"], "instruction": r["prompt"],
               "pos": list(r["positive"]), "neg": list(r["negative"]), "src": "toolret-train"}
        (fd if i in dev else ft).write(json.dumps(row, ensure_ascii=False) + "\n")

# token length stats on a sample
samp = [train[i] for i in keep[:3000]]
import numpy as np
dl = [len(tok(d)["input_ids"]) for r in samp for d in r["positive"][:1] + r["negative"][:1]]
ql = [len(tok(r["prompt"] + r["query"])["input_ids"]) for r in samp]
print("doc tokens p50/p90/p99/max", np.percentile(dl, [50, 90, 99, 100]))
print("query tokens p50/p90/p99/max", np.percentile(ql, [50, 90, 99, 100]))
print("pos per row", np.mean([len(r["positive"]) for r in samp]), "neg per row", np.mean([len(r["negative"]) for r in samp]))
