"""Assemble the public release folder for the reranker and check that the model card's
transformers-only snippet runs on it.

  python scripts/stage_release.py SRC_WEIGHTS_DIR CARD_DIR OUT_DIR
"""
import re, shutil, sys, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

src, card, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
m = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16)      # same dtype the evaluation used
m.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(src).save_pretrained(out)
for f in os.listdir(card):
    shutil.copy(os.path.join(card, f), os.path.join(out, f))
print("files:", sorted(os.listdir(out)))
print("size GB:", round(sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 1e9, 2))

# run the card's snippet verbatim against the local folder
text = open(os.path.join(card, "README.md")).read()
code = re.search(r"### With transformers only\n\n```python\n(.*?)```", text, re.S).group(1)
code = code.replace('name = "ai3inc/opentoolsearch-rerank-0.6b"', f'name = {out!r}')
ns = {}
exec(code, ns)
score = ns["score"]
right = '{"name": "send_email", "description": "Send an email to a recipient."}'
wrong = '{"name": "get_stock_price", "description": "Get the latest stock price for a ticker symbol."}'
a, b = score("email the Q3 numbers to finance", right), score("email the Q3 numbers to finance", wrong)
print("snippet scores: matching tool", round(a, 2), "| other tool", round(b, 2))
assert a > b, "card snippet ranks the wrong tool higher"
print("CARD SNIPPET OK")
