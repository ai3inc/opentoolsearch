"""Weight-space interpolation of causal-LM rerankers.  python scripts/soup_lm.py OUT A:0.5 B:0.5"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
out, specs = sys.argv[1], [s.rsplit(":", 1) for s in sys.argv[2:]]
ms = [AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float32) for p, _ in specs]
ws = [float(w) for _, w in specs]; sd = [m.state_dict() for m in ms]
avg = {k: sum(w * s[k] for w, s in zip(ws, sd)) if sd[0][k].is_floating_point() else sd[0][k] for k in sd[0]}
ms[0].load_state_dict(avg); ms[0].save_pretrained(out)
AutoTokenizer.from_pretrained(specs[-1][0]).save_pretrained(out); print("saved", out, ws)
