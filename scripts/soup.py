"""Weight-space interpolation of sentence-transformers checkpoints.
  python scripts/soup.py OUT A:0.5 B:0.5 [...]
"""
import sys, torch
from sentence_transformers import SentenceTransformer
out, specs = sys.argv[1], [s.rsplit(":", 1) for s in sys.argv[2:]]
models = [SentenceTransformer(p, device="cpu") for p, _ in specs]
ws = [float(w) for _, w in specs]
sd = [m.state_dict() for m in models]
avg = {k: sum(w * s[k].float() for w, s in zip(ws, sd)).to(sd[0][k].dtype) if sd[0][k].is_floating_point() else sd[0][k] for k in sd[0]}
models[0].load_state_dict(avg); models[0].save(out); print("saved", out, ws)
