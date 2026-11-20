from datasets import load_dataset
from opentoolsearch.data import load_tools, load_queries, TRAIN_REPO, TRAIN_REV
ds = load_dataset(TRAIN_REPO, split="train", revision=TRAIN_REV)
print(ds)
for i in (0, 100000):
    for k, v in ds[i].items():
        print("##", k, ":", str(v)[:600])
t = load_tools(); print(t); print({k: str(v)[:300] for k, v in t[0].items()})
for task in ("toolbench", "gorilla-pytorch", "metatool"):
    q = load_queries(task); print(task, len(q)); print(q[0])
