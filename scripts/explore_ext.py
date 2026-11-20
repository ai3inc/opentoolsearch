import json, sys
from datasets import load_dataset
from huggingface_hub import list_repo_files
fs = list_repo_files("Agent-Ark/Toucan-1.5M", repo_type="dataset")
import collections
print(collections.Counter(f.split("/")[0] for f in fs))
ds = load_dataset("Agent-Ark/Toucan-1.5M", data_files="Kimi-K2/train-00000-of-00040.parquet", split="train")
print(ds)
r = ds[0]
for k, v in r.items():
    print("##", k, ":", str(v)[:900])
print(collections.Counter(ds["subset_name"] if "subset_name" in ds.column_names else ds["subset"]))
g = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
print(g); print({k: str(v)[:1200] for k, v in g[1].items()})
