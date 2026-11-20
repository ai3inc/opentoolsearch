"""Attach generated instructions to external rows -> ext_train.jsonl
(rows still waiting for an instruction are kept with an empty one)."""
import json, sys
inst = json.load(open(sys.argv[2]))
n = m = 0
with open(sys.argv[3], "w") as f:
    for l in open(sys.argv[1]):
        r = json.loads(l)
        r["instruction"] = inst.get(r["id"], ""); r.pop("brief", None)
        n += 1; m += bool(r["instruction"])
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("rows", n, "with instruction", m)
