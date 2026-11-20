"""For ToolRet questions whose top-ranked tool is not a labelled answer, write
(labelled tool, top-ranked tool) pairs for the interchangeability judge.
  python scripts/toolret_top1_pairs.py FUSED_RUN.json OUT.jsonl
"""
import json, sys
from opentoolsearch.data import load_tools, load_all_queries
run = json.load(open(sys.argv[1]))
doc = {str(t["id"]): t["documentation"] for t in load_tools()}
n = wrong = 0
with open(sys.argv[2], "w") as f:
    for task, qs in load_all_queries().items():
        for q in qs:
            n += 1
            top = max(run[task][q["id"]].items(), key=lambda x: x[1])[0]
            if top in q["qrels"]:
                continue
            wrong += 1
            for g in list(q["qrels"])[:3]:
                f.write(json.dumps({"id": f"{task}|{q['id']}|{g}", "gold": g, "cand": top, "cos": 0.0, "task": task,
                                    "a": doc[g][:900], "b": doc[top][:900]}, ensure_ascii=False) + "\n")
print("questions", n, "top-1 not a labelled tool", wrong)
