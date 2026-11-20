"""Print failure examples for given tasks from a run file."""
import json, sys
from opentoolsearch.data import load_tools, load_queries
run = json.load(open(sys.argv[1]))
tools = load_tools(); doc = {str(t["id"]): t["documentation"] for t in tools}
for task in sys.argv[2:]:
    qs = load_queries(task)
    print(f"\n######## {task}  (n={len(qs)})")
    shown = 0
    for q in qs:
        ranked = [k for k, _ in sorted(run[task][q["id"]].items(), key=lambda x: -x[1])]
        top = ranked[:10]
        if any(g in top[:3] for g in q["qrels"]):
            continue
        print("Q:", q["query"][:300].replace("\n", " "))
        print("I:", q["instruction"][:300])
        for g in list(q["qrels"])[:2]:
            rk = (ranked.index(g) + 1) if g in ranked else ">N"
            print(f"  GOLD[{rk}] {g}: {doc[g][:700]}")
        for t in top[:3]:
            print(f"  TOP {t}: {doc[t][:420]}")
        print("  n_gold", len(q["qrels"]))
        shown += 1
        if shown >= 2: break
