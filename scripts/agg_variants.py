import json, sys
from collections import defaultdict
from opentoolsearch.data import TASK_2_CATEGORY
for p in sys.argv[1:]:
    a = json.load(open(p))["agg"]["NDCG@10"]
    s = defaultdict(float); n = defaultdict(int)
    for t, v in a["task"].items():
        c = TASK_2_CATEGORY[t]; s[c] += v * a["task_n"][t]; n[c] += a["task_n"][t]
    cm = {c: s[c] / n[c] for c in s}
    print(p, "| cat-micro:", {c: round(100 * v, 2) for c, v in cm.items()}, "Avg", round(100 * sum(cm.values()) / 3, 2),
          "| task-macro Avg", round(100 * a["avg"], 2))
    print("  tasks:", {t: (round(100 * v, 1), a["task_n"][t]) for t, v in sorted(a["task"].items())})
