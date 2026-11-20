"""Metrics and aggregation, matching the ToolRet release (pytrec_eval) and the
leaderboard convention: mean over a query set per task, mean of tasks per category,
mean of the three categories as Avg."""
from collections import defaultdict
import pytrec_eval
from .data import TASK_2_CATEGORY, CATEGORIES, SEEN_SOURCE_TASKS

K = (1, 5, 10, 20, 100)


def per_query(qrels, results):
    """qrels: {qid: {docid: rel}}, results: {qid: {docid: score}} -> {qid: {metric: v}}"""
    ks = ",".join(str(k) for k in K)
    ev = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut.{ks}", f"recall.{ks}", f"P.{ks}"})
    raw = ev.evaluate(results)
    out = {}
    for qid, s in raw.items():
        m = {}
        for k in K:
            m[f"NDCG@{k}"] = s[f"ndcg_cut_{k}"]
            m[f"Recall@{k}"] = s[f"recall_{k}"]
            m[f"P@{k}"] = s[f"P_{k}"]
            m[f"Comp@{k}"] = float(s[f"recall_{k}"] == 1)
        out[qid] = m
    return out


def aggregate(pq_by_task, metric="NDCG@10", drop_qids=None):
    """pq_by_task: {task: {qid: {metric: v}}}. Returns dict with per-task, per-category,
    Avg (leaderboard), micro (all queries), and seen/unseen-source breakdown."""
    drop_qids = drop_qids or set()
    task_score, task_n = {}, {}
    for t, pq in pq_by_task.items():
        vals = [m[metric] for q, m in pq.items() if (t, q) not in drop_qids]
        if vals:
            task_score[t] = sum(vals) / len(vals)
            task_n[t] = len(vals)
    cat = defaultdict(list)
    for t, s in task_score.items():
        cat[TASK_2_CATEGORY[t]].append(s)
    cat_score = {c: sum(v) / len(v) for c, v in cat.items()}
    avg = sum(cat_score[c] for c in CATEGORIES if c in cat_score) / len(cat_score)
    n = sum(task_n.values())
    micro = sum(task_score[t] * task_n[t] for t in task_score) / n

    def cat_avg(tasks):
        c2 = defaultdict(list)
        for t in tasks:
            if t in task_score:
                c2[TASK_2_CATEGORY[t]].append(task_score[t])
        if not c2:
            return None
        return sum(sum(v) / len(v) for v in c2.values()) / len(c2)

    unseen = [t for t in task_score if t not in SEEN_SOURCE_TASKS]
    seen = [t for t in task_score if t in SEEN_SOURCE_TASKS]
    return {
        "metric": metric, "avg": avg, "micro": micro, "n": n,
        "category": cat_score, "task": task_score, "task_n": task_n,
        "unseen_source_avg": cat_avg(unseen),
        "seen_source_mean": (sum(task_score[t] for t in seen) / len(seen)) if seen else None,
    }


def fmt(agg):
    c = agg["category"]
    return (f"{agg['metric']}: Avg {100*agg['avg']:.2f} | web {100*c.get('web',0):.2f} "
            f"code {100*c.get('code',0):.2f} cust {100*c.get('customized',0):.2f} | "
            f"micro {100*agg['micro']:.2f} (n={agg['n']}) | unseen-src {100*(agg['unseen_source_avg'] or 0):.2f} "
            f"seen-src {100*(agg['seen_source_mean'] or 0):.2f}")
