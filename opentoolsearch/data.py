"""ToolRet benchmark data loading (official release, pinned revisions)."""
import json
from datasets import load_dataset, concatenate_datasets

QUERY_REPO = "mangopy/ToolRet-Queries"
QUERY_REV = "b8c76ad3349ff17497b6bdb28bb5b8f61a0f6445"
TOOL_REPO = "mangopy/ToolRet-Tools"
TOOL_REV = "e06c38c75612b6536bd959e08cdd345894aba6a7"
TRAIN_REPO = "mangopy/ToolRet-Training-20w"
TRAIN_REV = "fdf5a317455b1e60785de7ba587496aa6cc878e4"

# Same mapping as toolret/config.py in mangopy/tool-retrieval-benchmark.
TASK_2_CATEGORY = {
    "craft-math-algebra": "code", "craft-tabmwp": "code", "craft-vqa": "code",
    "gorilla-huggingface": "code", "gorilla-pytorch": "code", "gorilla-tensor": "code",
    "toolink": "code",
    "apibank": "web", "apigen": "web", "mnms": "web", "reversechain": "web",
    "rotbench": "web", "t-eval-dialog": "web", "t-eval-step": "web",
    "taskbench-daily": "web", "toolace": "web", "toolbench": "web", "toolemu": "web",
    "tooleyes": "web", "toollens": "web", "ultratool": "web", "autotools-food": "web",
    "autotools-music": "web", "autotools-weather": "web", "restgpt-spotify": "web",
    "restgpt-tmdb": "web",
    "appbench": "customized", "gpt4tools": "customized", "gta": "customized",
    "taskbench-huggingface": "customized", "taskbench-multimedia": "customized",
    "metatool": "customized", "tool-be-honest": "customized", "toolalpaca": "customized",
    "toolbench-sam": "customized",
}
CATEGORIES = ["web", "code", "customized"]
TASKS = sorted(TASK_2_CATEGORY)
# Source datasets that ToolRet-Training-20w draws on (test questions from these are
# "seen-source"; the other 32 subsets are "unseen-source").
SEEN_SOURCE_TASKS = {"toolbench", "apigen", "toolace"}


def load_tools():
    """All 44,453 tools, concatenated in category order web, code, customized."""
    parts = []
    for c in CATEGORIES:
        ds = load_dataset(TOOL_REPO, c, revision=TOOL_REV)["tools"]
        ds = ds.add_column("category", [c] * len(ds))
        parts.append(ds)
    return concatenate_datasets(parts)


def load_queries(task):
    ds = load_dataset(QUERY_REPO, task, revision=QUERY_REV)["queries"]
    out = []
    for r in ds:
        labels = json.loads(r["labels"]) if isinstance(r["labels"], str) else r["labels"]
        out.append({
            "id": str(r["id"]), "query": r["query"], "instruction": r.get("instruction") or "",
            "qrels": {str(x["id"]): int(x.get("relevance", 1)) for x in labels},
            "task": task, "category": TASK_2_CATEGORY[task],
        })
    return out


def load_all_queries():
    return {t: load_queries(t) for t in TASKS}
