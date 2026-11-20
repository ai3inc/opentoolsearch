"""Open a pull request on the ToolRet leaderboard Space with our updated spreadsheets.

  HF_TOKEN=... python scripts/leaderboard_pr.py XLSX_DIR "TITLE" DESCRIPTION.md [--dry-run]
"""
import os, sys
from huggingface_hub import HfApi, CommitOperationAdd
d, title, desc = sys.argv[1], sys.argv[2], open(sys.argv[3]).read()
files = sorted(f for f in os.listdir(d) if f.endswith(".xlsx"))
print("files:", files)
if "--dry-run" in sys.argv:
    print(title); print(desc[:600]); sys.exit()
api = HfApi(token=os.environ["HF_TOKEN"])
ops = [CommitOperationAdd(path_in_repo=f"data/{f}", path_or_fileobj=os.path.join(d, f)) for f in files]
c = api.create_commit(repo_id="mangopy/ToolRet-leaderboard", repo_type="space", operations=ops,
                      commit_message=title, commit_description=desc, create_pr=True)
print("PR:", c.pr_url)
