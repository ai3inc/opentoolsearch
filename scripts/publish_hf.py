"""Publish a staged release folder to the Hugging Face Hub.

  HF_TOKEN=... python scripts/publish_hf.py REPO_ID FOLDER --public|--private

Visibility must be given explicitly. Refuses to touch a repository that already exists
unless --update is passed, and prints who the token belongs to before uploading.
"""
import os, sys
from huggingface_hub import HfApi

repo, folder = sys.argv[1], sys.argv[2]
flags = set(sys.argv[3:])
if ("--public" in flags) == ("--private" in flags):
    sys.exit("say exactly one of --public or --private")
private = "--private" in flags
api = HfApi(token=os.environ["HF_TOKEN"])
me = api.whoami()
orgs = [o["name"] for o in me.get("orgs", [])]
print("token user:", me.get("name"), "| orgs:", orgs)
ns = repo.split("/")[0]
if ns != me.get("name") and ns not in orgs:
    sys.exit(f"token has no access to namespace {ns!r}")
exists = api.repo_exists(repo, repo_type="model")
if exists and "--update" not in flags:
    sys.exit(f"{repo} already exists - pass --update to upload into it")
if not exists:
    api.create_repo(repo, repo_type="model", private=private, exist_ok=False)
info = api.model_info(repo)
if bool(info.private) != private:
    sys.exit(f"visibility mismatch: repo private={info.private}, asked private={private}")
api.upload_folder(folder_path=folder, repo_id=repo, repo_type="model",
                  commit_message=os.environ.get("HF_MSG", "OpenToolSearch Rerank 0.6B"))
print("published:", f"https://huggingface.co/{repo}", "| private:", api.model_info(repo).private)
