"""Find the right skill for what the user just said - a folder of SKILL.md files as a tool library.

  python examples/skill_search.py ~/.claude/skills "make a promo video for the BioQ launch"

Each skill is indexed by its name and description. The index is cached in ./.skill_index (override with
SKILL_INDEX); on the next run only skills that are new or whose text changed are embedded, and skills that
disappeared are dropped - so a skill the user wrote a minute ago is searchable right away and nothing else is
re-embedded. Set SKILL_RERANK=0 to skip the reranker (embedding search only; the right choice on a CPU).
In an agent, pass the last few turns of the conversation as the query, not only the last line.
"""
import glob, os, re, sys
from opentoolsearch import ToolIndex


def load_skills(root):
    skills = []
    for path in sorted(glob.glob(os.path.join(os.path.expanduser(root), "*", "SKILL.md"))):
        head = re.match(r"^---\n(.*?)\n---", open(path, encoding="utf-8").read(), re.S)
        if not head:
            continue
        fields = {m.group(1): " ".join(m.group(2).split()).strip("\"'")
                  for m in re.finditer(r"^([\w-]+):\s*(.*?)(?=^[\w-]+:|\Z)", head.group(1), re.S | re.M)}
        name = fields.get("name") or os.path.basename(os.path.dirname(path))
        if fields.get("description"):
            skills.append({"name": name, "description": fields["description"]})
    return skills


if __name__ == "__main__":
    root, query = sys.argv[1], " ".join(sys.argv[2:])
    cache = os.environ.get("SKILL_INDEX", ".skill_index")
    kw = {} if os.environ.get("SKILL_RERANK", "1") != "0" else {"reranker": None}
    index = ToolIndex.load(cache, **kw) if os.path.exists(os.path.join(cache, "meta.json")) else ToolIndex(**kw)
    print(index.sync(load_skills(root)))            # {'added': .., 'updated': .., 'removed': .., 'unchanged': ..}
    index.save(cache)
    for h in index.search(query, k=5):
        print(f"{h['score']:6.2f}  {h['name']:32s} {h['tool']['description'][:90]}")
