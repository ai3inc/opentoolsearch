"""Fill the <!-- table:NAME --> ... <!-- /table:NAME --> blocks of README.md / the model card from the result files.
   python scripts/update_tables.py README.md hf/opentoolsearch-rerank-0.6b/README.md
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as mt
FN = {"summary": mt.table_summary, "one": mt.table_one, "hosted": mt.table_hosted, "dense": mt.table_dense, "korean": mt.table_korean}
for path in sys.argv[1:]:
    s = open(path).read(); n = 0
    for name, fn in FN.items():
        pat = re.compile(rf"(<!-- table:{name} -->).*?(<!-- /table:{name} -->)", re.S)
        s, k = pat.subn(lambda m: m.group(1) + "\n" + fn() + "\n" + m.group(2), s); n += k
    open(path, "w").write(s); print(path, "tables filled:", n)
