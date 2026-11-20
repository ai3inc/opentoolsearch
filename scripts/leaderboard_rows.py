"""Add our row to copies of the ToolRet leaderboard spreadsheets.

  python scripts/leaderboard_rows.py LB_DIR OUT_DIR MODEL_NAME PARAMS INST.json NOINST.json

LB_DIR holds the Space's data/*.xlsx; INST/NOINST are the OUTJSON files written by fuse.py.
Only the two settings the Space displays are touched: w-w (with instruction), w-wo (without).
"""
import json, os, shutil, sys
import openpyxl
lb, out, name, params, inst, noinst = sys.argv[1:7]
os.makedirs(out, exist_ok=True)
for prefix, path in (("w-w", inst), ("w-wo", noinst)):
    m = json.load(open(path))["metrics"]
    for cat in ("Avg", "Code", "API", "Customized"):
        f = f"{prefix}-{cat}.xlsx"
        wb = openpyxl.load_workbook(os.path.join(lb, f)); ws = wb.active
        hdr = [c.value for c in ws[1]]
        assert hdr[:7] == ["Model", "Comp@10", "Recall@10", "Prec@10", "NDCG@10", "Number of Parameters", "Model Type"], hdr
        row = ws.max_row
        while row > 1 and ws.cell(row, 1).value in (None, ""):
            row -= 1
        vals = [name, round(m["Comp@10"][cat], 5), round(m["Recall@10"][cat], 5), round(m["P@10"][cat], 5),
                round(m["NDCG@10"][cat], 5), params, "re-ranking model"]
        for j, v in enumerate(vals, 1):
            ws.cell(row + 1, j).value = v
        wb.save(os.path.join(out, f))
        print(f, vals)
