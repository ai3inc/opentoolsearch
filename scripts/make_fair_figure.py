"""Fair comparison: identical first stage (ours), reranker replaced by a hosted decision model."""
import json, math, os, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "results", "fair_comparison.json")))
BG, INK, INK2, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#8f8e86", "#383835"
COL = {"ots": "#d95926", "jev": "#3987e5", "gdec": "#7fb3d5"}
FONT = "-apple-system,SF Pro Text,Helvetica Neue,Arial,sans-serif"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
t = lambda x, y, s, size=14, fill=INK2, anchor="start", weight=400: \
    f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>'
W, H = 1280, 640
body = t(48, 52, "Same first stage, different second stage", 27, INK, weight=700)
body += t(48, 80, "Our embedder finds the candidates on both benchmarks; the reranker is ours or a hosted decision model. Cost adds both stages.", 15.5, INK2)
body += f'<rect x="48" y="100" width="12" height="12" rx="3" fill="{COL["ots"]}"/>{t(67, 111, "OpenToolSearch", 14)}'
body += f'<rect x="230" y="100" width="12" height="12" rx="3" fill="{COL["jev"]}"/>{t(249, 111, "Jev as the reranker", 14)}'
body += f'<rect x="430" y="100" width="12" height="12" rx="3" fill="{COL["gdec"]}"/>{t(449, 111, "Gemini Decision as the reranker", 14)}'
panels = [("public", "Public benchmark (ToolRet per-source)", 40), ("own", "Our own set (OTS-MCP-1k)", 650)]
for key, title, x0 in panels:
    m = R[key]
    body += t(x0, 148, title, 18, INK, weight=700)
    y0, rowh, w = 210, 92, 370
    for j, depth in enumerate(("25", "50")):
        yy = y0 + j * 210
        body += t(x0, yy - 18, f"{depth} candidates reranked", 14, MUTED, weight=600)
        rows = [("OpenToolSearch", "ots", m[f"ots_{depth}"], 0.24 if depth == "25" else 0.475),
                ("Jev", "jev", m[f"jev_{depth}"], (m[f"dec_cost_{depth}"]["jev"] + (0.24 if depth == "25" else 0.475))),
                ("Gemini Decision", "gdec", m[f"gdec_{depth}"], (m[f"dec_cost_{depth}"]["gdec"] + (0.24 if depth == "25" else 0.475)))]
        best = max(r[2] for r in rows)
        for k, (name, g, q, cost) in enumerate(rows):
            y = yy + k * 28
            hw = q / 100 * w
            body += f'<path d="M{x0+132},{y} h{max(0,hw-4):.1f} a4,4 0 0 1 4,4 v10 a4,4 0 0 1 -4,4 h-{max(0,hw-4):.1f} z" fill="{COL[g]}"/>'
            body += t(x0 + 124, y + 14, name, 13.5, INK, "end", 700 if g == "ots" else 400)
            body += t(x0 + 132 + w + 10, y + 14, f"{q:.1f}", 13.5, INK, weight=600) + t(x0 + 132 + w + 56, y + 14, f"${cost:.2f}", 13, MUTED)
    body += t(x0 + 132 + w + 10, y0 - 44, "nDCG", 11.5, MUTED) + t(x0 + 132 + w + 52, y0 - 44, "cost / 1,000", 11.5, MUTED)
body += t(48, H - 46, "Cost = our first stage ($0.24 at 25 candidates, $0.48 at 50) plus what the decision model charged for the same candidates. Quality: nDCG@10, public = benchmark labels,", 12.5, MUTED)
body += t(48, H - 27, "own set = interchangeable tools credited. About 3 points is sampling noise at 200 to 769 questions, so the honest reading is: the decision models add 6-9 nDCG points for 1.3-2x the cost.", 12.5, MUTED)
svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}"><rect width="{W}" height="{H}" fill="{BG}"/>{body}</svg>'
html = os.path.join(ROOT, "data", "img", "fair.html"); open(html, "w").write(f'<!doctype html><html><body style="margin:0;background:{BG}">{svg}</body></html>')
out = os.path.join(ROOT, "assets", "fair.png")
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2", f"--window-size={W},{H}", f"--screenshot={out}", "file://" + html], capture_output=True)
print("wrote", out)
