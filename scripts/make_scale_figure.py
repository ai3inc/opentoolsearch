"""Figure: throughput vs quality as the reranker goes deeper (one B300)."""
import json, math, os, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open(os.path.join(ROOT, "results", "scale_curve.json")))
BG, INK, INK2, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#8f8e86", "#383835"
HUE = {"50": "#d95926", "25": "#e07a4f", "10": "#7fb3d5", "5": "#9ed0b0", "0": "#199e70"}
FONT = "-apple-system,SF Pro Text,Helvetica Neue,Arial,sans-serif"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
pts = [D["embedder_only"]] + D["rows"]
W, H = 1280, 700
t = lambda x, y, s, size=14, fill=INK2, anchor="start", weight=400: \
    f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>'
x0, y0, w, h = 96, 200, 700, 400
XL, XH = 5e3, 3e6
X = lambda v: x0 + (math.log10(max(v, XL)) - math.log10(XL)) / (math.log10(XH) - math.log10(XL)) * w
YL, YH = 55, 95
Y = lambda v: y0 + h - (v - YL) / (YH - YL) * h
body = t(48, 52, "How fast can one GPU search?", 27, INK, weight=700)
body += t(48, 80, "Throughput against quality as the reranker looks at more candidates. One B300, a 1,000-tool library. Further right and higher is better.", 15.5, INK2)
for gy in (60, 70, 80, 90):
    body += f'<line x1="{x0}" y1="{Y(gy)}" x2="{x0 + w}" y2="{Y(gy)}" stroke="{GRID}"/>' + t(x0 - 12, Y(gy) + 4, gy, 12, MUTED, "end")
for gx, lab in ((1e3, "1k/h"), (1e4, "10k/h"), (1e5, "100k/h"), (1e6, "1M/h")):
    body += f'<line x1="{X(gx)}" y1="{y0}" x2="{X(gx)}" y2="{y0 + h}" stroke="{GRID}"/>' + t(X(gx), y0 + h + 20, lab, 12, MUTED, "middle")
body += t(x0 + w / 2, y0 + h + 44, "searches an hour on one B300 (log scale)", 13, MUTED, "middle")
body += t(48, y0 - 6, "public-benchmark nDCG@10", 12.5, MUTED)
body += f'<rect x="860" y="130" width="380" height="520" rx="12" fill="#22201d"/>'
body += t(884, 164, "Rerank depth", 15, INK, weight=700) + t(884, 190, "candidates scored per search", 12.5, MUTED)
rows = [(d, lab) for d, lab in ((0, "none (embedder only)"), (10, "10"), (25, "25 (default)"), (50, "50"))]
body += t(904, 222, "nDCG@10 (public)", 12, MUTED) + t(1010, 222, "per search", 12, MUTED) + t(1100, 222, "/ hour", 12, MUTED) + t(1190, 222, "/ 1,000", 12, MUTED)
for i, (d, lab) in enumerate(rows):
    r = D["embedder_only"] if d == 0 else next(x for x in D["rows"] if x["rerank_depth"] == d)
    yy = 250 + i * 70
    body += f'<rect x="884" y="{yy - 18}" width="12" height="12" rx="3" fill="{HUE[str(d)]}"/>'
    body += t(904, yy - 6, lab, 15, INK, weight=700 if d == 25 else 400)
    body += t(904, yy + 14, f'{r["quality_public"]:.1f}', 13, INK2) + t(1010, yy + 14, f'{r["ms_per_search"]:.0f} ms', 13, INK2) + t(1100, yy + 14, f'{r["searches_per_hour"]:,}', 13, INK2) + t(1190, yy + 14, f'${r["usd_per_1k_our_pod"]:.3f}', 13, INK2)
for p in pts:
    cx, cy = X(p["searches_per_hour"]), Y(p["quality_public"])
    d = str(p["rerank_depth"])
    body += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="12" fill="{HUE[d]}" stroke="{BG}" stroke-width="3"/>'
labels = {0: (-16, 34, "end"), 5: (18, -10, "start"), 10: (24, 26, "start"), 25: (-18, -22, "end"), 50: (18, -16, "start")}
for p in pts:
    cx, cy = X(p["searches_per_hour"]), Y(p["quality_public"])
    dx, dy, anc = labels[p["rerank_depth"]]
    lab = {0: "embedder only", 5: "rerank 5", 10: "rerank 10", 25: "rerank 25 (default)", 50: "rerank 50"}[p["rerank_depth"]]
    body += t(cx + dx, cy + dy, lab, 14.5, INK, anc, 700 if p["rerank_depth"] == 25 else 500)
    body += t(cx + dx, cy + dy + 18, f'{p["searches_per_hour"]:,}/h · {p["ms_per_search"]:.0f} ms', 12.5, MUTED, anc)
body += t(48, H - 40, "Depth 25 is the default: 0.2 nDCG@10 below depth 50, double the throughput at 29,000 searches an hour. Measurements: 200 requests, waves of 8, one B300; quality recomputed from the stored per-question scores.", 12.5, MUTED)
svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}"><rect width="{W}" height="{H}" fill="{BG}"/>{body}</svg>'
html = os.path.join(ROOT, "data", "img", "scale.html"); open(html, "w").write(f'<!doctype html><html><body style="margin:0;background:{BG}">{svg}</body></html>')
out = os.path.join(ROOT, "assets", "scale.png")
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2", f"--window-size={W},{H}", f"--screenshot={out}", "file://" + html], capture_output=True)
print("wrote", out)
