"""Figure: one GPU, 15,000 searches an hour, and what that costs against hosted search."""
import json, math, os, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda p: json.load(open(os.path.join(ROOT, "results", p)))
PUB, TH, MKT = R("public_head_to_head.json"), R("head_to_head_throughput.json"), R("head_to_head_cost_by_market.json")
P = {r["system"]: r for r in PUB["rows"]}
BG, INK, INK2, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#8f8e86", "#383835"
COL = {"ots": "#d95926", "api": "#3987e5", "base": "#199e70"}
FONT = "-apple-system,SF Pro Text,Helvetica Neue,Arial,sans-serif"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
usd = lambda v: "$6.94" if v == 6.94 else ""
rows = [("OpenToolSearch", "ots", 0.36, "0.16 s", "open weights, our GPU"),
        ("Qwen3-Reranker-0.6B", "base", 0.36, "0.16 s", "released baseline"),
        ("Gemini Decision · choice", "api", P["Gemini Decision (single choice)"]["usd_per_1k"], f'{P["Gemini Decision (single choice)"]["latency_p50"]:.2f} s', "hosted decision API"),
        ("Jev · choice", "api", P["Jev (single choice)"]["usd_per_1k"], f'{P["Jev (single choice)"]["latency_p50"]:.2f} s', "hosted decision API"),
        ("Jev · yes/no", "api", P["Jev (yes/no per tool)"]["usd_per_1k"], f'{P["Jev (yes/no per tool)"]["latency_p50"]:.2f} s', "hosted decision API"),
        ("Gemini Decision · yes/no", "api", P["Gemini Decision (yes/no per tool)"]["usd_per_1k"], f'{P["Gemini Decision (yes/no per tool)"]["latency_p50"]:.2f} s', "hosted decision API"),
        ("Gemini API tool search · 3.6-flash", "api", P["Gemini API tool search (gemini-3.6-flash)"]["usd_per_1k"], f'{P["Gemini API tool search (gemini-3.6-flash)"]["latency_p50"]:.1f} s', "hosted tool search"),
        ("Gemini API tool search · 4-hydrogen", "api", P["Gemini API tool search (gemini-4-hydrogen)"]["usd_per_1k"], f'{P["Gemini API tool search (gemini-4-hydrogen)"]["latency_p50"]:.1f} s', "hosted tool search")]
W, H = 1280, 768
t = lambda x, y, s, size=14, fill=INK2, anchor="start", weight=400: \
    f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>'
body = t(48, 54, "One B300, fifteen thousand searches an hour", 28, INK, weight=700)
body += t(48, 82, "Measured on a 1,000-tool library. Bars below: cost of 1,000 searches on the public benchmark, log scale.", 15.5, INK2)
hero = [("0.22 s", "median search"), ("15,000", "searches / hour"), ("$0.42", "per 1,000 searches"), ("65-120x", "cheaper than hosted search")]
hx = 48
for big, small in hero:
    body += f'<rect x="{hx}" y="102" width="288" height="104" rx="12" fill="#24211d"/>'
    body += t(hx + 20, y=152, s=big, size=34, fill="#ffb08f" if big.startswith("$") else INK, weight=700)
    body += t(hx + 20, y=181, s=small, size=14, fill=MUTED)
    hx += 300
x0, barw, y0, rowh = 420, 520, 268, 46
lo, hi = 0.1, 100.0
X = lambda v: x0 + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * barw
body += t(48, y0 - 12, "cost of 1,000 searches", 13, MUTED) + t(x0 + barw + 150, y0 - 12, "median time per search", 13, MUTED, "end")
for gx, lab in ((0.1, "$0.10"), (1, "$1"), (10, "$10"), (100, "$100")):
    body += f'<line x1="{X(gx)}" y1="{y0}" x2="{X(gx)}" y2="{y0 + rowh * len(rows)}" stroke="{GRID}"/>' + t(X(gx), y0 + rowh * len(rows) + 18, lab, 12, MUTED, "middle")
for i, (n, g, cost, lat, note) in enumerate(rows):
    y = y0 + i * rowh
    if g == "ots":
        body += f'<rect x="36" y="{y - 4}" width="{W - 72}" height="{rowh - 2}" rx="8" fill="#26231f"/>'
    body += t(x0 - 16, y + 18, n, 15, INK, "end", 700 if g == "ots" else 400) + t(x0 - 16, y + 34, note, 12, MUTED, "end")
    w = X(cost) - x0
    body += f'<path d="M{x0},{y + 6} h{max(0, w - 4):.1f} a4,4 0 0 1 4,4 v12 a4,4 0 0 1 -4,4 h-{max(0, w - 4):.1f} z" fill="{COL[g]}"/>'
    body += t(X(cost) + 8, y + 21, f"${cost:.2f}" if cost < 10 else f"${cost:.0f}", 14, INK, weight=600)
    body += t(x0 + barw + 150, y + 21, lat, 15, INK if g == "ots" else INK2, "end", 600 if g == "ots" else 400)
body += t(48, H - 36, "Open-model cost: measured time on a B300 rented at Runpod's community-cloud price, $6.94/GPU-hour. Hosted: what the provider or gateway charged (Gemini from its reported token usage at", 12.5, MUTED)
body += t(48, H - 16, "list prices of Nov 2026; Jev and Gemini Decision as billed by the gateway, KRW at 1,400 per USD). Jev and Gemini Decision read only the 50 candidates our first stage hands them.", 12.5, MUTED)
svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}"><rect width="{W}" height="{H}" fill="{BG}"/>{body}</svg>'
html = os.path.join(ROOT, "data", "img", "economics.html"); open(html, "w").write(f'<!doctype html><html><body style="margin:0;background:{BG}">{svg}</body></html>')
out = os.path.join(ROOT, "assets", "economics.png")
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2", f"--window-size={W},{H}", f"--screenshot={out}", "file://" + html], capture_output=True)
print("wrote", out)
