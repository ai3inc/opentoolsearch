"""Render the README figures from the result files (HTML/SVG -> PNG with headless Chrome).

  python scripts/make_figures.py
Reads results/public_head_to_head.json, results/head_to_head.json and the two timing files.
"""
import json, math, os, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda p: json.load(open(os.path.join(ROOT, "results", p)))
PUB, OWN = R("public_head_to_head.json"), R("head_to_head.json")
T_PUB, T_OWN = R("public_head_to_head_timing.json"), R("head_to_head_timing.json")
HOST = {r["system"]: r for r in R("hosted_tool_search.json")}      # hosted tool search of every vendor, own set
DEF = {r["depth"]: r for r in R("default_config.json")["rows"]}    # depth x fusion sweep; the library default is depth 25, fusion 0.5
D25 = DEF[25]
D50_USD, D50_S = 0.47, 0.22        # depth 50: $ per 1,000 and median seconds, as in the README table


def lowest(r):
    """The lower of list price and cache-aware cost (we give the hosted systems the benefit of the doubt)."""
    v = [r.get("usd_per_1k"), r.get("usd_per_1k_list"), r.get("usd_per_1k_cached")]
    v = [x for x in v if x is not None]
    return min(v) if v else None
GPU_KRW_H, KRW = 9931.0, 1400.0
gpu_usd = lambda t: t["mean_s"] * 1000 / 3600 * GPU_KRW_H / KRW
BG, INK, INK2, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#8f8e86", "#383835"
COL = {"ots": "#d95926", "api": "#3987e5", "base": "#199e70", "dense": "#c9a227"}   # trio of the original figures + amber for dedicated open retrievers
FONT = "-apple-system,SF Pro Text,Helvetica Neue,Arial,sans-serif"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def render(name, w, h, body):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{FONT}"><rect width="{w}" height="{h}" fill="{BG}"/>{body}</svg>'
    html = os.path.join(ROOT, "data", "img", name + ".html")
    os.makedirs(os.path.dirname(html), exist_ok=True)
    open(html, "w").write(f'<!doctype html><html><body style="margin:0;background:{BG}">{svg}</body></html>')
    out = os.path.join(ROOT, "assets", name + ".png")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    f"--window-size={w},{h}", f"--screenshot={out}", "file://" + html], capture_output=True)
    print("wrote", out)


def t(x, y, s, size=14, fill=INK2, anchor="start", weight=400):
    return f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>'


def legend(x, y):
    out = ""
    for k, lab in (("ots", "OpenToolSearch (open weights)"), ("api", "Hosted APIs"), ("base", "Open baselines"), ("dense", "Dedicated open retrievers")):
        out += f'<rect x="{x}" y="{y-11}" width="12" height="12" rx="3" fill="{COL[k]}"/>' + t(x + 19, y, lab, 14)
        x += len(lab) * 8.0 + 50
    return out


# ---------- shared system tables
DENSE = {r["system"]: r for r in R("dense_tool_retrievers.json")}
KO_OPEN = [r for r in R("korean_open.json")]
KO_DEC = {(r["system"], r["candidates"]): r for r in R("korean_decisions.json")}
HOST_KO = {r["system"]: r for r in R("hosted_tool_search_ko.json")}
GPU_USD_H = 7.09


def dense_cost(r):
    """$ per 1,000 searches: embed one request (median ms, one at a time) on a B300 at $7.09/h; the tool vectors are precomputed."""
    return r["latency_ms_p50"] / 1000 * 1000 / 3600 * GPU_USD_H


def dense_rows(key):
    out = []
    for n, r in sorted(DENSE.items(), key=lambda kv: -kv[1][key]["ndcg10"]):
        label = {"callbus-emb-8b": "callbus-emb-8b", "Tool-DE (Tool-Embed-4B + Tool-Rank-4B)": "Tool-DE (Embed-4B + Rank-4B)"}.get(
            n, n.replace("ToolRet-trained-", "ToolRet-trained ").replace("-en-v1.5", "").replace("-v2", ""))
        out.append({"name": label,
                    "grp": "dense", "q": r[key]["ndcg10"], "cost": dense_cost(r), "lat": r["latency_ms_p50"] / 1000})
    return out


# (label, public system name, own-set key, Korean key)
HOSTED = [("Gemini tool search · 4-hydrogen", "Gemini API tool search (gemini-4-hydrogen)", "gemini-4-hydrogen", "gemini-4-hydrogen"),
          ("Gemini tool search · 3.8-flash", "Gemini API tool search (gemini-3.8-flash)", "gemini-3.8-flash", "gemini-3.8-flash"),
          ("Gemini tool search · 3.6-flash", "Gemini API tool search (gemini-3.6-flash)", "gemini-3.6-flash", "gemini-3.6-flash"),
          ("OpenAI tool search · gpt-6.2-sol", "OpenAI tool search (gpt-6.2-sol)", "oai-gpt-6.2-sol-run2", "oai-gpt-6.2-sol"),
          ("OpenAI tool search · gpt-6-luna", "OpenAI tool search (gpt-6-luna)", "oai-gpt-6-luna", "oai-gpt-6-luna"),
          ("Claude tool search · Opus 5.5 max", "Claude tool search (Opus 5.5, effort max)", "ant-opus-5-5-max", "ant-opus-5-5-max"),
          ("Claude tool search · Haiku 5.5", "Claude tool search (Haiku 5.5)", "ant-haiku-5-5-forced", "ant-haiku-5-5")]


def pub_rows():
    m = {r["system"]: r for r in PUB["rows"]}
    S = [("OpenToolSearch · default (depth 25)", "ots", {"ndcg10": D25["public"]["w0.5"]}, D25["usd_per_1k"], D25["ms_per_search"] / 1000),
         ("OpenToolSearch · depth 50", "ots", m["OpenToolSearch"], D50_USD, D50_S),
         ("Qwen3-Reranker-0.6B", "base", m["Qwen3-Reranker-0.6B as released"], D50_USD, D50_S),
         ("Nemotron embedder alone", "base", m["Nemotron-3-Embed-1B alone"], None, None)]
    for k, lab in (("Jev (single choice)", "Jev · choice"), ("Jev (yes/no per tool)", "Jev · yes/no"),
                   ("Gemini Decision (single choice)", "Gemini Decision · choice"), ("Gemini Decision (yes/no per tool)", "Gemini Decision · yes/no")):
        if k in m:
            S.append((lab, "api", m[k], lowest(m[k]), m[k].get("latency_p50")))
    for lab, pn, _, _ in HOSTED:
        if pn in m and m[pn].get("ndcg10") is not None:
            S.append((lab, "api", m[pn], lowest(m[pn]), m[pn].get("latency_p50")))
    rows = [{"name": a, "grp": b, "q": r["ndcg10"], "cost": c, "lat": l} for a, b, r, c, l in S]
    return rows + dense_rows("public")


def own_rows():
    m = {r["system"]: r for r in OWN["rows"]}
    S = [("OpenToolSearch · default (depth 25)", "ots", {"ndcg10_c": D25["own"]["w0.5"]}, D25["usd_per_1k"], D25["ms_per_search"] / 1000),
         ("OpenToolSearch · depth 50", "ots", m["rerank:published"], D50_USD, D50_S),
         ("Qwen3-Reranker-0.6B", "base", m["rerank:base"], D50_USD, D50_S),
         ("Nemotron embedder alone", "base", m["Nemotron-3-Embed-1B alone (first stage)"], None, None),
         ("Gemini Decision · choice", "api", m["Gemini Decision (single choice)"], m["Gemini Decision (single choice)"]["krw_per_1k"] / KRW, m["Gemini Decision (single choice)"]["latency_p50"]),
         ("Jev · choice", "api", m["Jev (single choice)"], m["Jev (single choice)"]["krw_per_1k"] / KRW, m["Jev (single choice)"]["latency_p50"]),
         ("Jev · yes/no", "api", m["Jev (yes/no per tool)"], m["Jev (yes/no per tool)"]["krw_per_1k"] / KRW, m["Jev (yes/no per tool)"]["latency_p50"]),
         ("Gemini Decision · yes/no", "api", m["Gemini Decision (yes/no per tool)"], m["Gemini Decision (yes/no per tool)"]["krw_per_1k"] / KRW, m["Gemini Decision (yes/no per tool)"]["latency_p50"])]
    q = lambda r: r["ndcg10_c"] if "ndcg10_c" in r else r["ndcg10"]
    rows = [{"name": a, "grp": b, "q": q(r), "cost": c, "lat": l} for a, b, r, c, l in S]
    for lab, _, k, _ in HOSTED:
        rows.append({"name": lab, "grp": "api", "q": HOST[k]["ndcg10"], "cost": lowest(HOST[k]), "lat": HOST[k]["latency_p50"]})
    return rows + dense_rows("own_en")


def ko_rows():
    g = {r["system"]: r for r in KO_OPEN if r["lang"] == "ko"}
    d25 = g["OpenToolSearch - default (depth 25)"]; d50 = g["OpenToolSearch - depth 50, fusion 0.25"]
    rows = [{"name": "OpenToolSearch · default (depth 25)", "grp": "ots", "q": d25["ndcg10"], "cost": D25["usd_per_1k"], "lat": D25["ms_per_search"] / 1000},
            {"name": "OpenToolSearch · depth 50", "grp": "ots", "q": d50["ndcg10"], "cost": D50_USD, "lat": D50_S},
            {"name": "Qwen3-Reranker-0.6B", "grp": "base", "q": g["Qwen3-Reranker-0.6B as released - depth 50, fusion 0.25"]["ndcg10"], "cost": D50_USD, "lat": D50_S},
            {"name": "Nemotron embedder alone", "grp": "base", "q": g["Nemotron-3-Embed-1B alone"]["ndcg10"], "cost": None, "lat": None}]
    for nm, lab in (("Jev (single choice)", "Jev · choice"), ("Gemini Decision (single choice)", "Gemini Decision · choice")):
        r = KO_DEC[(nm, 50)]
        rows.append({"name": lab, "grp": "api", "q": r["ndcg10"], "cost": r["usd_per_1k"], "lat": r["latency_p50"]})
    for lab, _, _, k in HOSTED:
        if k in HOST_KO:
            rows.append({"name": lab, "grp": "api", "q": HOST_KO[k]["ndcg10"], "cost": lowest(HOST_KO[k]), "lat": HOST_KO[k]["latency_p50"]})
    return rows + dense_rows("own_ko")


# ---------- scorecard: one row per system, three measures side by side
def scorecard(name, rows, title, sub, qlabel, foot):
    rows = sorted(rows, key=lambda r: -r["q"])
    W, rowh, top = 1280, 50, 176
    H = top + rowh * len(rows) + 30 + 22 * len(foot) + 16
    xq, wq = 330, 330            # quality bars
    xc, wc = 760, 210            # cost, log axis
    xl, wl = 1030, 170           # latency bars
    lmax = max(r["lat"] or 0 for r in rows) * 1.05
    lo, hi = 0.005, 200.0
    CX = lambda c: xc + (math.log10(min(hi, max(lo, c))) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * wc
    body = t(48, 52, title, 27, INK, weight=700) + t(48, 80, sub, 15.5, INK2) + legend(48, 112)
    body += t(xq, top - 22, qlabel, 13, MUTED) + t(xc, top - 22, "cost per 1,000 searches (log scale)", 13, MUTED) + t(xl, top - 22, "median seconds per search", 13, MUTED)
    yb = top + rowh * len(rows)
    for gx in (0, 25, 50, 75, 100):
        X = xq + gx / 100 * wq
        body += f'<line x1="{X}" y1="{top-8}" x2="{X}" y2="{yb-8}" stroke="{GRID}"/>' + t(X, yb + 10, gx, 12, MUTED, "middle")
    for gx, lab in ((0.01, "$0.01"), (0.1, "$0.1"), (1, "$1"), (10, "$10"), (100, "$100")):
        body += f'<line x1="{CX(gx)}" y1="{top-8}" x2="{CX(gx)}" y2="{yb-8}" stroke="{GRID}"/>' + t(CX(gx), yb + 10, lab, 12, MUTED, "middle")
    body += f'<line x1="{xl}" y1="{top-8}" x2="{xl}" y2="{yb-8}" stroke="{GRID}"/>'
    for i, r in enumerate(rows):
        y = top + i * rowh; c = COL[r["grp"]]
        if r["grp"] == "ots":
            body += f'<rect x="36" y="{y+1}" width="{W-72}" height="{rowh-4}" rx="8" fill="#26231f"/>'
        body += t(xq - 16, y + 30, r["name"], 15.5, INK, "end", 700 if r["grp"] == "ots" else 400)
        w = r["q"] / 100 * wq
        body += f'<path d="M{xq},{y+13} h{w-4:.1f} a4,4 0 0 1 4,4 v14 a4,4 0 0 1 -4,4 h-{w-4:.1f} z" fill="{c}"/>' + t(xq + w + 9, y + 30, f'{r["q"]:.1f}', 15, INK, weight=600)
        if r["cost"] is None:
            body += t(xc + 4, y + 30, "not measured", 13, MUTED)
        else:
            cx = CX(r["cost"]); lab = f'${r["cost"]:.2f}' if r["cost"] < 10 else f'${r["cost"]:.0f}'
            body += f'<line x1="{xc}" y1="{y+24}" x2="{cx:.1f}" y2="{y+24}" stroke="{c}" stroke-width="2" opacity="0.55"/><circle cx="{cx:.1f}" cy="{y+24}" r="7" fill="{c}" stroke="{BG}" stroke-width="2"/>'
            body += t(cx + 13, y + 30, lab, 15, INK, weight=600) if cx < xc + wc - 50 else t(cx - 13, y + 12, lab, 14, INK, "end", 600)
        if r["lat"] is None:
            body += t(xl + 4, y + 30, "not measured", 13, MUTED)
        else:
            w = max(3, r["lat"] / lmax * wl)
            body += f'<path d="M{xl},{y+13} h{max(0,w-4):.1f} a4,4 0 0 1 4,4 v14 a4,4 0 0 1 -4,4 h-{max(0,w-4):.1f} z" fill="{c}"/>'
            lab = f'{r["lat"]*1000:.0f} ms' if r["lat"] < 0.05 else (f'{r["lat"]:.2f} s' if r["lat"] < 10 else f'{r["lat"]:.1f} s')
            body += t(xl + w + 8, y + 30, lab, 15, INK, weight=600)
    for k, line in enumerate(foot):
        body += t(48, yb + 46 + 22 * k, line, 12.5, MUTED)
    render(name, W, H, body)


def fig_scorecards():
    scorecard("scorecard-toolret", pub_rows(),
              "Public benchmark: ToolRet, searched within each source dataset",
              f'{PUB["questions"]} questions from {PUB["subsets"]} subsets, the benchmark\'s own labels, request only. Quality up, cost and time down.',
              "nDCG@10 (strict labels)",
              ["Hosted tool search (Gemini, OpenAI, Claude): every tool of the dataset sent deferred, name + short description, the model searches itself. Cost from reported token usage at Nov 2026 list prices:",
               "the lower of list price and cache-aware. Jev, Gemini Decision: as billed by the gateway (₩1,400 = $1); they read the pool up to 200 tools, else the 100 candidates of our first stage.",
               "Open models on one B300 at ₩9,931/h: ours at its measured 29,000 searches an hour (depth 25) or 14,900 (depth 50). Dedicated retrievers: authors' weights, documented query format, one request at a time."])
    scorecard("scorecard-mcp", own_rows(),
              "Our set: OTS-MCP-1k, one library of 1,000 MCP tools",
              f'{OWN["questions"]} requests about MCP servers held out of training, request only. A tool an LLM judge found interchangeable with the labelled one counts.',
              "nDCG@10 (interchangeable tools credited)",
              ["Built by us from Toucan-1.5M, so read it next to the public benchmark. Jev and Gemini Decision rank the 50 candidates our first stage returns; hosted tool search reads all 1,000 tools.",
               "Cost: the lower of list price and cache-aware; with a cold cache list price applies (gpt-6.2-sol $135, gemini-4-hydrogen $88, Opus 5.5 max $105). About 3 points is noise at 200 requests."])
    scorecard("scorecard-korean", ko_rows(),
              "The same 200 requests, in Korean",
              "Requests translated to spoken Korean (Claude Sonnet 5.5, checked by a second model); the 1,000 MCP tools stay in English.",
              "nDCG@10 (interchangeable tools credited)",
              ["English-only retrievers (the four ToolRet-trained models) fall to 13-21 here. Jev and Gemini Decision rank the 50 candidates of our Korean first stage; hosted tool search reads all 1,000 tools.",
               "Costs and timing as in the English figure of this set. About 3 points is sampling noise at 200 requests."])


# ---------- figure 3: ToolRet, the whole 44,453-tool corpus (published systems)
def fig_toolret():
    rows = [("MagicSelector", "LLM query decomposition + 4B + 4B", 59.90, "api"), ("OpenToolSearch", "1.14B + 0.6B", 58.37, "ots"),
            ("Tool-DE", "4B + 4B", 57.01, "api"), ("Open-weights pipeline, untrained", "1.5B + BM25 + 1B", 55.33, "api"),
            ("callbus-emb-8b", "8B, embedding only", 54.36, "api"), ("Nemotron-3-Embed-1B alone", "1.14B, our first stage", 52.42, "base")]
    W, rowh, top, left, barw = 1280, 56, 160, 400, 560
    H = top + rowh * len(rows) + 70
    body = t(48, 52, "ToolRet: all 44,453 tools in one pool", 27, INK, weight=700)
    body += t(48, 80, "nDCG@10, question + instruction, 7,961 questions. Other systems as reported by their authors.", 15.5, INK2)
    body += f'<rect x="48" y="99" width="12" height="12" rx="3" fill="{COL["ots"]}"/>' + t(67, 110, "OpenToolSearch", 14)
    body += f'<rect x="200" y="99" width="12" height="12" rx="3" fill="{COL["api"]}"/>' + t(219, 110, "Other published systems", 14)
    body += f'<rect x="420" y="99" width="12" height="12" rx="3" fill="{COL["base"]}"/>' + t(439, 110, "Open baseline", 14)
    for gx in (0, 20, 40, 60):
        X = left + gx / 70 * barw
        body += f'<line x1="{X}" y1="{top-6}" x2="{X}" y2="{top+rowh*len(rows)-8}" stroke="{GRID}"/>' + t(X, top + rowh * len(rows) + 10, gx, 12, MUTED, "middle")
    for i, (n, size, v, g) in enumerate(rows):
        y = top + i * rowh; w = v / 70 * barw
        body += t(left - 16, y + 20, n, 15.5, INK, "end", 700 if g == "ots" else 500) + t(left - 16, y + 38, size, 12.5, MUTED, "end")
        body += f'<path d="M{left},{y+8} h{w-4:.1f} a4,4 0 0 1 4,4 v20 a4,4 0 0 1 -4,4 h-{w-4:.1f} z" fill="{COL[g]}"/>'
        body += t(left + w + 10, y + 29, f"{v:.2f}", 15, INK, weight=600)
    body += t(48, H - 26, "Model sizes are parameters used at query time. MagicSelector also calls a language model on every query.", 12.5, MUTED)
    render("toolret", W, H, body)


if __name__ == "__main__":
    fig_scorecards(); fig_toolret()
