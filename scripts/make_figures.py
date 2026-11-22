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
FSR = {r["first_stage"]: r for r in R("first_stage_plus_reranker.json")}
_SP = os.path.join(ROOT, "results", "default_speed.json")
SPEED = json.load(open(_SP)) if os.path.exists(_SP) else None       # the default pair, measured through the library (scripts/v10_default_speed.py)
if SPEED:
    NEW_MS, NEW_USD, NEW_MEAS = SPEED["waves"]["8"]["ms_per_search"], SPEED["waves"]["8"]["usd_per_1k_at_7.09"], True
else:
    NEW_MS, NEW_USD, NEW_MEAS = D25["ms_per_search"], D25["usd_per_1k"], False   # the earlier default's timing, shown with ≈
D50_USD, D50_S = 0.47, 0.22        # depth 50: $ per 1,000 and median seconds, as in the README table


def lowest(r):
    """The lower of list price and cache-aware cost (we give the hosted systems the benefit of the doubt)."""
    v = [r.get("usd_per_1k"), r.get("usd_per_1k_list"), r.get("usd_per_1k_cached")]
    v = [x for x in v if x is not None]
    return min(v) if v else None
GPU_KRW_H, KRW = 9931.0, 1400.0
gpu_usd = lambda t: t["mean_s"] * 1000 / 3600 * GPU_KRW_H / KRW
BG, INK, INK2, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#8f8e86", "#383835"
COL = {"ots": "#d95926", "api": "#3987e5", "base": "#199e70", "dense": "#c9a227", "dec": "#9b7be0"}   # trio of the original figures + amber for dedicated open retrievers
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
    S = [("OpenToolSearch · default (depth 25)", "ots", {"ndcg10": FSR["OpenToolSearch-Embed-0.6B"]["with_reranker"]["public"]}, NEW_USD, NEW_MS / 1000),
         ("OpenToolSearch · Nemotron embedder (earlier default)", "ots", {"ndcg10": D25["public"]["w0.5"]}, D25["usd_per_1k"], D25["ms_per_search"] / 1000),
         ("OpenToolSearch · Nemotron embedder, depth 50", "ots", m["OpenToolSearch"], D50_USD, D50_S),
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
    S = [("OpenToolSearch · default (depth 25)", "ots", {"ndcg10_c": FSR["OpenToolSearch-Embed-0.6B"]["with_reranker"]["own_en"]}, NEW_USD, NEW_MS / 1000),
         ("OpenToolSearch · Nemotron embedder (earlier default)", "ots", {"ndcg10_c": D25["own"]["w0.5"]}, D25["usd_per_1k"], D25["ms_per_search"] / 1000),
         ("OpenToolSearch · Nemotron embedder, depth 50", "ots", m["rerank:published"], D50_USD, D50_S),
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
    rows = [{"name": "OpenToolSearch · default (depth 25)", "grp": "ots", "q": FSR["OpenToolSearch-Embed-0.6B"]["with_reranker"]["own_ko"], "cost": NEW_USD, "lat": NEW_MS / 1000},
            {"name": "OpenToolSearch · Nemotron embedder (earlier default)", "grp": "ots", "q": d25["ndcg10"], "cost": D25["usd_per_1k"], "lat": D25["ms_per_search"] / 1000},
            {"name": "OpenToolSearch · Nemotron embedder, depth 50", "grp": "ots", "q": d50["ndcg10"], "cost": D50_USD, "lat": D50_S},
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
               "Open models on one B300 at ₩9,931/h: ours at its measured throughput (waves of 8 requests, depth 25; the depth-50 row is the earlier Nemotron-embedder pipeline, 14,900 an hour). Dedicated retrievers: authors' weights, documented query format, one request at a time."])
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


# ---------- summary: the README's one picture - the pitch on top, quality on three benchmarks + cost + latency below
OPEN_ROWS = ["Tool-DE (Embed-4B + Rank-4B)", "callbus-emb-8b"]
HOSTED_ROWS = ["Gemini tool search · 4-hydrogen", "Gemini tool search · 3.8-flash", "OpenAI tool search · gpt-6.2-sol", "OpenAI tool search · gpt-6-luna",
               "Claude tool search · Opus 5.5 max", "Claude tool search · Haiku 5.5"]
ACCENT, AMBER, SKY = "#ff7a3d", "#e8b931", "#4c9cff"


def fig_summary():
    P, O, K = ({r["name"]: r for r in f()} for f in (pub_rows, own_rows, ko_rows))
    get = lambda n, g: {"name": n, "grp": g, "q": [P[n]["q"], O[n]["q"], K[n]["q"]], "cost": P[n]["cost"], "lat": P[n]["lat"]}
    opens = [get(n, "ots" if n.startswith("OpenToolSearch") else "dense") for n in OPEN_ROWS]
    nw = FSR["OpenToolSearch-Embed-0.6B"]["with_reranker"]
    opens.append({"name": "OpenToolSearch · default (depth 25)", "grp": "ots", "q": [nw["public"], nw["own_en"], nw["own_ko"]],
                  "cost": NEW_USD, "lat": NEW_MS / 1000, "est": not NEW_MEAS})
    sso = R("second_stage_combos.json")["OpenToolSearch-Embed-0.6B"]; sb = sso["variants"]["B"]; tms = sso["ms_per_search_waves_of_8"]["toolrank"]
    opens.append({"name": "OpenToolSearch + Tool-Rank-4B", "grp": "ots", "q": [sb["public"], sb["own_en"], sb["own_ko"]], "cost": tms / 1000 * GPU_USD_H / 3600 * 1000, "lat": tms / 1000})
    cb = FSR["callbus-emb-8b"]["with_reranker"]; cbs = R("default_speed_callbus.json")["waves"]["8"]           # measured end to end through the library
    opens.append({"name": "OpenToolSearch + callbus-emb-8b", "grp": "ots", "q": [cb["public"], cb["own_en"], cb["own_ko"]], "cost": cbs["usd_per_1k_at_7.09"], "lat": cbs["ms_per_search"] / 1000})
    opens.sort(key=lambda r: (r["grp"] != "ots", r["name"] != "OpenToolSearch · default (depth 25)", -r["q"][0]))
    hosted = sorted([get(n, "api") for n in HOSTED_ROWS], key=lambda r: -r["q"][0])
    W = 1280
    qx, qw = [352, 520, 688], 112
    xc, wc = 892, 168
    xl, wl = 1116, 78
    lo, hi = 0.1, 100.0
    CX = lambda c: xc + (math.log10(min(hi, max(lo, c))) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * wc
    rowh, y0 = 48, 436
    H = y0 + 34 + rowh * len(opens) + 40 + rowh * len(hosted) + 160
    d = ('<defs>'
         '<radialGradient id="bgA" cx="18%" cy="0%" r="75%"><stop offset="0" stop-color="#3a2013" stop-opacity="0.95"/><stop offset="0.55" stop-color="#1d1a18" stop-opacity="0.6"/><stop offset="1" stop-color="#141413" stop-opacity="0"/></radialGradient>'
         '<radialGradient id="bgB" cx="100%" cy="100%" r="60%"><stop offset="0" stop-color="#14263d" stop-opacity="0.7"/><stop offset="1" stop-color="#141413" stop-opacity="0"/></radialGradient>'
         '<linearGradient id="title" x1="0" x2="1"><stop offset="0" stop-color="#ffb066"/><stop offset="0.55" stop-color="#ff6a2b"/><stop offset="1" stop-color="#e8b931"/></linearGradient>'
         '<linearGradient id="hl" x1="0" x2="1"><stop offset="0" stop-color="#ff7a3d" stop-opacity="0.30"/><stop offset="0.6" stop-color="#ff7a3d" stop-opacity="0.08"/><stop offset="1" stop-color="#ff7a3d" stop-opacity="0.02"/></linearGradient>'
         '<linearGradient id="card" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#2a2724"/><stop offset="1" stop-color="#1f1d1b"/></linearGradient>'
         '<filter id="glow" x="-30%" y="-80%" width="160%" height="260%"><feGaussianBlur stdDeviation="7" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
         '</defs>')
    body = d + f'<rect width="{W}" height="{H}" fill="url(#bgA)"/><rect width="{W}" height="{H}" fill="url(#bgB)"/>'
    # ---- hero
    body += t(48, 50, "OPEN-SOURCE  ·  APACHE-2.0  ·  BY AI3 INC.", 12.5, MUTED, weight=600).replace('<text ', '<text letter-spacing="2.4" ')
    body += f'<text x="46" y="118" font-size="62" font-weight="800" fill="url(#title)" letter-spacing="-1">OpenToolSearch</text>'
    body += t(48, 156, "Tool search your agent can run itself: the right few tools out of thousands,", 22, INK, weight=500)
    body += t(48, 184, "found in a tenth of a second, for pennies, with weights you own.", 22, INK, weight=500)
    a = "" if NEW_MEAS else "≈"
    cards = [(f"{a}{NEW_MS/1000:.2f} s", "per search", "25–120× faster than hosted tool search", ACCENT),
             (f"{a}${NEW_USD:.2f}", "per 1,000 searches", "90–330× cheaper than flagship tool search", AMBER),
             (f"{a}{round(3600000 / NEW_MS, -3):,.0f}", "searches an hour on one GPU", "full quality, depth 25", SKY),
             ("1.2B", "parameters, open weights", "two 0.6B models · your tools stay yours", "#3fcf9f")]
    cw, gap, cx0, cy, ch = 288, 16, 40, 222, 128
    for i, (big, lab, sub, col) in enumerate(cards):
        x = cx0 + i * (cw + gap)
        body += f'<rect x="{x}" y="{cy}" width="{cw}" height="{ch}" rx="16" fill="url(#card)" stroke="#3b3835"/>'
        body += f'<rect x="{x+18}" y="{cy}" width="{cw-36}" height="3" rx="1.5" fill="{col}"/>'
        body += f'<text x="{x+22}" y="{cy+58}" font-size="44" font-weight="800" fill="{col}">{big}</text>'
        body += t(x + 22, cy + 84, lab, 15.5, INK, weight=600) + t(x + 22, cy + 108, sub, 13, MUTED)
    # ---- chart header
    body += t(48, 398, "Same task, three tests", 19, INK, weight=700) + t(268, 398, "nDCG@10, higher is better", 14, MUTED)
    for i, lab in enumerate(("ToolRet (public)", "OTS-MCP-1k (ours)", "same, in Korean")):
        body += t(qx[i], y0 - 8, lab, 12.5, MUTED)
    body += t(xc, y0 - 8, "$ per 1,000 searches", 12.5, MUTED) + t(xl, y0 - 8, "seconds per search", 12.5, MUTED)

    def group(y, title, col):
        return f'<rect x="48" y="{y+9}" width="6" height="16" rx="3" fill="{col}"/>' + t(64, y + 22, title, 13, INK2, weight=700).replace('<text ', '<text letter-spacing="1.2" ')

    def draw(rows, y, glow_first=False):
        out = ""
        for j, r in enumerate(rows):
            yy = y + j * rowh; ours = r["grp"] == "ots"
            c = {"ots": ACCENT, "dense": AMBER, "api": SKY}[r["grp"]]
            if ours:
                out += f'<rect x="36" y="{yy+2}" width="{W-72}" height="{rowh-4}" rx="12" fill="url(#hl)" stroke="{ACCENT}" stroke-opacity="0.55"/>'
            out += t(64, yy + 30, r["name"], 15.5 if ours else 14.5, INK, weight=800 if ours else 400)
            for i in range(3):
                w = r["q"][i] / 100 * qw
                f = ' filter="url(#glow)"' if ours else ""
                out += f'<path d="M{qx[i]},{yy+15} h{w-4:.1f} a4,4 0 0 1 4,4 v12 a4,4 0 0 1 -4,4 h-{w-4:.1f} z" fill="{c}"{f}/>'
                out += t(qx[i] + w + 7, yy + 30, f'{r["q"][i]:.1f}', 14.5, INK, weight=700 if ours else 600)
            cxp = CX(r["cost"]); lab = ("≈" if r.get("est") else "") + (f'${r["cost"]:.2f}' if r["cost"] < 10 else f'${r["cost"]:.0f}')
            out += f'<line x1="{xc}" y1="{yy+25}" x2="{cxp:.1f}" y2="{yy+25}" stroke="{c}" stroke-width="2" opacity="0.5"/><circle cx="{cxp:.1f}" cy="{yy+25}" r="{8 if ours else 6.5}" fill="{c}" stroke="{BG}" stroke-width="2"' + (' filter="url(#glow)"' if ours else "") + '/>'
            out += t(cxp + 13, yy + 30, lab, 14.5, INK, weight=700 if ours else 600) if cxp < xc + wc - 46 else t(cxp - 13, yy + 12, lab, 14, INK, "end", 600)
            w = max(3, r["lat"] / 12.0 * wl)
            out += f'<path d="M{xl},{yy+15} h{max(0,w-4):.1f} a4,4 0 0 1 4,4 v12 a4,4 0 0 1 -4,4 h-{max(0,w-4):.1f} z" fill="{c}"' + (' filter="url(#glow)"' if ours else "") + '/>'
            lt = ("≈" if r.get("est") else "") + (f'{r["lat"]*1000:.0f} ms' if r["lat"] < 0.05 else (f'{r["lat"]:.2f} s' if r["lat"] < 10 else f'{r["lat"]:.1f} s'))
            out += t(xl + w + 8, yy + 30, lt, 14.5, INK, weight=700 if ours else 600)
        return out
    for i in range(3):
        for gx in (0, 50, 100):
            X = qx[i] + gx / 100 * qw
            body += f'<line x1="{X}" y1="{y0+2}" x2="{X}" y2="{H-156}" stroke="{GRID}" opacity="0.7"/>'
    for gx, lab in ((0.1, "$0.1"), (1, "$1"), (10, "$10"), (100, "$100")):
        body += f'<line x1="{CX(gx)}" y1="{y0+2}" x2="{CX(gx)}" y2="{H-156}" stroke="{GRID}" opacity="0.7"/>' + t(CX(gx), H - 140, lab, 12, MUTED, "middle")
    body += f'<line x1="{xl}" y1="{y0+2}" x2="{xl}" y2="{H-156}" stroke="{GRID}" opacity="0.7"/>'
    body += group(y0, "OPEN WEIGHTS  ·  run it yourself", ACCENT) + draw(opens, y0 + 34)
    y1 = y0 + 34 + rowh * len(opens) + 6
    body += group(y1, "HOSTED TOOL SEARCH  ·  OpenAI, Anthropic, Google", SKY) + draw(hosted, y1 + 34)
    foot = ["ToolRet (public): 769 questions, each searched within its own source dataset. OTS-MCP-1k: 1,000 MCP tools, 200 requests, interchangeable tools credited (built by us). Korean: same requests, tools in English.",
            "Hosted tool search: every tool sent deferred, the model searches itself (Opus 5.5 at max effort, the others at default). Cost: lower of list price and cache-aware, Nov 2026; open models: GPU time on one B300.",
            "Not shown: hosted decision models (Jev, Gemini Decision), which rank candidates a search stage hands them. Open pipelines: time per search at the measured throughput (waves of 8).",
            "Tool-DE is a 4B + 4B pair, run on plain tool text. Tool-Rank-4B is Tool-DE's reranker, here behind our embedder (third-party model, Apache-2.0).",
            "Every table, setting and failed run: BENCHMARKS.md and results/LEDGER.md  ·  github.com/ai3inc/opentoolsearch"]
    for k, line in enumerate(foot):
        body += t(48, H - 118 + 19 * k, line, 11.5, MUTED)
    render("summary", W, H, body)


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
    fig_scorecards(); fig_summary(); fig_toolret()
