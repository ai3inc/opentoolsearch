"""Print the comparison tables of the README and the model card from the result files, so that no number is typed by hand.

  python scripts/make_tables.py            # all tables
  python scripts/make_tables.py one hosted dense korean
Reads results/public_head_to_head.json, head_to_head.json, hosted_tool_search.json, hosted_tool_search_ko.json,
korean_open.json, korean_decisions.json, dense_tool_retrievers.json and default_config.json.
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda p: json.load(open(os.path.join(ROOT, "results", p)))
PUB = {r["system"]: r for r in R("public_head_to_head.json")["rows"]}
OWN = {r["system"]: r for r in R("head_to_head.json")["rows"]}
HOST = {r["system"]: r for r in R("hosted_tool_search.json")}
HKO = {r["system"]: r for r in R("hosted_tool_search_ko.json")}
KO_OPEN = {(r["lang"], r["system"]): r for r in R("korean_open.json")}
KO_DEC = {(r["system"], r["candidates"]): r for r in R("korean_decisions.json")}
DENSE = {r["system"]: r for r in R("dense_tool_retrievers.json")}
DEF = {r["depth"]: r for r in R("default_config.json")["rows"]}
FS = {r["first_stage"]: r for r in R("first_stage_plus_reranker.json")}
EMBED_MS = 1.7          # Nemotron-3-Embed-1B alone: 2,088,775 searches an hour
KRW, GPU_USD_H = 1400.0, 7.09
f1 = lambda x: "—" if x is None else f"{x:.1f}"
usd = lambda x: "—" if x is None else (f"${x:.2f}" if x < 10 else f"${x:,.0f}")
sec = lambda x: "—" if x is None else (f"{x:.2f} s" if x < 10 else f"{x:.1f} s")


def dense_cost(r):
    return r["latency_ms_p50"] / 1000 * 1000 / 3600 * GPU_USD_H


def cost_cell(r):
    """lower of list and cache-aware, list price in brackets when the cache made it cheaper"""
    lst, cac = r.get("usd_per_1k", r.get("usd_per_1k_list")), r.get("usd_per_1k_cached")
    if lst is None:
        return "—"
    if cac is not None and cac < lst * 0.97:
        return f"{usd(cac)} (list {usd(lst)})"
    return usd(lst)


# (label, kind, public key, own-EN lookup, Korean lookup)
HOSTED = [("Gemini API tool search · gemini-4-hydrogen", "Gemini API tool search (gemini-4-hydrogen)", "gemini-4-hydrogen", "gemini-4-hydrogen", "Google", "top"),
          ("Gemini API tool search · gemini-3.8-flash", "Gemini API tool search (gemini-3.8-flash)", "gemini-3.8-flash", "gemini-3.8-flash", "Google", "cost-efficient (latest)"),
          ("Gemini API tool search · gemini-3.6-flash", "Gemini API tool search (gemini-3.6-flash)", "gemini-3.6-flash", "gemini-3.6-flash", "Google", "cost-efficient (previous)"),
          ("OpenAI tool search · gpt-6.2-sol", "OpenAI tool search (gpt-6.2-sol)", "oai-gpt-6.2-sol-run2", "oai-gpt-6.2-sol", "OpenAI", "latest"),
          ("OpenAI tool search · gpt-6-luna", "OpenAI tool search (gpt-6-luna)", "oai-gpt-6-luna", "oai-gpt-6-luna", "OpenAI", "cost-efficient"),
          ("Claude tool search · Opus 5.5 (effort max)", "Claude tool search (Opus 5.5, effort max)", "ant-opus-5-5-max", "ant-opus-5-5-max", "Anthropic", "top"),
          ("Claude tool search · Haiku 5.5", "Claude tool search (Haiku 5.5)", "ant-haiku-5-5-forced", "ant-haiku-5-5", "Anthropic", "cost-efficient")]
SETTING = {"Gemini API tool search (gemini-4-hydrogen)": "thinking level not set; search not forced",
           "Gemini API tool search (gemini-3.8-flash)": "thinking level not set; search not forced",
           "Gemini API tool search (gemini-3.6-flash)": "thinking level not set; search not forced",
           "OpenAI tool search (gpt-6.2-sol)": "reasoning effort not set (default); search forced",
           "OpenAI tool search (gpt-6-luna)": "reasoning effort not set (default); search forced",
           "Claude tool search (Opus 5.5, effort max)": "effort max, adaptive thinking; search not forced (documented as refused)",
           "Claude tool search (Haiku 5.5)": "effort not set (default medium); search forced"}


def table_one():
    out = ["| System | Kind | ToolRet per-source nDCG@10 | Recall@10 | OTS-MCP-1k nDCG@10 | …in Korean | Cost / 1,000 searches | Median latency |",
           "|---|---|---|---|---|---|---|---|"]
    rows = []
    d25 = PUB["OpenToolSearch (default: depth 25, fusion 0.5)"]
    ko = lambda s: KO_OPEN[("ko", s)]["ndcg10"]
    rows.append((d25["ndcg10"], f"| **OpenToolSearch — default (depth 25)** | open weights | **{d25['ndcg10']:.1f}** | {f1(d25['recall10'])} | {f1(DEF[25]['own']['w0.5'])} | {f1(ko('OpenToolSearch - default (depth 25)'))} | {usd(DEF[25]['usd_per_1k'])} | {DEF[25]['ms_per_search']/1000:.2f} s |"))
    p = PUB["OpenToolSearch"]
    rows.append((p["ndcg10"], f"| OpenToolSearch — depth 50 | open weights | {f1(p['ndcg10'])} | {f1(p['recall10'])} | {f1(OWN['rerank:published']['ndcg10_c'])} | {f1(ko('OpenToolSearch - depth 50, fusion 0.25'))} | {usd(0.475)} | 0.22 s |"))
    p = PUB["Qwen3-Reranker-0.6B as released"]
    rows.append((p["ndcg10"], f"| Qwen3-Reranker-0.6B as released | open baseline | {f1(p['ndcg10'])} | {f1(p['recall10'])} | {f1(OWN['rerank:base']['ndcg10_c'])} | {f1(ko('Qwen3-Reranker-0.6B as released - depth 50, fusion 0.25'))} | {usd(0.475)} | 0.22 s |"))
    p = PUB["Nemotron-3-Embed-1B alone"]
    rows.append((p["ndcg10"], f"| Nemotron-3-Embed-1B alone | open baseline | {f1(p['ndcg10'])} | {f1(p['recall10'])} | {f1(OWN['Nemotron-3-Embed-1B alone (first stage)']['ndcg10_c'])} | {f1(ko('Nemotron-3-Embed-1B alone'))} | $0.00 | 0.002 s |"))
    for nm, lab, own_k in (("Gemini Decision (single choice)", "Gemini Decision — single choice", "Gemini Decision (single choice)"),
                           ("Jev (single choice)", "Jev — single choice", "Jev (single choice)"),
                           ("Gemini Decision (yes/no per tool)", "Gemini Decision — yes/no per tool", "Gemini Decision (yes/no per tool)"),
                           ("Jev (yes/no per tool)", "Jev — yes/no per tool", "Jev (yes/no per tool)")):
        p = PUB[nm]; kor = f1(KO_DEC[(nm, 50)]["ndcg10"]) if (nm, 50) in KO_DEC else "—"
        rows.append((p["ndcg10"], f"| {lab} | hosted decision model (ranks our candidates) | {f1(p['ndcg10'])} | {f1(p['recall10'])} | {f1(OWN[own_k]['ndcg10_c'])} | {kor} | {usd(p['usd_per_1k'])} | {sec(p['latency_p50'])} |"))
    for lab, pn, ok, kk, vendor, tier in HOSTED:
        p = PUB.get(pn)
        pub = f1(p["ndcg10"]) if p and p.get("ndcg10") is not None else "—"
        rec = f1(p["recall10"]) if p and p.get("recall10") is not None else "—"
        rows.append(((p or {}).get("ndcg10") or 0, f"| {lab} | hosted tool search | {pub} | {rec} | {f1(HOST[ok]['ndcg10'])} | {f1(HKO[kk]['ndcg10'])} | {cost_cell(p) if p else '—'} | {sec(p['latency_p50']) if p else '—'} |"))
    for n, r in DENSE.items():
        label = f"{n} ({r['params_m']/1000:.1f}B)" if r["params_m"] >= 1000 else f"{n} ({r['params_m']}M)"
        kind = "general embedder, reference" if n.startswith("Qwen3") else "dedicated open retriever"
        rows.append((r["public"]["ndcg10"], f"| {label} | {kind} | {f1(r['public']['ndcg10'])} | {f1(r['public']['recall10'])} | {f1(r['own_en']['ndcg10'])} | {f1(r['own_ko']['ndcg10'])} | {usd(dense_cost(r))} | {r['latency_ms_p50']/1000:.3f} s |"))
    out += [r[1] for r in sorted(rows, key=lambda x: -x[0])]
    return "\n".join(out)


SUMMARY_TOOLSEARCH = ["gemini-4-hydrogen", "gemini-3.8-flash", "oai-gpt-6.2-sol-run2", "oai-gpt-6-luna", "ant-opus-5-5-max", "ant-haiku-5-5-forced"]


def table_summary():
    """The one table of the README and the model card: ours, the two best open competitors, the six hosted tool-search models (a top and a cost-efficient one per vendor)."""
    out = ["| System | Kind | ToolRet (public) | OTS-MCP-1k (ours) | …in Korean | Cost / 1,000 searches | Median latency |", "|---|---|---|---|---|---|---|"]
    rows = []
    d25 = PUB["OpenToolSearch (default: depth 25, fusion 0.5)"]
    rows.append((d25["ndcg10"], f"| **OpenToolSearch** (1.14B + 0.6B) | open weights, this repo | **{d25['ndcg10']:.1f}** | **{DEF[25]['own']['w0.5']:.1f}** | **{KO_OPEN[('ko', 'OpenToolSearch - default (depth 25)')]['ndcg10']:.1f}** | **{usd(DEF[25]['usd_per_1k'])}** | **{DEF[25]['ms_per_search']/1000:.2f} s** |"))
    cb = FS["callbus-emb-8b"]["with_reranker"]; est_ms = DEF[25]["ms_per_search"] - EMBED_MS + DENSE["callbus-emb-8b"]["latency_ms_p50"]
    rows.append((cb["public"], f"| **OpenToolSearch reranker behind callbus-emb-8b** (7.6B + 0.6B) | open weights, this repo's reranker | **{cb['public']:.1f}** | **{cb['own_en']:.1f}** | **{cb['own_ko']:.1f}** | ≈{usd(est_ms/1000*1000/3600*GPU_USD_H)} | ≈{est_ms/1000:.2f} s |"))
    for n, lab in (("Tool-DE (Tool-Embed-4B + Tool-Rank-4B)", "Tool-DE (4B + 4B)"), ("callbus-emb-8b", "callbus-emb-8b (8B, embedding only)")):
        r = DENSE[n]
        rows.append((r["public"]["ndcg10"], f"| {lab} | open weights | {f1(r['public']['ndcg10'])} | {f1(r['own_en']['ndcg10'])} | {f1(r['own_ko']['ndcg10'])} | {usd(dense_cost(r))} | {r['latency_ms_p50']/1000:.2f} s |"))
    for lab, pn, ok, kk, vendor, tier in HOSTED:
        if ok not in SUMMARY_TOOLSEARCH:
            continue
        p = PUB[pn]
        rows.append((p["ndcg10"], f"| {lab.replace('Gemini API tool search', 'Gemini tool search').replace(' (effort max)', ' max effort')} | hosted tool search | {f1(p['ndcg10'])} | {f1(HOST[ok]['ndcg10'])} | {f1(HKO[kk]['ndcg10'])} | {cost_cell(p)} | {sec(p['latency_p50'])} |"))
    return "\n".join(out + [r[1] for r in sorted(rows, key=lambda x: -x[0])])


def table_firststage():
    out = ["| First stage | Alone (public / own / Korean) | + OpenToolSearch reranker | Gain | Every labelled tool in the top 25 |", "|---|---|---|---|---|"]
    size = {"callbus-emb-8b": "7.6B", "Qwen3-Embedding-8B": "7.6B", "Tool-Embed-4B": "4.0B", "Nemotron-3-Embed-1B": "1.14B", "Nemotron-3-Embed-8B": "8B"}
    for n, r in sorted(FS.items(), key=lambda kv: -kv[1]["with_reranker"]["public"]):
        a, w, c = r["alone"], r["with_reranker"], r["coverage"]["top25"]
        out.append(f"| {n} ({size[n]}){' - the library default' if n == 'Nemotron-3-Embed-1B' else ''} | {a['public']:.1f} / {a['own_en']:.1f} / {a['own_ko']:.1f} | **{w['public']:.1f} / {w['own_en']:.1f} / {w['own_ko']:.1f}** | "
                   f"{w['public']-a['public']:+.1f} / {w['own_en']-a['own_en']:+.1f} / {w['own_ko']-a['own_ko']:+.1f} | {c['public']['all_found']:.0f}% / {c['own_en']['all_found']:.0f}% / {c['own_ko']['all_found']:.0f}% |")
    return "\n".join(out)


def table_hosted():
    out = ["| Vendor | Model | Role | Setting | ToolRet per-source | OTS-MCP-1k | …in Korean | $ / 1,000 on OTS-MCP-1k: list → with cache hits | Median latency |",
           "|---|---|---|---|---|---|---|---|---|"]
    for lab, pn, ok, kk, vendor, tier in HOSTED:
        p, o, k = PUB.get(pn), HOST[ok], HKO[kk]
        pub = f1(p["ndcg10"]) if p and p.get("ndcg10") is not None else "—"
        out.append(f"| {vendor} | {lab.split(' · ')[1]} | {tier} | {SETTING[pn]} | {pub} | {f1(o['ndcg10'])} | {f1(k['ndcg10'])} | {usd(o['usd_per_1k_list'])} → {usd(o['usd_per_1k_cached'])} | {sec(o['latency_p50'])} |")
    return "\n".join(out)


def table_dense():
    out = ["| Model | Weights | Query format | ToolRet per-source | OTS-MCP-1k | …in Korean | ms per search | $ / 1,000 |", "|---|---|---|---|---|---|---|---|"]
    for n, r in sorted(DENSE.items(), key=lambda kv: -kv[1]["public"]["ndcg10"]):
        size = f"{r['params_m']/1000:.1f}B" if r["params_m"] >= 1000 else f"{r['params_m']}M"
        rev = f"@{r['revision']}" if r.get("revision") else ""
        out.append(f"| {n} ({size}) | `{r['weights']}{rev}` | {r['format']} | {f1(r['public']['ndcg10'])} | {f1(r['own_en']['ndcg10'])} | {f1(r['own_ko']['ndcg10'])} | {r['latency_ms_p50']:.1f} | {usd(dense_cost(r))} |")
    return "\n".join(out)


def table_korean():
    out = ["| System | OTS-MCP-1k in English | … in Korean | Change | What it returns (Korean) |", "|---|---|---|---|---|"]
    rows = []
    for key, lab, ret in (("OpenToolSearch - default (depth 25)", "OpenToolSearch — default (depth 25) †", "ranks 25 candidates"),
                          ("OpenToolSearch - depth 50, fusion 0.25", "OpenToolSearch — depth 50 †", "ranks 50 candidates"),
                          ("Qwen3-Reranker-0.6B as released - depth 50, fusion 0.25", "Qwen3-Reranker-0.6B as released (depth 50) †", "ranks 50 candidates"),
                          ("Nemotron-3-Embed-1B alone", "Nemotron-3-Embed-1B alone †", "top 50")):
        e, k = KO_OPEN[("en", key)]["ndcg10"], KO_OPEN[("ko", key)]["ndcg10"]
        rows.append((k, f"| {lab} | {e:.1f} | {k:.1f} | {k-e:+.1f} | {ret} |"))
    for nm, lab in (("Jev (single choice)", "Jev — single choice (50 candidates)"), ("Gemini Decision (single choice)", "Gemini Decision — single choice (50 candidates)")):
        e = OWN[nm]["ndcg10_c"]; k = KO_DEC[(nm, 50)]["ndcg10"]
        rows.append((k, f"| {lab} | {e:.1f} | {k:.1f} | {k-e:+.1f} | ranks 50 candidates |"))
    for lab, pn, ok, kk, vendor, tier in HOSTED:
        e, k = HOST[ok]["ndcg10"], HKO[kk]["ndcg10"]
        rows.append((k, f"| {lab} | {e:.1f} | {k:.1f} | {k-e:+.1f} | {HKO[kk]['avg_returned']:.1f} tools; searched on {HKO[kk]['answered']-HKO[kk]['no_search']} of {HKO[kk]['answered']} |"))
    for n, r in DENSE.items():
        e, k = r["own_en"]["ndcg10"], r["own_ko"]["ndcg10"]
        rows.append((k, f"| {n} | {e:.1f} | {k:.1f} | {k-e:+.1f} | embedding search, top 20 |"))
    return "\n".join(out + [r[1] for r in sorted(rows, key=lambda x: -x[0])])


if __name__ == "__main__":
    which = sys.argv[1:] or ["summary", "firststage", "one", "hosted", "dense", "korean"]
    for w in which:
        print(f"<!-- {w} -->"); print({"summary": table_summary, "firststage": table_firststage, "one": table_one, "hosted": table_hosted, "dense": table_dense, "korean": table_korean}[w]()); print()
