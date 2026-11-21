"""Markdown for results/LEDGER.md: every hosted-API and dense-retriever run of the comparison benchmarks, with its cost.
   python scripts/make_ledger_hosted.py
"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda p: json.load(open(os.path.join(ROOT, "results", p)))
PUB = {r["system"]: r for r in R("public_head_to_head.json")["rows"]}
OWN = {r["system"]: r for r in R("hosted_tool_search.json")}
KO = {r["system"]: r for r in R("hosted_tool_search_ko.json")}
rows, total = [], 0.0


def line(bench, name, n, r, note=""):
    global total
    lst = r.get("usd_per_1k_list", r.get("usd_per_1k")); cac = r.get("usd_per_1k_cached")
    billed = (cac if cac is not None else lst) * n / 1000 if lst is not None else None
    if billed is not None:
        total += billed
    f = lambda x: "—" if x is None else f"{x:.2f}"
    rows.append(f"| {bench} | {name} | {n} | {r['ndcg10']:.2f} | {f(r.get('recall10'))} | {r.get('no_search', '—')} | {f(lst)} | {f(cac)} | {f(billed)} | {r.get('latency_p50', '—')} | {note} |")


PUB_NOTE = {"OpenAI tool search (gpt-6.2-sol)": "cache never hit: every request wrote the cache (billed above list)",
            "Gemini API tool search (gemini-3.6-flash)": "6 errors counted as misses", "Gemini API tool search (gemini-3.8-flash)": "1 error counted as a miss",
            "Claude tool search (Opus 5.5, effort max)": "search not forced (documented as refused); effort max"}
for k in ("Gemini API tool search (gemini-4-hydrogen)", "Gemini API tool search (gemini-3.8-flash)", "Gemini API tool search (gemini-3.6-flash)",
          "OpenAI tool search (gpt-6.2-sol)", "OpenAI tool search (gpt-6-luna)", "Claude tool search (Opus 5.5, effort max)", "Claude tool search (Haiku 5.5)"):
    line("ToolRet per-source", k, PUB[k]["answered"], PUB[k], PUB_NOTE.get(k, ""))
OWN_NOTE = {"oai-gpt-6.2-sol-run2": "second run, cache hit on every request (the first run below: no hits)",
            "oai-gpt-6.2-sol": "first run: cache never hit, every request wrote the cache", "ant-haiku-5-5": "search not forced: 31 of 200 requests skipped it",
            "ant-haiku-5-5-forced": "search forced", "ant-fable-5-2": "search not forced; run once, dropped for cost; forcing it is rejected (HTTP 400)",
            "ant-opus-5-5-max": "search not forced (documented as refused); effort max", "oai-gpt-5.4": "flat deferred list + forced search; superseded by gpt-6 models"}
for k in ("gemini-4-hydrogen", "gemini-3.8-flash", "gemini-3.6-flash", "oai-gpt-6.2-sol-run2", "oai-gpt-6.2-sol", "oai-gpt-6-luna", "oai-gpt-5.4",
          "ant-opus-5-5-max", "ant-haiku-5-5-forced", "ant-haiku-5-5", "ant-fable-5-2"):
    line("OTS-MCP-1k (English)", k, OWN[k]["answered"], OWN[k], OWN_NOTE.get(k, ""))
for k in KO:
    line("OTS-MCP-1k (Korean)", k, KO[k]["answered"], KO[k], "cache never hit (billed above list)" if k == "oai-gpt-6.2-sol" else "")
print("| Benchmark | System | Answered | nDCG@10 | Recall@10 | Skipped the search | $/1,000 list | $/1,000 with cache | Run cost $ | Median s | Note |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
print("\n".join(rows))
print(f"\nSum of the run costs above (cache-aware, as billed): ${total:,.0f}")

# ---- dense retrievers: every query format that was run
print("\n| Model | Query format | OTS-MCP-1k English | Korean | ToolRet per-source | |")
print("|---|---|---|---|---|---|")
for r in sorted(R("dense_tool_retrievers.json"), key=lambda r: r["system"]):
    for fmt, v in r["all_formats"].items():
        mark = "**used**" if fmt == r["format"] else ""
        print(f"| {r['system']} | {fmt} | {v['own_en']['ndcg10']:.2f} | {v['own_ko']['ndcg10']:.2f} | {v['public']['ndcg10']:.2f} | {mark} |")
