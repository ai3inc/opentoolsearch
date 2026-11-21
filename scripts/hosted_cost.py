"""Token-usage -> dollars for the hosted tool-search systems, whatever the vendor's field names.

split_usage(u) -> (uncached input, cache read, cache write, output) token counts
costs(u, price) -> (list, cached) in USD, price = (input, output, cache_read, cache_write) per 1M tokens

list   prices every input token at the full input rate (an upper bound: no cache);
cached bills the cache hits the API reported at the cache-read rate and the cache writes at the cache-write rate,
       which is what a loop over a stable tool library pays when the cache stays warm.
"""


def split_usage(u):
    if "input_tokens_details" in u:                       # OpenAI Responses: input_tokens includes cached and written
        d = u["input_tokens_details"] or {}
        cr, cw = d.get("cached_tokens", 0) or 0, d.get("cache_write_tokens", 0) or 0
        return u["input_tokens"] - cr - cw, cr, cw, u.get("output_tokens", 0)
    if "cache_read_input_tokens" in u:                    # Anthropic: input_tokens excludes cache reads and writes
        return (u.get("input_tokens", 0), u.get("cache_read_input_tokens", 0) or 0,
                u.get("cache_creation_input_tokens", 0) or 0, u.get("output_tokens", 0))
    tin = (u.get("total_input_tokens") or 0) + (u.get("total_tool_use_tokens") or 0)   # Gemini: includes cached
    cr = u.get("total_cached_tokens", 0) or 0
    return tin - cr, cr, 0, (u.get("total_output_tokens") or 0) + (u.get("total_thought_tokens") or 0)


def costs(u, price):
    pi, po, pcr, pcw = price
    un, cr, cw, out = split_usage(u or {})
    return ((un + cr + cw) * pi + out * po) / 1e6, (un * pi + cr * pcr + cw * pcw + out * po) / 1e6


def parse_prices(text):
    """"name:in:out[:cache_read[:cache_write]],..." -> {name: (in, out, cache_read, cache_write)}"""
    out = {}
    for part in (text or "").split(","):
        if part:
            f = part.split(":"); i, o = float(f[1]), float(f[2])
            out[f[0]] = (i, o, float(f[3]) if len(f) > 3 else i, float(f[4]) if len(f) > 4 else i)
    return out
