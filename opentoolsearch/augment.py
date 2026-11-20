"""Format augmentation for tool definitions.

Tool libraries describe the same tool in many layouts (OpenAI, MCP, RapidAPI dumps,
Python signatures, ...). A retriever trained on one or two layouts learns to prefer
those layouts. Re-rendering every training document in a random layout removes that
shortcut: relevance has to come from what the tool does, not how it is written down.
"""
import json, random

SYN = {
    "name": ["tool_name", "api_name", "function_name", "title", "id"],
    "description": ["desc", "summary", "doc", "func_description", "about", "details"],
    "parameters": ["input_schema", "arguments", "args", "doc_arguments", "inputs", "params", "inputSchema"],
    "required_parameters": ["required", "required_args", "required_params"],
    "optional_parameters": ["optional", "optional_args", "optional_params"],
    "category_name": ["category", "domain", "group"],
    "template_response": ["response", "returns", "output_schema"],
    "method": ["http_method", "verb"],
}
AUX = ["template_response", "method", "category_name", "api_description", "tool_description"]


def _plain(d, depth=0):
    """key: value lines (YAML-like)."""
    out = []
    for k, v in d.items():
        if isinstance(v, dict) and v and depth < 2:
            out.append("  " * depth + f"{k}:")
            out.append(_plain(v, depth + 1))
        else:
            out.append("  " * depth + f"{k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}")
    return "\n".join(out)


def _signature(d):
    """def name(a: type, b: type): \"\"\"description\"\"\" """
    name = d.get("name"); desc = d.get("description", "")
    params = d.get("parameters")
    if not isinstance(name, str) or not isinstance(params, dict):
        return None
    props = params.get("properties", params) if isinstance(params.get("properties", params), dict) else {}
    args = []
    for k, v in props.items():
        if not isinstance(v, dict):
            return None
        t = v.get("type", "any")
        args.append(f"{k}: {t if isinstance(t, str) else 'any'}")
    doc = [str(desc)] + [f"    {k}: {v.get('description', '')}" for k, v in props.items() if isinstance(v, dict) and v.get("description")]
    return f"def {name}({', '.join(args)}):\n    \"\"\"" + "\n".join(doc) + "\"\"\""


def augment_doc(text, rng=random, p=0.6):
    """Return the same tool in a random layout (or unchanged with prob 1-p / if not JSON)."""
    if rng.random() >= p:
        return text
    try:
        d = json.loads(text)
    except Exception:
        return text
    if not isinstance(d, dict):
        return text
    d = dict(d)
    for k in AUX:                                   # drop auxiliary fields sometimes
        if k in d and rng.random() < 0.3:
            d.pop(k)
    r = rng.random()
    if r < 0.12:
        s = _signature(d)
        if s:
            return s
    if rng.random() < 0.6:                          # rename keys
        d = {(rng.choice(SYN[k]) if k in SYN and rng.random() < 0.5 else k): v for k, v in d.items()}
    if rng.random() < 0.6:                          # reorder keys
        ks = list(d); rng.shuffle(ks); d = {k: d[k] for k in ks}
    if r > 0.88:
        return _plain(d)
    if rng.random() < 0.12:
        d = {"type": "function", "function": d}
    style = rng.random()
    if style < 0.6:
        return json.dumps(d, ensure_ascii=False)
    if style < 0.8:
        return json.dumps(d, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(d, ensure_ascii=False, indent=2)
