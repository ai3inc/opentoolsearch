"""Deferred tool loading with any OpenAI-compatible chat model.

The model first sees only `tool_search`. When it calls it, we load the matching full
declarations into the next request - the same flow as server-side tool retrieval, run
on the client so it works with any provider.

  OPENAI_BASE_URL=... OPENAI_API_KEY=... python examples/deferred_openai.py "what's the weather in Seoul?"
"""
import json, os, sys, urllib.request
from opentoolsearch import ToolIndex

TOOLS = json.load(open(os.path.join(os.path.dirname(__file__), "tools.json")))
MODEL = os.environ.get("MODEL", "gpt-4o-mini")
index = ToolIndex().add(TOOLS)               # OpenToolSearch Embed + Rerank, downloaded from Hugging Face


def chat(messages, tools):
    body = {"model": MODEL, "messages": messages,
            "tools": [{"type": "function", "function": t} for t in tools]}
    req = urllib.request.Request(os.environ["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                                          "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]


def run(user):
    loaded = {}                                   # name -> declaration, grows as the model searches
    messages = [{"role": "system", "content": f"You can use {len(index)} tools. Only tool_search is loaded; "
                 "search for what you need, then call the loaded tool."},
                {"role": "user", "content": user}]
    for _ in range(6):
        msg = chat(messages, [index.search_tool()] + list(loaded.values()))
        messages.append(msg)
        if not msg.get("tool_calls"):
            return msg.get("content"), loaded
        for call in msg["tool_calls"]:
            name, args = call["function"]["name"], call["function"]["arguments"]
            if name == "tool_search":
                found = index.handle(args)
                for t in found:
                    loaded[t["name"]] = t
                result = {"loaded": [t["name"] for t in found]}
            else:
                result = {"ok": True, "note": f"(demo) {name} was called with {args}"}
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
    return None, loaded


if __name__ == "__main__":
    answer, loaded = run(sys.argv[1] if len(sys.argv) > 1 else "What's the weather in Seoul tomorrow?")
    print("loaded tools:", list(loaded)); print("answer:", answer)
