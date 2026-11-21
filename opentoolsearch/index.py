"""ToolIndex: an embedding index over tool definitions, and a drop-in `tool_search`
tool for any function-calling model (deferred tool loading on the client side).

    from opentoolsearch import ToolIndex
    index = ToolIndex()                      # loads the default opentoolsearch model
    index.add(tools)                         # OpenAI / Anthropic / Gemini / MCP style dicts
    index.search("refund the last order", k=5)

    # Deferred loading: send only `index.search_tool()` (+ a one-line catalogue) to the
    # model; when it calls tool_search, answer with `index.handle(call_args)` and add the
    # returned full declarations to the next request.
"""
import json
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_INSTRUCTION = "Given a user request, retrieve the tools that are needed to fulfil it"


def _unwrap(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Return the inner declaration of a tool given in any common wire format."""
    if "function" in tool and isinstance(tool["function"], dict):      # OpenAI chat-completions
        return tool["function"]
    return tool                                                         # Anthropic / Gemini / MCP / plain


def tool_name(tool: Dict[str, Any]) -> str:
    return str(_unwrap(tool).get("name", ""))


def tool_text(tool: Dict[str, Any]) -> str:
    """Text that gets embedded: the declaration as compact JSON, name and description first."""
    t = _unwrap(tool)
    head = {k: t[k] for k in ("name", "description") if k in t}
    rest = {k: v for k, v in t.items() if k not in head and k not in ("defer_loading", "short_description")}
    return json.dumps({**head, **rest}, ensure_ascii=False)


def short_line(tool: Dict[str, Any], max_chars: int = 160) -> str:
    t = _unwrap(tool)
    d = t.get("short_description") or (t.get("description") or "").split(". ")[0]
    d = " ".join(str(d).split())
    return f"{t.get('name')}: {d[:max_chars]}"


class ToolIndex:
    """Embedding index over tool definitions, with an optional cross-encoder second stage.

    model     any sentence-transformers embedder. Models that ship "query"/"document"
              prompts (e.g. Nemotron-3-Embed, opentoolsearch-embed) get them applied; others get
              the "Instruct: ...\nQuery: ..." format.
    reranker  optional path/name of a opentoolsearch reranker (Qwen3-Reranker format). When set,
              the top `rerank_top` hits (default 25) are re-scored and the two scores are fused
              with weight `fuse` (default 0.5). Depth and fusion were chosen on two benchmarks:
              depth 25 is 0.2 nDCG@10 below 50 on ToolRet per-source and twice as fast.
    """

    def __init__(self, model: str = DEFAULT_MODEL, reranker: Optional[str] = None,
                 device: Optional[str] = None, instruction: str = DEFAULT_INSTRUCTION,
                 max_doc_tokens: int = 1024, rerank_top: int = 25, fuse: float = 0.5, encoder=None):
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(model, device=device)
            encoder.max_seq_length = max_doc_tokens
        self.encoder = encoder
        prompts = getattr(encoder, "prompts", None) or {}
        self.q_prefix, self.d_prefix = prompts.get("query"), prompts.get("document", "")
        self.instruction = instruction
        self.rerank_top, self.fuse = rerank_top, fuse
        self.reranker = None
        if reranker:
            from .rerank import Reranker
            import torch
            self.reranker = Reranker(reranker, device=device,
                                     dtype=torch.bfloat16 if (device or "").startswith("cuda") or
                                     (device is None and torch.cuda.is_available()) else None)
            self.reranker.model.eval()
        self.tools: List[Dict[str, Any]] = []
        self._emb = None

    # ---- building
    def add(self, tools: Iterable[Dict[str, Any]], batch_size: int = 32) -> "ToolIndex":
        import numpy as np
        tools = list(tools)
        if not tools:
            return self
        e = self.encoder.encode([(self.d_prefix or "") + tool_text(t) for t in tools], batch_size=batch_size,
                                normalize_embeddings=True, convert_to_numpy=True)
        self._emb = e if self._emb is None else np.concatenate([self._emb, e])
        self.tools += tools
        return self

    def __len__(self):
        return len(self.tools)

    def _query_text(self, query: str, instruction: Optional[str]) -> str:
        if self.q_prefix is not None:                       # prefix-style models
            return self.q_prefix + (f"{instruction}\n{query}" if instruction else query)
        return f"Instruct: {instruction or self.instruction}\nQuery: {query}"

    # ---- searching
    def search(self, query: str, k: int = 5, instruction: Optional[str] = None,
               allowed: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        """Top-k tools for a request.

        instruction  optional one-line description of the capability needed
                     ("Given a `weather` task, retrieve tools that ..."); sharpens the search.
        allowed      restrict to a set of tool names (e.g. a per-user permission filter).
        """
        if not self.tools:
            return []
        import numpy as np
        qe = self.encoder.encode([self._query_text(query, instruction)], normalize_embeddings=True,
                                 convert_to_numpy=True)[0]
        scores = (self._emb @ qe).astype("float32")
        if allowed is not None:
            ok = set(allowed)
            for i, t in enumerate(self.tools):
                if tool_name(t) not in ok:
                    scores[i] = -1e9
        n = max(k, self.rerank_top) if self.reranker else k
        order = [int(i) for i in scores.argsort()[::-1][:n] if scores[i] > -1e8]
        final = {i: float(scores[i]) for i in order}
        if self.reranker and len(order) > 1:
            import torch
            with torch.no_grad():
                rr = self.reranker.score([(instruction or "", query, tool_text(self.tools[i])) for i in order]).tolist()
            z = lambda v: (np.asarray(v) - np.mean(v)) / (np.std(v) or 1.0)
            fused = z(rr) + self.fuse * z([final[i] for i in order])
            final = {i: float(s) for i, s in zip(order, fused)}
            order = sorted(order, key=lambda i: -final[i])
        return [{"name": tool_name(self.tools[i]), "score": final[i], "tool": self.tools[i]} for i in order[:k]]

    # ---- deferred loading helpers
    def catalogue(self, max_chars: int = 160) -> str:
        """One line per tool (name: first sentence) - the cheap listing a model can read."""
        return "\n".join(short_line(t, max_chars) for t in self.tools)

    def search_tool(self, name: str = "tool_search", k: int = 5) -> Dict[str, Any]:
        """Function declaration (JSON Schema) to hand to the model in place of the full tool list."""
        return {
            "name": name,
            "description": (f"Search the tool library ({len(self.tools)} tools) and load the full "
                            f"definitions of the best matches. Call this before using any tool you "
                            f"have not loaded yet. Describe the capability you need in plain words."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What the tool should be able to do."},
                    "k": {"type": "integer", "description": f"How many tools to load (default {k}).",
                          "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
            },
        }

    def handle(self, arguments: Any, k: int = 5, allowed: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        """Answer a tool_search call: returns the full declarations to load."""
        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")
        hits = self.search(arguments.get("query", ""), k=int(arguments.get("k") or k), allowed=allowed)
        return [h["tool"] for h in hits]
