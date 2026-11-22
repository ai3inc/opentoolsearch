"""ToolIndex: an embedding index over tool definitions, and a drop-in `tool_search`
tool for any function-calling model (deferred tool loading on the client side).

    from opentoolsearch import ToolIndex
    index = ToolIndex()                      # OpenToolSearch Embed 0.6B + Rerank 0.6B, both from Hugging Face
    index.add(tools)                         # OpenAI / Anthropic / Gemini / MCP style dicts
    index.search("refund the last order", k=5)

    # Keep it current without re-embedding the rest (the tool's name is its key):
    index.update([new_or_changed_tool])      # add or replace one tool - only that one is embedded
    index.remove("old_tool")                 # drop by name
    index.sync(all_tools_now)                # add the new, re-embed the changed, drop the missing
    index.save("tool_index/"); ToolIndex.load("tool_index/")    # vectors are kept, nothing is re-embedded

    # Deferred loading: send only `index.search_tool()` (+ a one-line catalogue) to the
    # model; when it calls tool_search, answer with `index.handle(call_args)` and add the
    # returned full declarations to the next request.
"""
import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_MODEL = "ai3inc/opentoolsearch-embed-0.6b"
DEFAULT_RERANKER = "ai3inc/opentoolsearch-rerank-0.6b"
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


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def short_line(tool: Dict[str, Any], max_chars: int = 160) -> str:
    t = _unwrap(tool)
    d = t.get("short_description") or (t.get("description") or "").split(". ")[0]
    d = " ".join(str(d).split())
    return f"{t.get('name')}: {d[:max_chars]}"


class ToolIndex:
    """Embedding index over tool definitions with a cross-encoder second stage: OpenToolSearch.

    model     any sentence-transformers embedder; the default is OpenToolSearch Embed 0.6B. Models that
              ship "query"/"document" prompts in their config get them applied; others get the
              "Instruct: ...\nQuery: ..." format. `query_prefix` / `doc_prefix` override both.
    reranker  name or path of an OpenToolSearch reranker (Qwen3-Reranker format; default OpenToolSearch
              Rerank 0.6B), or None for embedding search only. The top `rerank_top` hits (default 25)
              are re-scored and the two scores fused with weight `fuse` (default 0.5). Depth 25 is about
              as good as 50 on ToolRet per-source and twice as fast.
    """

    def __init__(self, model: str = DEFAULT_MODEL, reranker: Optional[str] = DEFAULT_RERANKER,
                 device: Optional[str] = None, instruction: str = DEFAULT_INSTRUCTION,
                 max_doc_tokens: int = 512, rerank_top: int = 25, fuse: float = 0.5, encoder=None,
                 query_prefix: Optional[str] = None, doc_prefix: Optional[str] = None,
                 embed_instruction: bool = False):
        self.model_name = model if encoder is None else "custom"
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(model, device=device)
            encoder.max_seq_length = max_doc_tokens
        self.encoder = encoder
        prompts = getattr(encoder, "prompts", None) or {}
        self.q_prefix = query_prefix if query_prefix is not None else prompts.get("query")
        self.d_prefix = doc_prefix if doc_prefix is not None else prompts.get("document", "")
        self.instruction = instruction
        self.embed_instruction = embed_instruction      # also give the optional `instruction` to the embedder
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
        self._hashes: List[str] = []        # digest of each tool's embedded text, to see what changed

    # ---- building and keeping current
    def _embed_docs(self, texts: List[str], batch_size: int):
        return self.encoder.encode([(self.d_prefix or "") + x for x in texts], batch_size=batch_size,
                                   normalize_embeddings=True, convert_to_numpy=True)

    def add(self, tools: Iterable[Dict[str, Any]], batch_size: int = 32) -> "ToolIndex":
        """Embed the given tools and append them. Only these are embedded; what is in the index stays as it is.
        Use `update` instead if a tool of the same name may already be there."""
        import numpy as np
        tools = list(tools)
        if not tools:
            return self
        texts = [tool_text(t) for t in tools]
        e = self._embed_docs(texts, batch_size)
        self._emb = e if self._emb is None else np.concatenate([self._emb, e])
        self.tools += tools
        self._hashes += [_digest(x) for x in texts]
        return self

    def names(self) -> List[str]:
        return [tool_name(t) for t in self.tools]

    def _drop(self, drop: Iterable[int]) -> int:
        drop = set(drop)
        keep = [i for i in range(len(self.tools)) if i not in drop]
        if len(keep) == len(self.tools):
            return 0
        n = len(self.tools) - len(keep)
        self.tools = [self.tools[i] for i in keep]
        self._hashes = [self._hashes[i] for i in keep]
        self._emb = self._emb[keep] if keep else None
        return n

    def remove(self, names) -> int:
        """Drop every tool with one of these names (a name or a list of names). Returns how many were removed."""
        names = {names} if isinstance(names, str) else set(names)
        return self._drop(i for i, t in enumerate(self.tools) if tool_name(t) in names)

    def update(self, tools: Iterable[Dict[str, Any]], batch_size: int = 32) -> "ToolIndex":
        """Add or replace by name: a tool whose name is already in the index takes the old one's place (only the
        given tools are embedded), a new name is appended. Names must be unique within `tools`."""
        tools = list(tools)
        if not tools:
            return self
        names = [tool_name(t) for t in tools]
        if len(set(names)) != len(names):
            raise ValueError("duplicate tool names in `tools`; the name is the key")
        texts = [tool_text(t) for t in tools]
        e = self._embed_docs(texts, batch_size)
        first: Dict[str, int] = {}
        for i, t in enumerate(self.tools):
            first.setdefault(tool_name(t), i)
        incoming = set(names)
        self._drop(i for i, t in enumerate(self.tools) if tool_name(t) in incoming and first[tool_name(t)] != i)   # stray duplicates
        first = {}
        for i, t in enumerate(self.tools):
            first.setdefault(tool_name(t), i)
        new = []
        for j, (t, n) in enumerate(zip(tools, names)):
            if n in first:
                i = first[n]
                self.tools[i], self._emb[i], self._hashes[i] = t, e[j], _digest(texts[j])
            else:
                new.append(j)
        if new:
            import numpy as np
            self._emb = e[new] if self._emb is None else np.concatenate([self._emb, e[new]])
            self.tools += [tools[j] for j in new]
            self._hashes += [_digest(texts[j]) for j in new]
        return self

    def sync(self, tools: Iterable[Dict[str, Any]], batch_size: int = 32) -> Dict[str, int]:
        """Make the index hold exactly these tools, embedding only what is new or whose text changed - e.g. every
        time a skills folder is scanned. Names are keys and must be unique. Returns the counts."""
        tools = list(tools)
        names = [tool_name(t) for t in tools]
        if len(set(names)) != len(names):
            raise ValueError("duplicate tool names in `tools`; the name is the key")
        digests = [_digest(tool_text(t)) for t in tools]
        have: Dict[str, str] = {}
        for t, h in zip(self.tools, self._hashes):
            have.setdefault(tool_name(t), h)
        gone = [n for n in have if n not in set(names)]
        changed = [t for t, n, h in zip(tools, names, digests) if n in have and have[n] != h]
        fresh = [t for t, n in zip(tools, names) if n not in have]
        removed = self.remove(gone)
        self.update(changed, batch_size)
        self.add(fresh, batch_size)
        return {"added": len(fresh), "updated": len(changed), "removed": removed,
                "unchanged": len(tools) - len(fresh) - len(changed)}

    def save(self, path: str) -> None:
        """Write the tools and their vectors to a folder so a later process can `load` them without embedding again."""
        import numpy as np
        from . import __version__
        os.makedirs(path, exist_ok=True)
        emb = self._emb if self._emb is not None else np.zeros((0, 0), dtype="float32")
        np.save(os.path.join(path, "embeddings.npy"), emb)
        with open(os.path.join(path, "tools.json"), "w", encoding="utf-8") as f:
            json.dump(self.tools, f, ensure_ascii=False)
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"format": 1, "opentoolsearch": __version__, "model": self.model_name, "doc_prefix": self.d_prefix or "",
                       "count": len(self.tools), "dim": int(emb.shape[1]), "hashes": self._hashes}, f)

    @classmethod
    def load(cls, path: str, check: bool = True, **kwargs) -> "ToolIndex":
        """Read an index written by `save`. The embedder is loaded for queries (default: the one that built the index);
        with check=True a different embedder or document prefix raises instead of silently mixing vector spaces.
        Other keyword arguments go to the constructor (reranker, device, ...)."""
        import numpy as np
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        if "encoder" not in kwargs:
            kwargs.setdefault("model", meta["model"])
        idx = cls(**kwargs)
        if check:
            if meta["model"] != "custom" and idx.model_name != "custom" and meta["model"] != idx.model_name:
                raise ValueError(f"the index was built with {meta['model']!r}, not {idx.model_name!r} (check=False to force)")
            if meta["doc_prefix"] != (idx.d_prefix or ""):
                raise ValueError("document prefix differs from the one the index was built with (check=False to force)")
            get_dim = getattr(idx.encoder, "get_embedding_dimension", None) or getattr(idx.encoder, "get_sentence_embedding_dimension", None)
            dim = get_dim() if get_dim else None
            if dim is not None and meta["count"] and meta["dim"] != dim:
                raise ValueError(f"vectors have {meta['dim']} dimensions, this embedder makes {dim}")
        with open(os.path.join(path, "tools.json"), encoding="utf-8") as f:
            idx.tools = json.load(f)
        emb = np.load(os.path.join(path, "embeddings.npy"))
        idx._emb = emb if len(idx.tools) else None
        idx._hashes = list(meta["hashes"])
        if len(idx.tools) != len(idx._hashes) or (idx._emb is not None and len(idx._emb) != len(idx.tools)):
            raise ValueError("index files are inconsistent")
        return idx

    def __len__(self):
        return len(self.tools)

    def _query_text(self, query: str, instruction: Optional[str]) -> str:
        if self.q_prefix is not None:                       # prefix-style models
            return self.q_prefix + (f"{instruction}\n{query}" if instruction and self.embed_instruction else query)
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
