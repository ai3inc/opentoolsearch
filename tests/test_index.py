"""Index maintenance (add / update / remove / sync / save / load) with a stand-in embedder, so no model is downloaded.

  python tests/test_index.py        (or: python -m pytest tests)
"""
import hashlib, os, re, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from opentoolsearch import ToolIndex


class Stub:
    """Bag-of-words embedder: words hashed into `dim` buckets. Counts how many texts it was asked to embed."""
    prompts = {}
    max_seq_length = 512

    def __init__(self, dim=64):
        self.dim, self.calls = dim, 0

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(self, texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True):
        self.calls += len(texts)
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)


def mk(**kw):
    s = Stub(); return ToolIndex(encoder=s, reranker=None, **kw), s


T = lambda n, d: {"name": n, "description": d}
BASE = [T("weather_forecast", "forecast temperature rain wind for a city"), T("send_email", "send an email message to a recipient"),
        T("convert_currency", "convert money between currencies exchange rate")]
top = lambda idx, q: idx.search(q, k=1)[0]["name"]


def test_add_is_incremental():
    idx, s = mk(); idx.add(BASE); before = idx._emb.copy(); n0 = s.calls
    idx.add([T("book_flight", "book a flight ticket airline")])
    assert s.calls - n0 == 1, "only the new tool is embedded"
    assert np.array_equal(before, idx._emb[:3]) and len(idx) == 4
    assert top(idx, "book a flight ticket") == "book_flight"


def test_update_replaces_in_place():
    idx, s = mk(); idx.add(BASE); n0 = s.calls
    idx.update([T("send_email", "post a message to a slack channel for the team")])
    assert s.calls - n0 == 1 and len(idx) == 3 and idx.names() == ["weather_forecast", "send_email", "convert_currency"]
    assert top(idx, "post to the slack channel") == "send_email"
    idx.update([T("new_tool", "translate text between languages")])
    assert len(idx) == 4 and idx.names()[-1] == "new_tool" and top(idx, "translate text languages") == "new_tool"


def test_remove():
    idx, _ = mk(); idx.add(BASE)
    assert idx.remove("send_email") == 1 and len(idx) == 2 and "send_email" not in idx.names()
    assert idx.remove(["weather_forecast", "nope"]) == 1 and idx.remove("nope") == 0
    assert idx.remove("convert_currency") == 1 and len(idx) == 0 and idx.search("anything") == []
    idx.add(BASE[:1]); assert len(idx) == 1 and top(idx, "forecast rain") == "weather_forecast"      # usable again after emptying


def test_sync_embeds_only_what_changed():
    idx, s = mk(); idx.add(BASE); n0 = s.calls
    now = [BASE[0], T("send_email", "post a message to a slack channel"), T("book_flight", "book a flight ticket")]    # 1 same, 1 changed, 1 new, 1 gone
    r = idx.sync(now)
    assert r == {"added": 1, "updated": 1, "removed": 1, "unchanged": 1}, r
    assert s.calls - n0 == 2 and sorted(idx.names()) == ["book_flight", "send_email", "weather_forecast"]
    assert idx.sync(now) == {"added": 0, "updated": 0, "removed": 0, "unchanged": 3} and s.calls - n0 == 2


def test_duplicate_names_rejected():
    idx, _ = mk(); idx.add(BASE)
    for call in (idx.update, idx.sync):
        try:
            call([T("a", "x"), T("a", "y")]); assert False
        except ValueError:
            pass


def test_update_cleans_stray_duplicates():
    idx, _ = mk(); idx.add(BASE + [T("send_email", "second copy with the same name")])
    idx.update([T("send_email", "post to slack")]); assert idx.names().count("send_email") == 1 and len(idx) == 3


def test_allowed_still_works_after_changes():
    idx, _ = mk(); idx.add(BASE); idx.remove("send_email")
    assert [h["name"] for h in idx.search("send an email", k=3, allowed=["convert_currency"])] == ["convert_currency"]


def test_save_load_roundtrip():
    idx, s = mk(); idx.add(BASE)
    with tempfile.TemporaryDirectory() as d:
        idx.save(d)
        s2 = Stub(); back = ToolIndex.load(d, encoder=s2, reranker=None)
        assert s2.calls == 0, "loading must not embed anything"
        assert back.names() == idx.names() and np.array_equal(back._emb, idx._emb)
        assert [h["name"] for h in back.search("exchange rate money", k=3)] == [h["name"] for h in idx.search("exchange rate money", k=3)]
        n = s2.calls; back.update([T("send_email", "post to slack")]); assert s2.calls - n == 1        # still maintainable after a load
        try:
            ToolIndex.load(d, encoder=Stub(dim=32), reranker=None); assert False
        except ValueError:
            pass
        try:
            ToolIndex.load(d, encoder=Stub(), reranker=None, doc_prefix="passage: "); assert False
        except ValueError:
            pass
        ToolIndex.load(d, encoder=Stub(dim=32), reranker=None, check=False)                # explicit override


def test_save_load_empty():
    idx, _ = mk()
    with tempfile.TemporaryDirectory() as d:
        idx.save(d); back = ToolIndex.load(d, encoder=Stub(), reranker=None)
        assert len(back) == 0 and back.search("x") == []; back.add(BASE); assert len(back) == 3


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
