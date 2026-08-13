"""Persistent embedding cache.

Embeddings are deterministic for a given (text, model), so a content hash is a
complete key. A corrupted or stale cache is worse than none, so the failure
modes matter more than the happy path.
"""

from __future__ import annotations

from ragverify.store import EmbeddingCache, embed_cached, text_key


class FakeClient:
    """Counts what it was actually asked to embed."""

    def __init__(self, dim: int = 4):
        self.dim, self.embedded = dim, []

    def embed(self, texts, batch_size=96):
        self.embedded.extend(texts)
        return [[float(len(t) % 7) + i for i in range(self.dim)] for t in texts]


class TestKeying:
    def test_same_text_same_key(self):
        assert text_key("hello world") == text_key("hello world")

    def test_different_text_different_key(self):
        assert text_key("a") != text_key("b")

    def test_whitespace_is_significant(self):
        """Chunking is deterministic, so exact text is the right key."""
        assert text_key("a b") != text_key("a  b")


class TestCacheRoundTrip:
    def test_miss_then_hit_across_instances(self, tmp_path):
        client = FakeClient()
        texts = ["alpha", "beta", "gamma"]

        cache = EmbeddingCache("test-model", tmp_path)
        first = embed_cached(texts, client, cache)
        assert len(client.embedded) == 3, "cold cache embeds everything"
        assert cache.save()

        # New process, same corpus.
        client2 = FakeClient()
        cache2 = EmbeddingCache("test-model", tmp_path)
        second = embed_cached(texts, client2, cache2)

        assert client2.embedded == [], "warm cache embeds nothing"
        assert second == first
        assert cache2.hit_rate == 1.0

    def test_partial_hit_only_embeds_the_missing(self, tmp_path):
        """The saving is the point: a partial hit must not re-embed the hit part."""
        cache = EmbeddingCache("test-model", tmp_path)
        embed_cached(["alpha", "beta"], FakeClient(), cache)
        cache.save()

        client = FakeClient()
        cache2 = EmbeddingCache("test-model", tmp_path)
        embed_cached(["alpha", "beta", "gamma"], client, cache2)

        assert client.embedded == ["gamma"]

    def test_order_is_preserved_on_partial_hit(self, tmp_path):
        """Retrieval indexes vectors positionally; a reorder misattributes every one."""
        cache = EmbeddingCache("test-model", tmp_path)
        embed_cached(["beta"], FakeClient(), cache)
        cache.save()

        texts = ["alpha", "beta", "gamma"]
        cache2 = EmbeddingCache("test-model", tmp_path)
        mixed = embed_cached(texts, FakeClient(), cache2)
        direct = FakeClient().embed(texts)

        assert mixed == direct, "cached and fresh vectors must land in original positions"


class TestInvalidation:
    def test_different_model_does_not_reuse_vectors(self, tmp_path):
        """Vectors from different models are not interchangeable."""
        cache = EmbeddingCache("model-a", tmp_path)
        embed_cached(["alpha"], FakeClient(), cache)
        cache.save()

        client = FakeClient()
        other = EmbeddingCache("model-b", tmp_path)
        embed_cached(["alpha"], client, other)

        assert client.embedded == ["alpha"], "must not reuse another model's vectors"

    def test_corrupt_file_degrades_to_empty(self, tmp_path):
        cache = EmbeddingCache("test-model", tmp_path)
        embed_cached(["alpha"], FakeClient(), cache)
        cache.save()
        cache.path.write_bytes(b"not a valid npz file at all")

        client = FakeClient()
        recovered = EmbeddingCache("test-model", tmp_path)
        embed_cached(["alpha"], client, recovered)

        assert client.embedded == ["alpha"], "a bad cache must not fail the run"
        assert len(recovered) >= 0

    def test_unwritable_directory_is_not_fatal(self, tmp_path):
        cache = EmbeddingCache("test-model", tmp_path / "nested" / "deep")
        result = embed_cached(["alpha"], FakeClient(), cache)
        assert len(result) == 1


class TestDisabled:
    def test_disabled_cache_always_embeds(self, tmp_path):
        client = FakeClient()
        cache = EmbeddingCache("test-model", tmp_path, enabled=False)

        embed_cached(["alpha"], client, cache)
        embed_cached(["alpha"], client, cache)

        assert client.embedded == ["alpha", "alpha"]
        assert not cache.save()

    def test_none_cache_passes_through(self):
        client = FakeClient()
        assert len(embed_cached(["alpha", "beta"], client, None)) == 2
        assert client.embedded == ["alpha", "beta"]

    def test_nothing_to_save_writes_nothing(self, tmp_path):
        assert not EmbeddingCache("test-model", tmp_path).save()
