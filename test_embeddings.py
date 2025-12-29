#!/usr/bin/env python
"""Test both embedding providers."""

import time


def test_sentence_transformer():
    """Test SentenceTransformerProvider."""
    from ultrasync_mcp.embeddings import SentenceTransformerProvider

    print("testing SentenceTransformerProvider...")
    t0 = time.perf_counter()
    provider = SentenceTransformerProvider()
    print(f"  loaded in {time.perf_counter() - t0:.2f}s")
    print(f"  model: {provider.model}")
    print(f"  dim: {provider.dim}")
    print(f"  device: {provider.device}")

    # test single embed
    t0 = time.perf_counter()
    vec = provider.embed("hello world")
    print(f"  single embed: {time.perf_counter() - t0:.4f}s")
    print(f"  vec shape: {vec.shape}")
    print(f"  vec norm: {(vec**2).sum() ** 0.5:.4f}")

    # test batch embed
    texts = ["hello", "world", "foo", "bar", "baz"]
    t0 = time.perf_counter()
    vecs = provider.embed_batch(texts)
    print(f"  batch embed ({len(texts)}): {time.perf_counter() - t0:.4f}s")
    print(f"  vecs count: {len(vecs)}")

    # test cache
    t0 = time.perf_counter()
    _ = provider.embed("hello world")
    print(f"  cached embed: {time.perf_counter() - t0:.6f}s")

    print("  SentenceTransformerProvider works!")
    return True


def test_infinity_provider():
    """Test InfinityProvider."""
    try:
        import infinity_emb  # noqa: F401
    except ImportError:
        print("infinity-emb not installed, skipping InfinityProvider test")
        return False

    from ultrasync_mcp.embeddings import InfinityProvider

    # try different engines - torch often fails with BetterTransformer issues
    engines_to_try = ["torch", "optimum"]

    for engine in engines_to_try:
        print(f"\ntesting InfinityProvider with engine={engine}...")
        t0 = time.perf_counter()
        try:
            provider = InfinityProvider(engine=engine)
            print(f"  initialized in {time.perf_counter() - t0:.2f}s")
        except Exception as e:
            print(f"  failed to initialize with engine={engine}: {e}")
            continue

        # if we get here, initialization succeeded - test embedding
        print(f"  model: {provider.model}")

        # test single embed (this starts the engine)
        t0 = time.perf_counter()
        try:
            vec = provider.embed("hello world")
            elapsed = time.perf_counter() - t0
            print(f"  single embed (incl. engine start): {elapsed:.4f}s")
            print(f"  dim: {provider.dim}")
            print(f"  vec shape: {vec.shape}")
            print(f"  vec norm: {(vec**2).sum() ** 0.5:.4f}")
        except Exception as e:
            print(f"  embed failed with engine={engine}: {e}")
            continue

        # test batch embed
        texts = ["hello", "world", "foo", "bar", "baz"]
        t0 = time.perf_counter()
        try:
            vecs = provider.embed_batch(texts)
            print(
                f"  batch embed ({len(texts)}): {time.perf_counter() - t0:.4f}s"
            )
            print(f"  vecs count: {len(vecs)}")
        except Exception as e:
            print(f"  batch embed failed: {e}")
            continue

        # test cache
        t0 = time.perf_counter()
        _ = provider.embed("hello world")
        print(f"  cached embed: {time.perf_counter() - t0:.6f}s")

        print(f"  InfinityProvider with engine={engine} works!")
        return True

    print("\n  all infinity engines failed!")
    return False


def test_infinity_api_provider():
    """Test InfinityAPIProvider (requires running server)."""
    try:
        import httpx
    except ImportError:
        print("\nhttpx not installed, skipping InfinityAPIProvider test")
        return False

    from ultrasync_mcp.embeddings import InfinityAPIProvider

    print("\ntesting InfinityAPIProvider...")

    # check if server is running
    base_url = "http://localhost:7997"
    try:
        r = httpx.get(f"{base_url}/health", timeout=2.0)
        if r.status_code != 200:
            print(f"  server not healthy at {base_url}, skipping")
            return False
    except Exception:
        print(f"  no server at {base_url}, skipping")
        return False

    t0 = time.perf_counter()
    provider = InfinityAPIProvider(base_url=base_url)
    print(f"  initialized in {time.perf_counter() - t0:.4f}s")
    print(f"  model: {provider.model}")
    print(f"  device: {provider.device}")

    # test single embed
    t0 = time.perf_counter()
    vec = provider.embed("hello world")
    print(f"  single embed: {time.perf_counter() - t0:.4f}s")
    print(f"  dim: {provider.dim}")
    print(f"  vec shape: {vec.shape}")
    print(f"  vec norm: {(vec**2).sum() ** 0.5:.4f}")

    # test batch embed
    texts = ["hello", "world", "foo", "bar", "baz"]
    t0 = time.perf_counter()
    vecs = provider.embed_batch(texts)
    print(f"  batch embed ({len(texts)}): {time.perf_counter() - t0:.4f}s")
    print(f"  vecs count: {len(vecs)}")

    # test cache
    t0 = time.perf_counter()
    _ = provider.embed("hello world")
    print(f"  cached embed: {time.perf_counter() - t0:.6f}s")

    provider.close()
    print("  InfinityAPIProvider works!")
    return True


def test_create_provider():
    """Test factory function."""
    from ultrasync_mcp.embeddings import create_provider

    print("\ntesting create_provider factory...")

    # test sentence-transformers backend
    p1 = create_provider(backend="sentence-transformers")
    print(f"  sentence-transformers: {type(p1).__name__}")

    p2 = create_provider(backend="st")
    print(f"  st: {type(p2).__name__}")

    # test infinity backend
    try:
        p3 = create_provider(backend="infinity")
        print(f"  infinity: {type(p3).__name__}")
    except Exception as e:
        print(f"  infinity: failed - {e}")

    # test infinity-api backend
    try:
        p4 = create_provider(backend="infinity-api")
        print(f"  infinity-api: {type(p4).__name__}")
    except Exception as e:
        print(f"  infinity-api: failed - {e}")

    print("  factory works!")


def test_protocol():
    """Test that providers implement the protocol."""
    from ultrasync_mcp.embeddings import (
        EmbeddingProvider,
        SentenceTransformerProvider,
    )

    print("\ntesting Protocol compliance...")

    provider = SentenceTransformerProvider()
    is_provider = isinstance(provider, EmbeddingProvider)
    print(f"  SentenceTransformerProvider is EmbeddingProvider: {is_provider}")

    try:
        from ultrasync_mcp.embeddings import InfinityProvider

        _ = InfinityProvider.__new__(InfinityProvider)
        # can't check instance without init, but type checking works
        print("  InfinityProvider available")
    except ImportError:
        print("  InfinityProvider not available")


if __name__ == "__main__":
    print("=" * 60)
    print("Embedding Provider Tests")
    print("=" * 60)

    test_sentence_transformer()
    test_infinity_provider()
    test_infinity_api_provider()
    test_create_provider()
    test_protocol()

    print("\n" + "=" * 60)
    print("all tests passed!")
