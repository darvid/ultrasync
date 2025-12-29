# ultrasync-index

Rust-backed mmapped index and vector search for
[ultrasync](https://github.com/davidgidwani/ultrasync).

This package provides the low-level index primitives:

- **GlobalIndex** - memory-mapped file index with zero-copy access
- **ThreadIndex** - in-memory vector index for fast similarity search
- **TreeSitter scanner** - AST-based symbol extraction (optional)
- **Candle embeddings** - Rust-native embeddings via HuggingFace (optional)

## Installation

```bash
pip install ultrasync-index
```

Or as part of the full ultrasync package:

```bash
pip install ultrasync
```

## Usage

```python
from ultrasync_index import GlobalIndex, ThreadIndex

# Memory-mapped global index
index = GlobalIndex("index.dat", "blob.dat")
data = index.slice_for_key(key_hash)

# In-memory thread-local index
thread_idx = ThreadIndex()
thread_idx.add(key_hash, embedding_vector)
results = thread_idx.search(query_vector, top_k=10)
```

## Features

- `scanner` - Tree-sitter based symbol extraction (enabled by default)
- `embeddings` - Candle-based Rust-native embeddings (enabled by default)
- `mkl` - Intel MKL acceleration for embeddings
- `cuda` - CUDA acceleration for embeddings

## License

MIT
