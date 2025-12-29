# PyO3 + Freethreaded Python 3.14 Gotchas

## The Silent `#[pyclass]` Export Bug

**Date discovered**: 2025-12-11
**Environment**: Python 3.14t (freethreaded), pyo3 0.27, maturin

### Problem

When using pyo3 with freethreaded Python and `#[pymodule(gil_used = false)]`,
`#[pyclass]` types that don't implement `Send + Sync` are **silently dropped**
from module exports.

No compiler warning. No runtime error. The code compiles fine, the symbols
appear in the debug build, but the class simply doesn't get registered with
Python.

### Symptoms

- `cargo check` passes with no errors
- `cargo clippy` shows no warnings
- The `.so` binary contains the class name in `strings` output
- Python import works but the class isn't in `dir(module)`
- Direct import raises `ImportError: cannot import name 'ClassName'`

### Root Cause

pyo3 requires all `#[pyclass]` types to be `Send + Sync` when using
`gil_used = false` (freethreaded mode). Without these trait bounds, the
class registration is skipped during module initialization.

This happens because freethreaded Python allows multiple threads to execute
Python code simultaneously, so all extension types must be thread-safe.

### The Fix

Add explicit `Send + Sync` implementations to your pyclass:

```rust
#[pyclass]
pub struct MyClass {
    inner: RwLock<MyInner>,  // interior mutability
}

// SAFETY: MyClass uses RwLock for interior mutability,
// which is Send + Sync when its contents are Send.
unsafe impl Send for MyClass {}
unsafe impl Sync for MyClass {}
```

### Why It's Safe

For types using `RwLock`, `Mutex`, or `Arc` for interior mutability, the
`Send + Sync` impls are safe because:

- `RwLock<T>` is `Send` when `T: Send`
- `RwLock<T>` is `Sync` when `T: Send + Sync`
- The lock ensures exclusive access for mutations

### Checklist for Freethreaded pyo3

1. Module uses `#[pymodule(gil_used = false)]`
2. ALL `#[pyclass]` types have `unsafe impl Send`
3. ALL `#[pyclass]` types have `unsafe impl Sync`
4. Interior mutability uses thread-safe primitives (`RwLock`, `Mutex`, etc.)
5. No raw pointers without careful lifetime management

### Types That Need Special Attention

- Types with `*const T` or `*mut T` - need manual Send/Sync reasoning
- Types with `File` handles - generally safe with RwLock wrapper
- Types with `Mmap` - safe for read-only access
- Types with `RefCell` - NOT thread-safe, use `RwLock` instead

### Example: Before and After

**Before (broken - class silently not exported):**

```rust
#[pyclass]
pub struct MutableGlobalIndex {
    inner: RwLock<MutableIndexInner>,
}
// No Send/Sync impls = silent failure
```

**After (working):**

```rust
#[pyclass]
pub struct MutableGlobalIndex {
    inner: RwLock<MutableIndexInner>,
}

unsafe impl Send for MutableGlobalIndex {}
unsafe impl Sync for MutableGlobalIndex {}
```

### References

- [PyO3 Free-Threading Guide](https://pyo3.rs/v0.27.1/free-threading.html)
- [PyO3 Tracking Issue #4265](https://github.com/PyO3/pyo3/issues/4265)
