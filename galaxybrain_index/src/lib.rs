use std::ffi::c_int;
use std::fs::File;
use std::mem;
use std::path::Path;
use std::sync::RwLock;

use memmap2::Mmap;
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;

const MAGIC: &[u8; 8] = b"FXINDEX\0";
const HEADER_SIZE: usize = 32;
const EMPTY_KEY: u64 = 0;

#[repr(C)]
#[derive(Clone, Copy)]
struct Header {
    magic: [u8; 8],
    version: u32,
    reserved: u32,
    capacity: u64,
    bucket_offset: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Bucket {
    key_hash: u64,
    offset: u64,
    length: u32,
    flags: u32,
}

/// Zero-copy view into mmapped blob data.
///
/// Implements Python's buffer protocol for direct memory access.
/// The view is only valid while the parent GlobalIndex is alive.
#[pyclass]
pub struct BlobView {
    ptr: *const u8,
    len: usize,
}

// SAFETY: BlobView holds a read-only pointer into an mmap.
unsafe impl Send for BlobView {}
unsafe impl Sync for BlobView {}

#[pymethods]
impl BlobView {
    /// Length in bytes.
    fn __len__(&self) -> usize {
        self.len
    }

    /// Implement buffer protocol for zero-copy access.
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let this = slf.borrow();

        // reject writable requests
        if (flags & ffi::PyBUF_WRITABLE) != 0 {
            return Err(PyValueError::new_err("buffer is read-only"));
        }

        unsafe {
            (*view).buf = this.ptr as *mut _;
            (*view).len = this.len as isize;
            (*view).readonly = 1;
            (*view).itemsize = 1;
            (*view).format = if (flags & ffi::PyBUF_FORMAT) != 0 {
                c"B".as_ptr() as *mut _
            } else {
                std::ptr::null_mut()
            };
            (*view).ndim = 1;
            (*view).shape = if (flags & ffi::PyBUF_ND) != 0 {
                &mut (*view).len as *mut _ as *mut _
            } else {
                std::ptr::null_mut()
            };
            (*view).strides = if (flags & ffi::PyBUF_STRIDES) != 0 {
                &mut (*view).itemsize as *mut _ as *mut _
            } else {
                std::ptr::null_mut()
            };
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = std::ptr::null_mut();
            // prevent Python from freeing the buffer
            (*view).obj = slf.as_ptr();
            ffi::Py_INCREF((*view).obj);
        }

        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {
        // nothing to release - mmap stays alive via GlobalIndex
    }
}

/// Mmapped global index for zero-copy blob access.
///
/// Maps `index.dat` (header + buckets) and `blob.dat` (raw bytes).
/// Provides `slice_for_key` to get a memoryview into blob data.
#[pyclass]
pub struct GlobalIndex {
    #[allow(dead_code)]
    index_mmap: Mmap,
    blob_mmap: Mmap,
    capacity: u64,
    bucket_base: *const u8,
}

// SAFETY: GlobalIndex only holds read-only mmaps and a pointer derived from them.
// The pointer is only used for reading and the mmap lifetime is tied to the struct.
unsafe impl Send for GlobalIndex {}
unsafe impl Sync for GlobalIndex {}

#[pymethods]
impl GlobalIndex {
    #[new]
    pub fn new(index_path: &str, blob_path: &str) -> PyResult<Self> {
        let index_file = File::open(Path::new(index_path))?;
        let index_mmap = unsafe { Mmap::map(&index_file)? };

        if index_mmap.len() < HEADER_SIZE {
            return Err(PyValueError::new_err("index too small"));
        }

        let header = unsafe { *(index_mmap.as_ptr() as *const Header) };
        if &header.magic != MAGIC {
            return Err(PyValueError::new_err("bad magic"));
        }
        if header.capacity == 0 {
            return Err(PyValueError::new_err("capacity zero"));
        }

        let bucket_base = unsafe { index_mmap.as_ptr().add(header.bucket_offset as usize) };

        let blob_file = File::open(Path::new(blob_path))?;
        let blob_mmap = unsafe { Mmap::map(&blob_file)? };

        Ok(Self { index_mmap, blob_mmap, capacity: header.capacity, bucket_base })
    }

    /// Look up a key and return a zero-copy view into blob data.
    ///
    /// Returns None if key_hash is 0 (empty) or not found.
    /// The returned BlobView implements the buffer protocol.
    pub fn slice_for_key(&self, key_hash: u64) -> PyResult<Option<BlobView>> {
        if key_hash == EMPTY_KEY {
            return Ok(None);
        }
        if let Some((offset, length)) = self.lookup(key_hash) {
            let start = offset as usize;
            let end = start + length as usize;
            if end > self.blob_mmap.len() {
                return Err(PyValueError::new_err("slice out of bounds"));
            }
            Ok(Some(BlobView {
                ptr: unsafe { self.blob_mmap.as_ptr().add(start) },
                len: length as usize,
            }))
        } else {
            Ok(None)
        }
    }

    /// Get a zero-copy view of the entire blob.
    ///
    /// Useful for scanning without key lookup.
    pub fn blob_view(&self) -> BlobView {
        BlobView { ptr: self.blob_mmap.as_ptr(), len: self.blob_mmap.len() }
    }

    /// Look up a key and return (offset, length) into the blob.
    ///
    /// Use with blob_view() to get a slice: `memoryview(blob_view)[offset:offset+length]`
    pub fn offset_for_key(&self, key_hash: u64) -> Option<(u64, u32)> {
        if key_hash == EMPTY_KEY {
            return None;
        }
        self.lookup(key_hash)
    }

    /// Number of buckets in the index.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }
}

impl GlobalIndex {
    fn lookup(&self, key_hash: u64) -> Option<(u64, u32)> {
        let mut idx = (key_hash as u128 % self.capacity as u128) as u64;
        for _ in 0..self.capacity {
            let bucket = unsafe { self.read_bucket(idx as usize) };
            if bucket.key_hash == EMPTY_KEY {
                return None;
            }
            if bucket.key_hash == key_hash {
                return Some((bucket.offset, bucket.length));
            }
            idx = (idx + 1) % self.capacity;
        }
        None
    }

    unsafe fn read_bucket(&self, i: usize) -> Bucket {
        let off = i * mem::size_of::<Bucket>();
        *(self.bucket_base.add(off) as *const Bucket)
    }
}

struct ThreadIndexInner {
    ids: Vec<u32>,
    vecs: Vec<f32>,
}

/// In-memory vector index for small working sets.
///
/// Stores vectors in a flat buffer and does brute-force cosine similarity.
/// Optimized for N < 500 items. Thread-safe for free-threaded Python.
#[pyclass]
pub struct ThreadIndex {
    dim: usize,
    inner: RwLock<ThreadIndexInner>,
}

#[pymethods]
impl ThreadIndex {
    #[new]
    pub fn new(dim: usize) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dimension must be > 0"));
        }
        Ok(Self { dim, inner: RwLock::new(ThreadIndexInner { ids: Vec::new(), vecs: Vec::new() }) })
    }

    /// Insert or update a vector by id.
    pub fn upsert(&self, id: u32, vec: Vec<f32>) -> PyResult<()> {
        if vec.len() != self.dim {
            return Err(PyValueError::new_err("dimension mismatch"));
        }

        let mut inner = self.inner.write().unwrap();

        if let Some(pos) = inner.ids.iter().position(|&x| x == id) {
            let start = pos * self.dim;
            inner.vecs[start..start + self.dim].copy_from_slice(&vec);
            return Ok(());
        }

        inner.ids.push(id);
        inner.vecs.extend_from_slice(&vec);
        Ok(())
    }

    /// Find the k nearest neighbors by cosine similarity.
    ///
    /// Returns list of (id, score) tuples, sorted by descending score.
    pub fn search(&self, query: Vec<f32>, k: usize) -> PyResult<Vec<(u32, f32)>> {
        if query.len() != self.dim {
            return Err(PyValueError::new_err("dimension mismatch"));
        }

        let inner = self.inner.read().unwrap();
        let n = inner.ids.len();
        if n == 0 || k == 0 {
            return Ok(Vec::new());
        }

        let mut scores = Vec::with_capacity(n);
        for (i, id) in inner.ids.iter().enumerate() {
            let start = i * self.dim;
            let score = cosine(&query, &inner.vecs[start..start + self.dim]);
            scores.push((*id, score));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k.min(n));
        Ok(scores)
    }

    /// Remove a vector by id. Returns true if found and removed.
    pub fn remove(&self, id: u32) -> bool {
        let mut inner = self.inner.write().unwrap();
        if let Some(pos) = inner.ids.iter().position(|&x| x == id) {
            inner.ids.remove(pos);
            let start = pos * self.dim;
            inner.vecs.drain(start..start + self.dim);
            true
        } else {
            false
        }
    }

    /// Number of vectors stored.
    pub fn len(&self) -> usize {
        self.inner.read().unwrap().ids.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.inner.read().unwrap().ids.is_empty()
    }

    /// Vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Clear all vectors.
    pub fn clear(&self) {
        let mut inner = self.inner.write().unwrap();
        inner.ids.clear();
        inner.vecs.clear();
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[pymodule(gil_used = false)]
fn galaxybrain_index(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BlobView>()?;
    m.add_class::<GlobalIndex>()?;
    m.add_class::<ThreadIndex>()?;
    Ok(())
}
