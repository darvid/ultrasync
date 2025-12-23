use std::ffi::c_int;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem;
use std::path::Path;
use std::sync::RwLock;

use memmap2::Mmap;
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;

const MAGIC: &[u8; 8] = b"FXINDEX\0";
const HEADER_SIZE: usize = 32;
const BUCKET_SIZE: usize = 24;
const EMPTY_KEY: u64 = 0;
const TOMBSTONE_KEY: u64 = u64::MAX;

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

#[pyclass]
pub struct MutableGlobalIndex {
    inner: RwLock<MutableIndexInner>,
}

// SAFETY: MutableGlobalIndex uses RwLock for interior mutability,
// which is Send + Sync when its contents are Send.
unsafe impl Send for MutableGlobalIndex {}
unsafe impl Sync for MutableGlobalIndex {}

struct MutableIndexInner {
    path: String,
    capacity: u64,
    count: u64,
    tombstone_count: u64,
    file: File,
}

#[pymethods]
impl MutableGlobalIndex {
    #[staticmethod]
    pub fn create(path: &str, initial_capacity: u64) -> PyResult<Self> {
        let capacity = initial_capacity.next_power_of_two().max(64);

        let mut file =
            OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path)?;

        let header = Header {
            magic: *MAGIC,
            version: 1,
            reserved: 0,
            capacity,
            bucket_offset: HEADER_SIZE as u64,
        };

        let header_bytes: [u8; HEADER_SIZE] = unsafe { std::mem::transmute_copy(&header) };
        file.write_all(&header_bytes)?;

        let zeros = vec![0u8; capacity as usize * BUCKET_SIZE];
        file.write_all(&zeros)?;
        file.sync_all()?;

        Ok(Self {
            inner: RwLock::new(MutableIndexInner {
                path: path.to_string(),
                capacity,
                count: 0,
                tombstone_count: 0,
                file,
            }),
        })
    }

    #[staticmethod]
    pub fn open(path: &str) -> PyResult<Self> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;

        let mut header_bytes = [0u8; HEADER_SIZE];
        file.read_exact(&mut header_bytes)?;
        let header: Header = unsafe { std::mem::transmute_copy(&header_bytes) };

        if &header.magic != MAGIC {
            return Err(PyValueError::new_err("bad magic"));
        }

        let mut count = 0u64;
        let mut tombstone_count = 0u64;

        for i in 0..header.capacity {
            file.seek(SeekFrom::Start(header.bucket_offset + i * BUCKET_SIZE as u64))?;
            let mut bucket_bytes = [0u8; BUCKET_SIZE];
            file.read_exact(&mut bucket_bytes)?;
            let key_hash = u64::from_le_bytes(bucket_bytes[0..8].try_into().unwrap());

            if key_hash == TOMBSTONE_KEY {
                tombstone_count += 1;
            } else if key_hash != EMPTY_KEY {
                count += 1;
            }
        }

        Ok(Self {
            inner: RwLock::new(MutableIndexInner {
                path: path.to_string(),
                capacity: header.capacity,
                count,
                tombstone_count,
                file,
            }),
        })
    }

    pub fn insert(&self, key_hash: u64, offset: u64, length: u32) -> PyResult<bool> {
        if key_hash == EMPTY_KEY || key_hash == TOMBSTONE_KEY {
            return Err(PyValueError::new_err("invalid key_hash"));
        }

        let mut inner = self.inner.write().unwrap();

        let load = (inner.count + inner.tombstone_count) as f64 / inner.capacity as f64;
        if load > 0.7 {
            Self::grow_inner(&mut inner)?;
        }

        let mut idx = key_hash % inner.capacity;

        for _ in 0..inner.capacity {
            let bucket = Self::read_bucket(&mut inner.file, idx)?;

            if bucket.key_hash == EMPTY_KEY || bucket.key_hash == TOMBSTONE_KEY {
                let was_tombstone = bucket.key_hash == TOMBSTONE_KEY;
                Self::write_bucket(&mut inner.file, idx, key_hash, offset, length)?;
                inner.count += 1;
                if was_tombstone {
                    inner.tombstone_count -= 1;
                }
                return Ok(true);
            }

            if bucket.key_hash == key_hash {
                Self::write_bucket(&mut inner.file, idx, key_hash, offset, length)?;
                return Ok(false);
            }

            idx = (idx + 1) % inner.capacity;
        }

        Err(PyValueError::new_err("index full"))
    }

    pub fn remove(&self, key_hash: u64) -> PyResult<bool> {
        if key_hash == EMPTY_KEY || key_hash == TOMBSTONE_KEY {
            return Ok(false);
        }

        let mut inner = self.inner.write().unwrap();
        let mut idx = key_hash % inner.capacity;

        for _ in 0..inner.capacity {
            let bucket = Self::read_bucket(&mut inner.file, idx)?;

            if bucket.key_hash == EMPTY_KEY {
                return Ok(false);
            }

            if bucket.key_hash == key_hash {
                Self::write_bucket(&mut inner.file, idx, TOMBSTONE_KEY, 0, 0)?;
                inner.count -= 1;
                inner.tombstone_count += 1;
                return Ok(true);
            }

            idx = (idx + 1) % inner.capacity;
        }

        Ok(false)
    }

    pub fn lookup(&self, key_hash: u64) -> PyResult<Option<(u64, u32)>> {
        if key_hash == EMPTY_KEY || key_hash == TOMBSTONE_KEY {
            return Ok(None);
        }

        let mut inner = self.inner.write().unwrap();
        let mut idx = key_hash % inner.capacity;

        for _ in 0..inner.capacity {
            let bucket = Self::read_bucket(&mut inner.file, idx)?;

            if bucket.key_hash == EMPTY_KEY {
                return Ok(None);
            }

            if bucket.key_hash == key_hash {
                return Ok(Some((bucket.offset, bucket.length)));
            }

            idx = (idx + 1) % inner.capacity;
        }

        Ok(None)
    }

    pub fn capacity(&self) -> u64 {
        self.inner.read().unwrap().capacity
    }

    pub fn count(&self) -> u64 {
        self.inner.read().unwrap().count
    }

    pub fn tombstone_count(&self) -> u64 {
        self.inner.read().unwrap().tombstone_count
    }

    pub fn needs_compaction(&self) -> bool {
        let inner = self.inner.read().unwrap();
        inner.tombstone_count > inner.capacity / 4
    }

    pub fn size_bytes(&self) -> u64 {
        let inner = self.inner.read().unwrap();
        (HEADER_SIZE + inner.capacity as usize * BUCKET_SIZE) as u64
    }
}

impl MutableGlobalIndex {
    fn read_bucket(file: &mut File, idx: u64) -> PyResult<Bucket> {
        let pos = HEADER_SIZE as u64 + idx * BUCKET_SIZE as u64;
        file.seek(SeekFrom::Start(pos))?;

        let mut bytes = [0u8; BUCKET_SIZE];
        file.read_exact(&mut bytes)?;

        Ok(Bucket {
            key_hash: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            offset: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            length: u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            flags: u32::from_le_bytes(bytes[20..24].try_into().unwrap()),
        })
    }

    fn write_bucket(
        file: &mut File,
        idx: u64,
        key_hash: u64,
        offset: u64,
        length: u32,
    ) -> PyResult<()> {
        let pos = HEADER_SIZE as u64 + idx * BUCKET_SIZE as u64;
        file.seek(SeekFrom::Start(pos))?;

        let mut bytes = [0u8; BUCKET_SIZE];
        bytes[0..8].copy_from_slice(&key_hash.to_le_bytes());
        bytes[8..16].copy_from_slice(&offset.to_le_bytes());
        bytes[16..20].copy_from_slice(&length.to_le_bytes());
        bytes[20..24].copy_from_slice(&0u32.to_le_bytes());

        file.write_all(&bytes)?;
        Ok(())
    }

    fn grow_inner(inner: &mut MutableIndexInner) -> PyResult<()> {
        let new_capacity = inner.capacity * 2;
        let new_path = format!("{}.new", inner.path);

        let mut new_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&new_path)?;

        let header = Header {
            magic: *MAGIC,
            version: 1,
            reserved: 0,
            capacity: new_capacity,
            bucket_offset: HEADER_SIZE as u64,
        };

        let header_bytes: [u8; HEADER_SIZE] = unsafe { std::mem::transmute_copy(&header) };
        new_file.write_all(&header_bytes)?;

        let zeros = vec![0u8; new_capacity as usize * BUCKET_SIZE];
        new_file.write_all(&zeros)?;
        new_file.sync_all()?;

        let mut new_count = 0u64;
        for i in 0..inner.capacity {
            let bucket = Self::read_bucket(&mut inner.file, i)?;
            if bucket.key_hash != EMPTY_KEY && bucket.key_hash != TOMBSTONE_KEY {
                let mut idx = bucket.key_hash % new_capacity;
                loop {
                    let existing = Self::read_bucket(&mut new_file, idx)?;
                    if existing.key_hash == EMPTY_KEY {
                        Self::write_bucket(
                            &mut new_file,
                            idx,
                            bucket.key_hash,
                            bucket.offset,
                            bucket.length,
                        )?;
                        new_count += 1;
                        break;
                    }
                    idx = (idx + 1) % new_capacity;
                }
            }
        }

        std::fs::rename(&new_path, &inner.path)?;

        inner.file = OpenOptions::new().read(true).write(true).open(&inner.path)?;
        inner.capacity = new_capacity;
        inner.count = new_count;
        inner.tombstone_count = 0;

        Ok(())
    }
}

// =============================================================================
// Graph Snapshot - Read-only mmap'd graph for benchmarking
// =============================================================================

const GRAPH_MAGIC: &[u8; 4] = b"GRPH";
const GRAPH_HEADER_SIZE: usize = 48;
const GRAPH_NODE_SIZE: usize = 22; // node_id(8) + type_idx(2) + payload_off(8) + payload_len(4)

#[repr(C)]
#[derive(Clone, Copy)]
struct GraphHeader {
    magic: [u8; 4],
    version: u32,
    node_count: u64,
    edge_count: u64,
    intern_offset: u64,
    node_offset: u64,
    adj_offset: u64,
}

/// A node entry in the graph snapshot.
#[pyclass]
#[derive(Clone)]
pub struct GraphNodeEntry {
    #[pyo3(get)]
    pub node_id: u64,
    #[pyo3(get)]
    pub type_idx: u16,
    #[pyo3(get)]
    pub payload_offset: u64,
    #[pyo3(get)]
    pub payload_length: u32,
}

/// An adjacency entry (edge) in the graph snapshot.
#[pyclass]
#[derive(Clone)]
pub struct GraphAdjEntry {
    #[pyo3(get)]
    pub rel_id: u32,
    #[pyo3(get)]
    pub target_id: u64,
    #[pyo3(get)]
    pub flags: u8,
}

/// Read-only mmapped graph snapshot for benchmarking and regression detection.
///
/// Binary format:
/// - Header (48 bytes): magic, version, counts, offsets
/// - Intern table: null-terminated strings for node types/relation names
/// - Node table: sorted by node_id for binary search
/// - Adjacency arrays: packed per-node edge lists
#[pyclass]
pub struct GraphSnapshot {
    mmap: Mmap,
    header: GraphHeader,
}

unsafe impl Send for GraphSnapshot {}
unsafe impl Sync for GraphSnapshot {}

#[pymethods]
impl GraphSnapshot {
    /// Open a graph snapshot file.
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let file = File::open(Path::new(path))?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < GRAPH_HEADER_SIZE {
            return Err(PyValueError::new_err("graph file too small"));
        }

        let header = unsafe { *(mmap.as_ptr() as *const GraphHeader) };

        if &header.magic != GRAPH_MAGIC {
            return Err(PyValueError::new_err("bad graph magic"));
        }

        if header.version != 1 {
            return Err(PyValueError::new_err("unsupported graph version"));
        }

        Ok(Self { mmap, header })
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> u64 {
        self.header.node_count
    }

    /// Number of edges in the graph.
    pub fn edge_count(&self) -> u64 {
        self.header.edge_count
    }

    /// Get a node by ID using binary search.
    ///
    /// Returns None if node not found.
    pub fn get_node(&self, node_id: u64) -> Option<GraphNodeEntry> {
        let node_count = self.header.node_count as usize;
        if node_count == 0 {
            return None;
        }

        let base = self.header.node_offset as usize;

        // Binary search for node_id
        let mut left = 0;
        let mut right = node_count;

        while left < right {
            let mid = left + (right - left) / 2;
            let offset = base + mid * GRAPH_NODE_SIZE;

            if offset + GRAPH_NODE_SIZE > self.mmap.len() {
                return None;
            }

            let mid_id = self.read_node_id(offset);

            if mid_id == node_id {
                return Some(self.read_node_entry(offset));
            } else if mid_id < node_id {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        None
    }

    /// Get outgoing edges for a node.
    ///
    /// Returns list of (rel_id, target_id, flags) tuples.
    pub fn get_out(&self, node_id: u64) -> Vec<GraphAdjEntry> {
        self.get_adjacency(node_id, true)
    }

    /// Get incoming edges for a node.
    ///
    /// Returns list of (rel_id, source_id, flags) tuples.
    pub fn get_in(&self, node_id: u64) -> Vec<GraphAdjEntry> {
        self.get_adjacency(node_id, false)
    }

    /// Get an interned string by index.
    pub fn get_intern(&self, idx: u32) -> Option<String> {
        let base = self.header.intern_offset as usize;
        let end = self.header.node_offset as usize;

        if base >= self.mmap.len() || base >= end {
            return None;
        }

        // Skip to idx-th null-terminated string
        let mut offset = base;
        let mut current_idx = 0u32;

        while offset < end && current_idx < idx {
            while offset < end && self.mmap[offset] != 0 {
                offset += 1;
            }
            offset += 1; // skip null
            current_idx += 1;
        }

        if current_idx != idx || offset >= end {
            return None;
        }

        // Read string until null
        let start = offset;
        while offset < end && self.mmap[offset] != 0 {
            offset += 1;
        }

        String::from_utf8(self.mmap[start..offset].to_vec()).ok()
    }

    /// Iterate all nodes.
    pub fn iter_nodes(&self) -> Vec<GraphNodeEntry> {
        let mut nodes = Vec::with_capacity(self.header.node_count as usize);
        let base = self.header.node_offset as usize;

        for i in 0..self.header.node_count as usize {
            let offset = base + i * GRAPH_NODE_SIZE;
            if offset + GRAPH_NODE_SIZE <= self.mmap.len() {
                nodes.push(self.read_node_entry(offset));
            }
        }

        nodes
    }
}

impl GraphSnapshot {
    fn read_node_id(&self, offset: usize) -> u64 {
        let bytes: [u8; 8] = self.mmap[offset..offset + 8].try_into().unwrap();
        u64::from_le_bytes(bytes)
    }

    fn read_node_entry(&self, offset: usize) -> GraphNodeEntry {
        let node_id = self.read_node_id(offset);
        let type_idx = u16::from_le_bytes(self.mmap[offset + 8..offset + 10].try_into().unwrap());
        let payload_offset =
            u64::from_le_bytes(self.mmap[offset + 10..offset + 18].try_into().unwrap());
        let payload_length =
            u32::from_le_bytes(self.mmap[offset + 18..offset + 22].try_into().unwrap());

        GraphNodeEntry { node_id, type_idx, payload_offset, payload_length }
    }

    fn get_adjacency(&self, node_id: u64, outgoing: bool) -> Vec<GraphAdjEntry> {
        // Adjacency format: node_id(8) + count(4) + entries...
        // Each entry: rel_id(4) + target_id(8) + flags(1) = 13 bytes
        const ADJ_ENTRY_SIZE: usize = 13;

        let base = self.header.adj_offset as usize;
        let mut offset = base;
        let end = self.mmap.len();

        // Scan for node_id (adjacency lists are stored by node_id)
        // Format: [direction(1)][node_id(8)][count(4)][entries...]
        while offset + 13 < end {
            let dir = self.mmap[offset];
            let is_out = dir == b'O';
            offset += 1;

            let adj_node_id = u64::from_le_bytes(self.mmap[offset..offset + 8].try_into().unwrap());
            offset += 8;

            let count = u32::from_le_bytes(self.mmap[offset..offset + 4].try_into().unwrap());
            offset += 4;

            let entries_size = count as usize * ADJ_ENTRY_SIZE;

            if adj_node_id == node_id && is_out == outgoing {
                // Found it - read entries
                let mut entries = Vec::with_capacity(count as usize);

                for _ in 0..count {
                    if offset + ADJ_ENTRY_SIZE > end {
                        break;
                    }

                    let rel_id =
                        u32::from_le_bytes(self.mmap[offset..offset + 4].try_into().unwrap());
                    let target_id =
                        u64::from_le_bytes(self.mmap[offset + 4..offset + 12].try_into().unwrap());
                    let flags = self.mmap[offset + 12];

                    entries.push(GraphAdjEntry { rel_id, target_id, flags });
                    offset += ADJ_ENTRY_SIZE;
                }

                return entries;
            }

            // Skip this adjacency list
            offset += entries_size;
        }

        Vec::new()
    }
}

/// Export a graph snapshot from LMDB data.
///
/// Args:
///     output_path: Path to write the snapshot file
///     nodes: List of (node_id, type_idx, payload_offset, payload_length)
///     adj_out: Dict of node_id -> [(rel_id, target_id, flags), ...]
///     adj_in: Dict of node_id -> [(rel_id, source_id, flags), ...]
///     intern_strings: List of strings to intern (types, relations)
///
/// Returns:
///     Dict with stats (bytes_written, node_count, edge_count)
#[pyfunction]
pub fn export_graph_snapshot(
    output_path: &str,
    nodes: Vec<(u64, u16, u64, u32)>,
    adj_out: std::collections::HashMap<u64, Vec<(u32, u64, u8)>>,
    adj_in: std::collections::HashMap<u64, Vec<(u32, u64, u8)>>,
    intern_strings: Vec<String>,
) -> PyResult<std::collections::HashMap<String, u64>> {
    let mut file = OpenOptions::new().write(true).create(true).truncate(true).open(output_path)?;

    // Build intern table
    let mut intern_bytes = Vec::new();
    for s in &intern_strings {
        intern_bytes.extend_from_slice(s.as_bytes());
        intern_bytes.push(0); // null terminator
    }

    // Sort nodes by ID
    let mut sorted_nodes = nodes.clone();
    sorted_nodes.sort_by_key(|(id, _, _, _)| *id);

    // Calculate offsets
    let intern_offset = GRAPH_HEADER_SIZE as u64;
    let node_offset = intern_offset + intern_bytes.len() as u64;
    let adj_offset = node_offset + (sorted_nodes.len() * GRAPH_NODE_SIZE) as u64;

    // Count edges
    let edge_count: u64 = adj_out.values().map(|v| v.len() as u64).sum();

    // Write header
    let header = GraphHeader {
        magic: *GRAPH_MAGIC,
        version: 1,
        node_count: sorted_nodes.len() as u64,
        edge_count,
        intern_offset,
        node_offset,
        adj_offset,
    };

    let header_bytes: [u8; GRAPH_HEADER_SIZE] = unsafe { mem::transmute_copy(&header) };
    file.write_all(&header_bytes)?;

    // Write intern table
    file.write_all(&intern_bytes)?;

    // Write nodes
    for (node_id, type_idx, payload_off, payload_len) in &sorted_nodes {
        file.write_all(&node_id.to_le_bytes())?;
        file.write_all(&type_idx.to_le_bytes())?;
        file.write_all(&payload_off.to_le_bytes())?;
        file.write_all(&payload_len.to_le_bytes())?;
    }

    // Write adjacency lists
    // Outgoing
    for (node_id, edges) in &adj_out {
        file.write_all(b"O")?; // direction marker
        file.write_all(&node_id.to_le_bytes())?;
        file.write_all(&(edges.len() as u32).to_le_bytes())?;
        for (rel_id, target_id, flags) in edges {
            file.write_all(&rel_id.to_le_bytes())?;
            file.write_all(&target_id.to_le_bytes())?;
            file.write_all(&[*flags])?;
        }
    }

    // Incoming
    for (node_id, edges) in &adj_in {
        file.write_all(b"I")?; // direction marker
        file.write_all(&node_id.to_le_bytes())?;
        file.write_all(&(edges.len() as u32).to_le_bytes())?;
        for (rel_id, source_id, flags) in edges {
            file.write_all(&rel_id.to_le_bytes())?;
            file.write_all(&source_id.to_le_bytes())?;
            file.write_all(&[*flags])?;
        }
    }

    file.sync_all()?;

    let bytes_written = file.metadata()?.len();

    let mut stats = std::collections::HashMap::new();
    stats.insert("bytes_written".to_string(), bytes_written);
    stats.insert("node_count".to_string(), sorted_nodes.len() as u64);
    stats.insert("edge_count".to_string(), edge_count);

    Ok(stats)
}

#[pymodule(gil_used = false)]
fn ultrasync_index(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BlobView>()?;
    m.add_class::<GlobalIndex>()?;
    m.add_class::<ThreadIndex>()?;
    m.add_class::<MutableGlobalIndex>()?;
    m.add_class::<GraphSnapshot>()?;
    m.add_class::<GraphNodeEntry>()?;
    m.add_class::<GraphAdjEntry>()?;
    m.add_function(wrap_pyfunction!(export_graph_snapshot, m)?)?;
    Ok(())
}
