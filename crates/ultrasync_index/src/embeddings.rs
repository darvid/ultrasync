//! Rust-native embeddings using candle.
//!
//! Provides fast embedding generation using HuggingFace's candle framework.
//! Supports CPU and optional CUDA acceleration.

use pyo3::prelude::*;
use pyo3::types::PyList;
use std::sync::Arc;

#[cfg(feature = "embeddings")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "embeddings")]
use candle_nn::VarBuilder;
#[cfg(feature = "embeddings")]
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
#[cfg(feature = "embeddings")]
use hf_hub::{api::sync::Api, Repo, RepoType};
#[cfg(feature = "embeddings")]
use tokenizers::Tokenizer;

/// Default model for embeddings
const DEFAULT_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";
/// Embedding dimension for all-MiniLM-L6-v2
const EMBEDDING_DIM: usize = 384;
/// Maximum sequence length (tokens)
const MAX_SEQ_LEN: usize = 256;
/// Default max chars before truncation (~4 chars per token)
const DEFAULT_MAX_CHARS: usize = 1500;

#[cfg(feature = "embeddings")]
struct EmbedderInner {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    max_chars: usize,
}

#[cfg(feature = "embeddings")]
impl EmbedderInner {
    fn new(model_id: &str, device: Device, max_chars: usize) -> Result<Self, String> {
        // Download model files from HuggingFace hub
        let api = Api::new().map_err(|e| format!("Failed to create HF API: {e}"))?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        // Get paths to model files
        let config_path = repo
            .get("config.json")
            .map_err(|e| format!("Failed to get config.json: {e}"))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| format!("Failed to get tokenizer.json: {e}"))?;
        let weights_path = repo
            .get("model.safetensors")
            .map_err(|e| format!("Failed to get model.safetensors: {e}"))?;

        // Load config
        let config_str =
            std::fs::read_to_string(&config_path).map_err(|e| format!("Failed to read config: {e}"))?;
        let config: BertConfig =
            serde_json::from_str(&config_str).map_err(|e| format!("Failed to parse config: {e}"))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {e}"))?;

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)
                .map_err(|e| format!("Failed to load weights: {e}"))?
        };

        let model =
            BertModel::load(vb, &config).map_err(|e| format!("Failed to load BERT model: {e}"))?;

        Ok(Self {
            model,
            tokenizer,
            device,
            max_chars,
        })
    }

    fn truncate(&self, text: &str) -> String {
        if text.len() <= self.max_chars {
            text.to_string()
        } else {
            text[..self.max_chars].to_string()
        }
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Truncate texts
        let texts: Vec<String> = texts.iter().map(|t| self.truncate(t)).collect();

        // Tokenize all texts
        let encodings = self
            .tokenizer
            .encode_batch(texts.clone(), true)
            .map_err(|e| format!("Tokenization failed: {e}"))?;

        // Find max length for padding
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len().min(MAX_SEQ_LEN))
            .max()
            .unwrap_or(0);

        if max_len == 0 {
            return Ok(vec![vec![0.0; EMBEDDING_DIM]; texts.len()]);
        }

        // Build input tensors with padding
        let batch_size = encodings.len();
        let mut input_ids = vec![0u32; batch_size * max_len];
        let mut attention_mask = vec![0u32; batch_size * max_len];
        let mut token_type_ids = vec![0u32; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();
            let len = ids.len().min(max_len);

            for j in 0..len {
                input_ids[i * max_len + j] = ids[j];
                attention_mask[i * max_len + j] = mask[j];
                token_type_ids[i * max_len + j] = type_ids[j];
            }
        }

        // Create tensors
        let input_ids = Tensor::from_vec(input_ids, (batch_size, max_len), &self.device)
            .map_err(|e| format!("Failed to create input_ids tensor: {e}"))?;
        let attention_mask = Tensor::from_vec(attention_mask, (batch_size, max_len), &self.device)
            .map_err(|e| format!("Failed to create attention_mask tensor: {e}"))?;
        let token_type_ids = Tensor::from_vec(token_type_ids, (batch_size, max_len), &self.device)
            .map_err(|e| format!("Failed to create token_type_ids tensor: {e}"))?;

        // Run model forward pass
        let embeddings = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| format!("Model forward failed: {e}"))?;

        // Mean pooling over sequence length (masked)
        let attention_mask_f32 = attention_mask
            .to_dtype(DType::F32)
            .map_err(|e| format!("Failed to convert mask to f32: {e}"))?;

        // Expand mask to match embedding dims: [batch, seq, 1]
        let mask_expanded = attention_mask_f32
            .unsqueeze(2)
            .map_err(|e| format!("Failed to expand mask: {e}"))?
            .broadcast_as(embeddings.shape())
            .map_err(|e| format!("Failed to broadcast mask: {e}"))?;

        // Masked sum
        let masked_embeddings = embeddings
            .mul(&mask_expanded)
            .map_err(|e| format!("Failed to apply mask: {e}"))?;
        let sum_embeddings = masked_embeddings
            .sum(1)
            .map_err(|e| format!("Failed to sum embeddings: {e}"))?;

        // Sum of mask for averaging
        let mask_sum = attention_mask_f32
            .sum(1)
            .map_err(|e| format!("Failed to sum mask: {e}"))?
            .unsqueeze(1)
            .map_err(|e| format!("Failed to unsqueeze mask sum: {e}"))?
            .broadcast_as(sum_embeddings.shape())
            .map_err(|e| format!("Failed to broadcast mask sum: {e}"))?;

        // Mean
        let mean_embeddings = sum_embeddings
            .div(&mask_sum)
            .map_err(|e| format!("Failed to compute mean: {e}"))?;

        // L2 normalize
        let norm = mean_embeddings
            .sqr()
            .map_err(|e| format!("Failed to square for norm: {e}"))?
            .sum(1)
            .map_err(|e| format!("Failed to sum for norm: {e}"))?
            .sqrt()
            .map_err(|e| format!("Failed to sqrt for norm: {e}"))?
            .unsqueeze(1)
            .map_err(|e| format!("Failed to unsqueeze norm: {e}"))?
            .broadcast_as(mean_embeddings.shape())
            .map_err(|e| format!("Failed to broadcast norm: {e}"))?;

        let normalized = mean_embeddings
            .div(&norm)
            .map_err(|e| format!("Failed to normalize: {e}"))?;

        // Convert to Vec<Vec<f32>>
        let flat: Vec<f32> = normalized
            .to_vec2()
            .map_err(|e| format!("Failed to convert to vec: {e}"))?
            .into_iter()
            .flatten()
            .collect();

        let results: Vec<Vec<f32>> = flat
            .chunks(EMBEDDING_DIM)
            .map(|chunk| chunk.to_vec())
            .collect();

        Ok(results)
    }
}

/// Rust embedding provider using candle.
///
/// Provides fast sentence embeddings using HuggingFace's candle framework.
/// Supports CPU and optional CUDA acceleration.
#[cfg(feature = "embeddings")]
#[pyclass]
pub struct RustEmbedder {
    inner: Arc<EmbedderInner>,
    model_id: String,
}

#[cfg(feature = "embeddings")]
#[pymethods]
impl RustEmbedder {
    /// Create a new RustEmbedder.
    ///
    /// Args:
    ///     model_id: HuggingFace model ID (default: sentence-transformers/all-MiniLM-L6-v2)
    ///     device: Device to use - "cpu" or "cuda" (default: "cpu")
    ///     max_chars: Maximum characters before truncation (default: 1500)
    #[new]
    #[pyo3(signature = (model_id=None, device=None, max_chars=None))]
    pub fn new(
        model_id: Option<&str>,
        device: Option<&str>,
        max_chars: Option<usize>,
    ) -> PyResult<Self> {
        let model_id = model_id.unwrap_or(DEFAULT_MODEL);
        let max_chars = max_chars.unwrap_or(DEFAULT_MAX_CHARS);

        let device = match device.unwrap_or("cpu") {
            "cuda" | "gpu" => {
                #[cfg(feature = "cuda")]
                {
                    Device::new_cuda(0).map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "Failed to create CUDA device: {e}"
                        ))
                    })?
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "CUDA support not compiled. Rebuild with --features cuda",
                    ));
                }
            }
            _ => Device::Cpu,
        };

        let inner = EmbedderInner::new(model_id, device, max_chars)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Ok(Self {
            inner: Arc::new(inner),
            model_id: model_id.to_string(),
        })
    }

    /// Return the model ID.
    #[getter]
    pub fn model(&self) -> &str {
        &self.model_id
    }

    /// Return the embedding dimension.
    #[getter]
    pub fn dim(&self) -> usize {
        EMBEDDING_DIM
    }

    /// Return the device being used.
    #[getter]
    pub fn device(&self) -> &str {
        match &self.inner.device {
            Device::Cpu => "cpu",
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => "cuda",
            _ => "unknown",
        }
    }

    /// Embed a single text string.
    ///
    /// Returns a list of floats (the embedding vector).
    pub fn embed(&self, text: &str) -> PyResult<Vec<f32>> {
        let results = self
            .inner
            .embed_batch(&[text.to_string()])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Ok(results.into_iter().next().unwrap_or_default())
    }

    /// Embed multiple texts in a batch.
    ///
    /// Returns a list of embedding vectors.
    pub fn embed_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Py<PyList>> {
        let results = self
            .inner
            .embed_batch(&texts)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        // Convert to Python list of lists
        let py_list = PyList::empty(py);
        for vec in results {
            let inner_list = PyList::new(py, vec)?;
            py_list.append(inner_list)?;
        }

        Ok(py_list.into())
    }
}

/// Check if the embeddings feature is available.
#[pyfunction]
pub fn has_rust_embeddings() -> bool {
    cfg!(feature = "embeddings")
}

/// Register embedding classes with the Python module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(has_rust_embeddings, m)?)?;

    #[cfg(feature = "embeddings")]
    {
        m.add_class::<RustEmbedder>()?;
    }

    Ok(())
}
