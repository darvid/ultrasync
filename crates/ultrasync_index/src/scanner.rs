//! Fast file scanning with tree-sitter.
//!
//! Extracts symbols (functions, classes, etc.) from source files using
//! tree-sitter grammars. Much faster than Python's ast module and supports
//! parallel batch processing via rayon.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::path::Path;
use tree_sitter::{Language, Node, Parser};

// Note: No init_scanner() function - grammars are loaded lazily on first scan
// to avoid segfault during PyO3 module initialization

/// Maximum file size to scan (500KB) - skip larger files
const MAX_SCAN_BYTES: usize = 500_000;

/// A symbol extracted from source code.
#[pyclass]
#[derive(Clone, Debug)]
pub struct SymbolInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub line: usize,
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub end_line: Option<usize>,
}

#[pymethods]
impl SymbolInfo {
    fn __repr__(&self) -> String {
        format!(
            "SymbolInfo(name={:?}, kind={:?}, line={}, end_line={:?})",
            self.name, self.kind, self.line, self.end_line
        )
    }
}

/// Metadata extracted from a source file.
#[pyclass]
#[derive(Clone, Debug)]
pub struct FileMetadata {
    #[pyo3(get)]
    pub path: String,
    #[pyo3(get)]
    pub filename_no_ext: String,
    #[pyo3(get)]
    pub symbols: Vec<SymbolInfo>,
    #[pyo3(get)]
    pub exported_symbols: Vec<String>,
    #[pyo3(get)]
    pub component_names: Vec<String>,
    #[pyo3(get)]
    pub top_comments: Vec<String>,
}

#[pymethods]
impl FileMetadata {
    /// Build embedding-friendly text from metadata.
    fn to_embedding_text(&self) -> String {
        let mut parts = vec![self.path.clone(), self.filename_no_ext.clone()];
        parts.extend(self.exported_symbols.iter().cloned());
        parts.extend(self.component_names.iter().cloned());
        parts.extend(self.top_comments.iter().cloned());
        parts.join(" ")
    }

    fn __repr__(&self) -> String {
        format!(
            "FileMetadata(path={:?}, symbols={}, exports={})",
            self.path,
            self.symbols.len(),
            self.exported_symbols.len()
        )
    }
}

/// Result of scanning a single file.
#[pyclass]
#[derive(Clone)]
pub struct ScanResult {
    #[pyo3(get)]
    pub path: String,
    #[pyo3(get)]
    pub metadata: Option<FileMetadata>,
    #[pyo3(get)]
    pub error: Option<String>,
}

/// Fast file scanner using tree-sitter.
#[pyclass]
#[derive(Default)]
pub struct TreeSitterScanner {
    // We can't store parsers because they're not Send/Sync
    // Instead we create them per-thread in scan methods
}

#[pymethods]
impl TreeSitterScanner {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Scan a single file and extract metadata.
    ///
    /// Args:
    ///     path: Path to the source file
    ///     content: Optional pre-read content (bytes)
    ///
    /// Returns:
    ///     FileMetadata or None if unsupported/error
    #[pyo3(signature = (path, content=None))]
    pub fn scan(&self, path: &str, content: Option<&[u8]>) -> PyResult<Option<FileMetadata>> {
        let path_obj = Path::new(path);

        // Get extension
        let ext = match path_obj.extension().and_then(|e| e.to_str()) {
            Some(e) => e.to_lowercase(),
            None => return Ok(None),
        };

        // Get content
        let content_bytes: Vec<u8>;
        let content_ref = match content {
            Some(c) => c,
            None => {
                content_bytes = match std::fs::read(path) {
                    Ok(c) => c,
                    Err(_) => return Ok(None),
                };
                &content_bytes
            }
        };

        // Skip large files
        if content_ref.len() > MAX_SCAN_BYTES {
            return Ok(Some(FileMetadata {
                path: path.to_string(),
                filename_no_ext: path_obj
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string(),
                symbols: vec![],
                exported_symbols: vec![],
                component_names: vec![],
                top_comments: vec![],
            }));
        }

        // Parse based on extension
        let metadata = match ext.as_str() {
            "py" | "pyi" => scan_python(path, content_ref),
            "ts" | "tsx" => scan_typescript(path, content_ref, true),
            "js" | "jsx" | "mjs" | "cjs" => scan_typescript(path, content_ref, false),
            "rs" => scan_rust(path, content_ref),
            _ => return Ok(None),
        };

        Ok(metadata)
    }

    /// Scan multiple files in parallel using rayon.
    ///
    /// Args:
    ///     paths: List of file paths to scan
    ///
    /// Returns:
    ///     List of ScanResult (one per input path)
    pub fn scan_batch(&self, paths: Vec<String>) -> Vec<ScanResult> {
        paths
            .into_par_iter()
            .map(|path| match self.scan(&path, None) {
                Ok(metadata) => ScanResult { path: path.clone(), metadata, error: None },
                Err(e) => {
                    ScanResult { path: path.clone(), metadata: None, error: Some(e.to_string()) }
                }
            })
            .collect()
    }

    /// Scan multiple files with pre-read content in parallel.
    ///
    /// Args:
    ///     items: List of (path, content) tuples
    ///
    /// Returns:
    ///     List of ScanResult
    pub fn scan_batch_with_content(&self, items: Vec<(String, Vec<u8>)>) -> Vec<ScanResult> {
        items
            .into_par_iter()
            .map(|(path, content)| match self.scan(&path, Some(&content)) {
                Ok(metadata) => ScanResult { path: path.clone(), metadata, error: None },
                Err(e) => {
                    ScanResult { path: path.clone(), metadata: None, error: Some(e.to_string()) }
                }
            })
            .collect()
    }
}

// =============================================================================
// Python scanning
// =============================================================================

fn scan_python(path: &str, content: &[u8]) -> Option<FileMetadata> {
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_python::LANGUAGE.into()).ok()?;

    let tree = parser.parse(content, None)?;
    let root = tree.root_node();

    let path_obj = Path::new(path);
    let filename_no_ext = path_obj.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();

    let mut symbols = Vec::new();
    let mut exported_symbols = Vec::new();
    let mut component_names = Vec::new();
    let mut top_comments = Vec::new();

    // Walk the tree directly instead of using queries (more reliable)
    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        match child.kind() {
            "class_definition" => {
                if let Some(name_node) = child.child_by_field_name("name") {
                    let name = name_node.utf8_text(content).unwrap_or("").to_string();
                    exported_symbols.push(name.clone());
                    symbols.push(SymbolInfo {
                        name: name.clone(),
                        line: child.start_position().row + 1,
                        kind: "class".to_string(),
                        end_line: Some(child.end_position().row + 1),
                    });

                    // Check for component-like classes
                    if has_component_method(&child, content) {
                        component_names.push(name);
                    }
                }
            }
            "function_definition" => {
                if let Some(name_node) = child.child_by_field_name("name") {
                    let name = name_node.utf8_text(content).unwrap_or("").to_string();

                    // Skip private functions
                    if !name.starts_with('_') {
                        exported_symbols.push(name.clone());

                        // Check if async
                        let is_async =
                            child.child(0).map(|c: Node| c.kind() == "async").unwrap_or(false);

                        let kind = if is_async { "async function" } else { "function" };

                        symbols.push(SymbolInfo {
                            name,
                            line: child.start_position().row + 1,
                            kind: kind.to_string(),
                            end_line: Some(child.end_position().row + 1),
                        });
                    }
                }
            }
            "expression_statement" => {
                // Check for assignments or docstrings
                let mut inner_cursor = child.walk();
                for inner in child.children(&mut inner_cursor) {
                    if inner.kind() == "assignment" {
                        // Get the left side (target)
                        if let Some(target) = inner.child_by_field_name("left") {
                            if target.kind() == "identifier" {
                                let name = target.utf8_text(content).unwrap_or("").to_string();
                                if !name.starts_with('_') {
                                    exported_symbols.push(name.clone());
                                    symbols.push(SymbolInfo {
                                        name,
                                        line: child.start_position().row + 1,
                                        kind: "const".to_string(),
                                        end_line: Some(child.end_position().row + 1),
                                    });
                                }
                            }
                        }
                    } else if inner.kind() == "string" {
                        // Module-level docstring
                        if top_comments.is_empty() && inner.start_position().row < 10 {
                            let doc = inner.utf8_text(content).unwrap_or("");
                            let doc_str: String =
                                doc.trim_matches(|c: char| c == '"' || c == '\'').to_string();
                            if let Some(first_line) = doc_str.lines().next() {
                                let trimmed = first_line.trim();
                                if !trimmed.is_empty() {
                                    top_comments.push(trimmed.to_string());
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Some(FileMetadata {
        path: path.to_string(),
        filename_no_ext,
        symbols,
        exported_symbols,
        component_names,
        top_comments,
    })
}

fn has_component_method(class_node: &tree_sitter::Node, content: &[u8]) -> bool {
    let mut cursor = class_node.walk();
    for child in class_node.children(&mut cursor) {
        if child.kind() == "block" {
            let mut block_cursor = child.walk();
            for block_child in child.children(&mut block_cursor) {
                if block_child.kind() == "function_definition" {
                    if let Some(name_node) = block_child.child_by_field_name("name") {
                        let name = name_node.utf8_text(content).unwrap_or("");
                        if name == "render" || name == "__call__" || name == "forward" {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

// =============================================================================
// TypeScript/JavaScript scanning
// =============================================================================

fn scan_typescript(path: &str, content: &[u8], is_typescript: bool) -> Option<FileMetadata> {
    let mut parser = Parser::new();

    // Use TSX parser for .tsx files, regular TS otherwise
    let is_tsx = path.ends_with(".tsx") || path.ends_with(".jsx");
    let language: Language = if is_tsx {
        tree_sitter_typescript::LANGUAGE_TSX.into()
    } else if is_typescript {
        tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()
    } else {
        // For JS, use TSX parser as it handles JSX too
        tree_sitter_typescript::LANGUAGE_TSX.into()
    };

    parser.set_language(&language).ok()?;

    let tree = parser.parse(content, None)?;
    let root = tree.root_node();

    let path_obj = Path::new(path);
    let filename_no_ext = path_obj.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();

    let mut symbols = Vec::new();
    let mut exported_symbols = Vec::new();
    let mut component_names = Vec::new();
    let mut top_comments = Vec::new();

    // Walk the AST for exports and declarations
    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        match child.kind() {
            "export_statement" => {
                extract_export(&child, content, &mut symbols, &mut exported_symbols);
            }
            "comment" => {
                if top_comments.is_empty() && child.start_position().row < 20 {
                    let text = child.utf8_text(content).unwrap_or("");
                    let cleaned = text
                        .trim_start_matches("//")
                        .trim_start_matches("/*")
                        .trim_end_matches("*/")
                        .trim_start_matches('*')
                        .trim();
                    if !cleaned.is_empty() && !cleaned.starts_with('@') {
                        top_comments.push(cleaned.to_string());
                    }
                }
            }
            "function_declaration" | "class_declaration" | "lexical_declaration" => {
                // Top-level declarations that might be components
                if let Some(name) = get_declaration_name(&child, content) {
                    if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        component_names.push(name);
                    }
                }
            }
            _ => {}
        }
    }

    Some(FileMetadata {
        path: path.to_string(),
        filename_no_ext,
        symbols,
        exported_symbols,
        component_names,
        top_comments,
    })
}

fn extract_export(
    export_node: &tree_sitter::Node,
    content: &[u8],
    symbols: &mut Vec<SymbolInfo>,
    exported_symbols: &mut Vec<String>,
) {
    let mut cursor = export_node.walk();
    for child in export_node.children(&mut cursor) {
        match child.kind() {
            "function_declaration" => {
                if let Some(name_node) = child.child_by_field_name("name") {
                    let name = name_node.utf8_text(content).unwrap_or("").to_string();
                    exported_symbols.push(name.clone());
                    symbols.push(SymbolInfo {
                        name,
                        line: child.start_position().row + 1,
                        kind: "function".to_string(),
                        end_line: Some(child.end_position().row + 1),
                    });
                }
            }
            "class_declaration" => {
                if let Some(name_node) = child.child_by_field_name("name") {
                    let name = name_node.utf8_text(content).unwrap_or("").to_string();
                    exported_symbols.push(name.clone());
                    symbols.push(SymbolInfo {
                        name,
                        line: child.start_position().row + 1,
                        kind: "class".to_string(),
                        end_line: Some(child.end_position().row + 1),
                    });
                }
            }
            "lexical_declaration" => {
                // const/let/var
                let kind_str = child
                    .child(0)
                    .map(|c| c.utf8_text(content).unwrap_or("const"))
                    .unwrap_or("const");

                let mut decl_cursor = child.walk();
                for decl_child in child.children(&mut decl_cursor) {
                    if decl_child.kind() == "variable_declarator" {
                        if let Some(name_node) = decl_child.child_by_field_name("name") {
                            let name = name_node.utf8_text(content).unwrap_or("").to_string();
                            exported_symbols.push(name.clone());
                            symbols.push(SymbolInfo {
                                name,
                                line: child.start_position().row + 1,
                                kind: kind_str.to_string(),
                                end_line: Some(child.end_position().row + 1),
                            });
                        }
                    }
                }
            }
            "interface_declaration" | "type_alias_declaration" => {
                if let Some(name_node) = child.child_by_field_name("name") {
                    let name = name_node.utf8_text(content).unwrap_or("").to_string();
                    let kind =
                        if child.kind() == "interface_declaration" { "interface" } else { "type" };
                    exported_symbols.push(name.clone());
                    symbols.push(SymbolInfo {
                        name,
                        line: child.start_position().row + 1,
                        kind: kind.to_string(),
                        end_line: Some(child.end_position().row + 1),
                    });
                }
            }
            "enum_declaration" => {
                if let Some(name_node) = child.child_by_field_name("name") {
                    let name = name_node.utf8_text(content).unwrap_or("").to_string();
                    exported_symbols.push(name.clone());
                    symbols.push(SymbolInfo {
                        name,
                        line: child.start_position().row + 1,
                        kind: "enum".to_string(),
                        end_line: Some(child.end_position().row + 1),
                    });
                }
            }
            _ => {}
        }
    }
}

fn get_declaration_name(node: &tree_sitter::Node, content: &[u8]) -> Option<String> {
    match node.kind() {
        "function_declaration" | "class_declaration" => node
            .child_by_field_name("name")
            .and_then(|n| n.utf8_text(content).ok())
            .map(|s| s.to_string()),
        "lexical_declaration" => {
            // Find variable_declarator child
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() == "variable_declarator" {
                    return child
                        .child_by_field_name("name")
                        .and_then(|n| n.utf8_text(content).ok())
                        .map(|s| s.to_string());
                }
            }
            None
        }
        _ => None,
    }
}

// =============================================================================
// Rust scanning
// =============================================================================

fn scan_rust(path: &str, content: &[u8]) -> Option<FileMetadata> {
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_rust::LANGUAGE.into()).ok()?;

    let tree = parser.parse(content, None)?;
    let root = tree.root_node();

    let path_obj = Path::new(path);
    let filename_no_ext = path_obj.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();

    let mut symbols = Vec::new();
    let mut exported_symbols = Vec::new();
    let mut component_names = Vec::new();
    let mut top_comments = Vec::new();

    // Walk the tree directly
    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        match child.kind() {
            "function_item" => {
                if has_pub_visibility(&child, content) {
                    if let Some(name_node) = child.child_by_field_name("name") {
                        let name = name_node.utf8_text(content).unwrap_or("").to_string();
                        exported_symbols.push(name.clone());
                        symbols.push(SymbolInfo {
                            name,
                            line: child.start_position().row + 1,
                            kind: "fn".to_string(),
                            end_line: Some(child.end_position().row + 1),
                        });
                    }
                }
            }
            "struct_item" => {
                if has_pub_visibility(&child, content) {
                    if let Some(name_node) = child.child_by_field_name("name") {
                        let name = name_node.utf8_text(content).unwrap_or("").to_string();
                        exported_symbols.push(name.clone());
                        symbols.push(SymbolInfo {
                            name,
                            line: child.start_position().row + 1,
                            kind: "struct".to_string(),
                            end_line: Some(child.end_position().row + 1),
                        });
                    }
                }
            }
            "enum_item" => {
                if has_pub_visibility(&child, content) {
                    if let Some(name_node) = child.child_by_field_name("name") {
                        let name = name_node.utf8_text(content).unwrap_or("").to_string();
                        exported_symbols.push(name.clone());
                        symbols.push(SymbolInfo {
                            name,
                            line: child.start_position().row + 1,
                            kind: "enum".to_string(),
                            end_line: Some(child.end_position().row + 1),
                        });
                    }
                }
            }
            "trait_item" => {
                if has_pub_visibility(&child, content) {
                    if let Some(name_node) = child.child_by_field_name("name") {
                        let name = name_node.utf8_text(content).unwrap_or("").to_string();
                        exported_symbols.push(name.clone());
                        symbols.push(SymbolInfo {
                            name,
                            line: child.start_position().row + 1,
                            kind: "trait".to_string(),
                            end_line: Some(child.end_position().row + 1),
                        });
                    }
                }
            }
            "impl_item" => {
                // Get the type being implemented
                if let Some(type_node) = child.child_by_field_name("type") {
                    let name = type_node.utf8_text(content).unwrap_or("").to_string();
                    if !component_names.contains(&name) {
                        component_names.push(name);
                    }
                }
            }
            "line_comment" => {
                if top_comments.is_empty() && child.start_position().row < 20 {
                    let text = child.utf8_text(content).unwrap_or("");
                    let comment = text.trim_start_matches("//").trim();
                    if (comment.starts_with('!') || comment.starts_with('/')) && !comment.is_empty()
                    {
                        let cleaned = comment.trim_start_matches(['!', '/']).trim();
                        if !cleaned.is_empty() {
                            top_comments.push(cleaned.to_string());
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Some(FileMetadata {
        path: path.to_string(),
        filename_no_ext,
        symbols,
        exported_symbols,
        component_names,
        top_comments,
    })
}

fn has_pub_visibility(node: &Node, content: &[u8]) -> bool {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "visibility_modifier" {
            let text = child.utf8_text(content).unwrap_or("");
            return text.starts_with("pub");
        }
    }
    false
}

// =============================================================================
// Standalone functions for batch scanning
// =============================================================================

/// Scan files in parallel and return results.
///
/// This is the main entry point for batch scanning from Python.
#[pyfunction]
pub fn batch_scan_files(paths: Vec<String>) -> Vec<ScanResult> {
    let scanner = TreeSitterScanner::new();
    scanner.scan_batch(paths)
}

/// Scan files with pre-read content in parallel.
#[pyfunction]
pub fn batch_scan_files_with_content(items: Vec<(String, Vec<u8>)>) -> Vec<ScanResult> {
    let scanner = TreeSitterScanner::new();
    scanner.scan_batch_with_content(items)
}
