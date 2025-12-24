import ast
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from ultrasync.regex_safety import RegexTimeout, safe_compile

# Try to import the fast Rust scanner
try:
    import ultrasync_index as _rust_scanner

    _HAS_RUST_SCANNER = True
except ImportError:
    _HAS_RUST_SCANNER = False


@dataclass
class SymbolInfo:
    """A symbol with its location."""

    name: str
    line: int
    kind: str  # "class", "function", "const", etc.
    end_line: int | None = None


@dataclass
class FileMetadata:
    """Extracted metadata from a source file."""

    path: Path
    filename_no_ext: str
    exported_symbols: list[str] = field(default_factory=list)
    symbol_info: list[SymbolInfo] = field(default_factory=list)
    component_names: list[str] = field(default_factory=list)
    top_comments: list[str] = field(default_factory=list)

    def to_embedding_text(self) -> str:
        """Build the embedding-friendly string."""
        parts = [
            str(self.path),
            self.filename_no_ext,
            *self.exported_symbols,
            *self.component_names,
            *self.top_comments,
        ]
        return " ".join(parts)


class FileScanner:
    """Scans source files and extracts metadata for indexing.

    By default, uses a fast Rust/tree-sitter backend if available.
    Falls back to Python AST parsing if Rust scanner is not installed.
    """

    # file extensions we know how to parse
    PYTHON_EXTS = {".py", ".pyi"}
    TS_JS_EXTS = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}
    RUST_EXTS = {".rs"}

    # skip symbol extraction for large files (likely bundled/minified)
    MAX_SCAN_BYTES = 500_000  # 500KB

    def __init__(self, *, use_rust: bool | None = None):
        """Initialize the file scanner.

        Args:
            use_rust: Use the fast Rust/tree-sitter backend.
                      None (default) = auto-detect (use if available)
                      True = force Rust (raises if unavailable)
                      False = force Python backend
        """
        if use_rust is None:
            self._use_rust = _HAS_RUST_SCANNER
        elif use_rust:
            if not _HAS_RUST_SCANNER:
                raise ImportError(
                    "Rust scanner requested but ultrasync_index not installed. "
                    "Install with: uv run maturin develop -m "
                    "crates/ultrasync_index/Cargo.toml --release"
                )
            self._use_rust = True
        else:
            self._use_rust = False

        if self._use_rust:
            self._rust_scanner = _rust_scanner.TreeSitterScanner()

    @classmethod
    def has_rust_backend(cls) -> bool:
        """Check if the fast Rust scanner is available."""
        return _HAS_RUST_SCANNER

    def _convert_rust_result(
        self, rust_meta: "_rust_scanner.FileMetadata"
    ) -> FileMetadata:
        """Convert Rust FileMetadata to Python FileMetadata."""
        symbols = [
            SymbolInfo(
                name=sym.name,
                line=sym.line,
                kind=sym.kind,
                end_line=sym.end_line,
            )
            for sym in rust_meta.symbols
        ]
        return FileMetadata(
            path=Path(rust_meta.path),
            filename_no_ext=rust_meta.filename_no_ext,
            exported_symbols=list(rust_meta.exported_symbols),
            symbol_info=symbols,
            component_names=list(rust_meta.component_names),
            top_comments=list(rust_meta.top_comments),
        )

    def scan(
        self,
        path: Path,
        content: bytes | str | None = None,
    ) -> FileMetadata | None:
        """Scan a file and extract metadata.

        Args:
            path: Path to the file
            content: Optional pre-read content (bytes or str)
        """
        if not path.is_file():
            return None

        ext = path.suffix.lower()
        supported_exts = self.PYTHON_EXTS | self.TS_JS_EXTS | self.RUST_EXTS
        if ext not in supported_exts:
            return None

        # Use Rust backend if available
        if self._use_rust:
            content_bytes = None
            if content is not None:
                if isinstance(content, str):
                    content_bytes = content.encode("utf-8")
                else:
                    content_bytes = content

            rust_result = self._rust_scanner.scan(str(path), content_bytes)
            if rust_result is None:
                return None
            return self._convert_rust_result(rust_result)

        # Fall back to Python implementation
        filename_no_ext = path.stem

        metadata = FileMetadata(
            path=path,
            filename_no_ext=filename_no_ext,
        )

        # Use provided content or read from file
        if content is None:
            try:
                text_content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                return metadata
        elif isinstance(content, bytes):
            text_content = content.decode("utf-8", errors="ignore")
        else:
            text_content = content

        # skip symbol extraction for very large files (likely bundled/minified)
        if len(text_content) > self.MAX_SCAN_BYTES:
            return metadata

        if ext in self.PYTHON_EXTS:
            self._scan_python(text_content, metadata)
        elif ext in self.TS_JS_EXTS:
            self._scan_typescript(text_content, metadata)
        elif ext in self.RUST_EXTS:
            self._scan_rust(text_content, metadata)

        return metadata

    def scan_batch(self, paths: list[Path]) -> list[FileMetadata]:
        """Scan multiple files in parallel (Rust backend only).

        Uses rayon for parallel processing when Rust backend is available.
        Falls back to sequential scanning if using Python backend.

        Args:
            paths: List of file paths to scan

        Returns:
            List of FileMetadata for successfully scanned files
        """
        if self._use_rust:
            # Use fast parallel Rust scanner
            str_paths = [str(p) for p in paths]
            results = _rust_scanner.batch_scan_files(str_paths)
            return [
                self._convert_rust_result(r.metadata)
                for r in results
                if r.metadata is not None
            ]
        else:
            # Fall back to sequential Python scanning
            return [m for p in paths if (m := self.scan(p)) is not None]

    def scan_directory(
        self,
        root: Path,
        extensions: set[str] | None = None,
        exclude_dirs: set[str] | None = None,
    ) -> list[FileMetadata]:
        """Recursively scan a directory for source files.

        When using the Rust backend, uses parallel batch scanning for
        optimal performance.
        """
        if extensions is None:
            extensions = self.PYTHON_EXTS | self.TS_JS_EXTS | self.RUST_EXTS

        if exclude_dirs is None:
            exclude_dirs = {
                "node_modules",
                ".git",
                "__pycache__",
                ".venv",
                ".ultrasync",
                "venv",
                "target",
                "dist",
                "build",
                ".next",
            }

        # Collect all matching paths
        paths: list[Path] = []
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                # check if any parent is in exclude_dirs
                if any(part in exclude_dirs for part in path.parts):
                    continue
                paths.append(path)

        # Use batch scanning for better performance
        if self._use_rust and paths:
            return self.scan_batch(paths)

        # Fall back to sequential scanning
        return [m for p in paths if (m := self.scan(p)) is not None]

    def _scan_python(self, content: str, metadata: FileMetadata) -> None:
        """Extract symbols from Python code."""
        try:
            # suppress SyntaxWarnings from user code (deprecated escape
            # sequences, etc.) - only syntax errors matter for parsing
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=SyntaxWarning)
                tree = ast.parse(content, filename=str(metadata.path))
        except SyntaxError:
            return

        for node in ast.iter_child_nodes(tree):
            # top-level docstring
            if isinstance(node, ast.Expr) and isinstance(
                node.value, ast.Constant
            ):
                val = node.value.value
                if isinstance(val, str):
                    doc = val.strip()
                    if doc:
                        # first line only
                        metadata.top_comments.append(doc.split("\n")[0])

            # class definitions
            elif isinstance(node, ast.ClassDef):
                metadata.exported_symbols.append(node.name)
                metadata.symbol_info.append(
                    SymbolInfo(
                        name=node.name,
                        line=node.lineno,
                        kind="class",
                        end_line=node.end_lineno,
                    )
                )
                # check if it looks like a component (has render, __call__, etc)
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name in ("render", "__call__", "forward"):
                            metadata.component_names.append(node.name)
                            break

            # function definitions
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                # only export public functions (no leading underscore)
                if not node.name.startswith("_"):
                    metadata.exported_symbols.append(node.name)
                    kind = (
                        "async function"
                        if isinstance(node, ast.AsyncFunctionDef)
                        else "function"
                    )
                    metadata.symbol_info.append(
                        SymbolInfo(
                            name=node.name,
                            line=node.lineno,
                            kind=kind,
                            end_line=node.end_lineno,
                        )
                    )

            # assignments that look like exports
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if not target.id.startswith("_"):
                            metadata.exported_symbols.append(target.id)
                            metadata.symbol_info.append(
                                SymbolInfo(
                                    name=target.id,
                                    line=node.lineno,
                                    kind="const",
                                    end_line=node.end_lineno,
                                )
                            )

    def _scan_typescript(self, content: str, metadata: FileMetadata) -> None:
        """Extract symbols from TypeScript/JavaScript code (regex-based)."""

        def get_line_number(pos: int) -> int:
            return content[:pos].count("\n") + 1

        def find_block_end(start_pos: int) -> int | None:
            """Find the end line of a brace-delimited block.

            Uses indentation-based heuristic: find the first line after
            the opening brace that starts with `}` at the same or lesser
            indentation as the declaration. This is more robust than
            char-by-char brace counting which breaks on braces in
            strings/comments.
            """
            # find the opening brace line
            brace_pos = content.find("{", start_pos)
            if brace_pos == -1:
                return None

            # get indentation of the declaration line
            decl_line_start = content.rfind("\n", 0, start_pos) + 1
            decl_indent = 0
            for ch in content[decl_line_start:]:
                if ch == " ":
                    decl_indent += 1
                elif ch == "\t":
                    decl_indent += 4  # treat tab as 4 spaces
                else:
                    break

            # scan lines after the brace for closing `}`
            lines_after = content[brace_pos:].split("\n")
            brace_line = get_line_number(brace_pos)

            for i, line in enumerate(lines_after[1:], start=1):
                stripped = line.lstrip()
                if stripped.startswith("}"):
                    # check indentation
                    line_indent = len(line) - len(stripped)
                    if line_indent <= decl_indent:
                        return brace_line + i

            return None

        # exported functions and classes - safe pattern, no timeout needed
        export_pattern = re.compile(
            r"export\s+(?:default\s+)?(?:async\s+)?"
            r"(function|class|const|let|var|interface|type|enum)\s+"
            r"(\w+)",
            re.MULTILINE,
        )
        for match in export_pattern.finditer(content):
            kind = match.group(1)
            name = match.group(2)

            # find end of block for braced items
            end_line = None
            if kind in ("function", "class", "interface", "type", "enum"):
                end_line = find_block_end(match.end())

            metadata.exported_symbols.append(name)
            metadata.symbol_info.append(
                SymbolInfo(
                    name=name,
                    line=get_line_number(match.start()),
                    kind=kind,
                    end_line=end_line,
                )
            )

        # React components (function components)
        # matches: function ComponentName, const ComponentName =
        # handles arrow functions with destructured params like ({ foo }) =>
        # uses safe_compile with timeout to prevent ReDoS on pathological input
        try:
            component_pattern = safe_compile(
                r"(?:export\s+(?:default\s+)?)?(?:function|const)\s+"
                r"([A-Z]\w*)\s*(?:=\s*\([^)]*\)\s*=>|=\s*\w+\s*=>|\()",
                flags=re.MULTILINE,
                timeout_ms=500,  # 500ms should be plenty for any real file
            )
            for match in component_pattern.finditer(content):
                name = match.group(1)
                if name not in metadata.component_names:
                    metadata.component_names.append(name)
        except RegexTimeout:
            # pathological input - skip component detection for this file
            pass

        # top-level comments (JSDoc or // comments at start)
        lines = content.split("\n")
        for line in lines[:20]:  # check first 20 lines
            stripped = line.strip()
            if stripped.startswith("//"):
                comment = stripped[2:].strip()
                if comment and not comment.startswith("@"):
                    metadata.top_comments.append(comment)
                    break
            elif stripped.startswith("/*") or stripped.startswith("*"):
                # skip JSDoc tags
                if "@" not in stripped:
                    comment = stripped.lstrip("/*").lstrip("*").strip()
                    if comment:
                        metadata.top_comments.append(comment)
                        break
            elif stripped and not stripped.startswith("import"):
                break

    def _scan_rust(self, content: str, metadata: FileMetadata) -> None:
        """Extract symbols from Rust code (regex-based)."""
        lines = content.split("\n")

        def get_line_number(pos: int) -> int:
            return content[:pos].count("\n") + 1

        def find_block_end(start_pos: int) -> int | None:
            """Find the end line of a brace-delimited block."""
            # find the opening brace
            brace_pos = content.find("{", start_pos)
            if brace_pos == -1:
                return None

            depth = 1
            pos = brace_pos + 1
            while pos < len(content) and depth > 0:
                ch = content[pos]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                pos += 1

            if depth == 0:
                return get_line_number(pos - 1)
            return None

        # pub structs, enums, functions, traits
        pub_pattern = re.compile(
            r"pub\s+(?:async\s+)?(struct|enum|fn|trait|type|const|static)\s+"
            r"(\w+)",
            re.MULTILINE,
        )
        for match in pub_pattern.finditer(content):
            kind = match.group(1)
            name = match.group(2)
            start_line = get_line_number(match.start())

            # find end of block for braced items
            end_line = None
            if kind in ("struct", "enum", "fn", "trait", "impl"):
                end_line = find_block_end(match.end())

            metadata.exported_symbols.append(name)
            metadata.symbol_info.append(
                SymbolInfo(
                    name=name,
                    line=start_line,
                    kind=kind,
                    end_line=end_line,
                )
            )

        # impl blocks for types
        impl_pattern = re.compile(r"impl(?:<[^>]+>)?\s+(\w+)", re.MULTILINE)
        for match in impl_pattern.finditer(content):
            name = match.group(1)
            if name not in metadata.component_names:
                metadata.component_names.append(name)

        # doc comments at top of file
        lines = content.split("\n")
        for line in lines[:20]:
            stripped = line.strip()
            if stripped.startswith("//!") or stripped.startswith("///"):
                comment = stripped.lstrip("/!").strip()
                if comment:
                    metadata.top_comments.append(comment)
                    break
            elif stripped and not stripped.startswith("//"):
                break
