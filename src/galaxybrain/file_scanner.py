import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from galaxybrain.regex_safety import RegexTimeout, safe_compile


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
    """Scans source files and extracts metadata for indexing."""

    # file extensions we know how to parse
    PYTHON_EXTS = {".py", ".pyi"}
    TS_JS_EXTS = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}
    RUST_EXTS = {".rs"}

    # skip symbol extraction for large files (likely bundled/minified)
    MAX_SCAN_BYTES = 500_000  # 500KB

    def scan(self, path: Path) -> FileMetadata | None:
        """Scan a file and extract metadata."""
        if not path.is_file():
            return None

        ext = path.suffix.lower()
        supported_exts = self.PYTHON_EXTS | self.TS_JS_EXTS | self.RUST_EXTS
        if ext not in supported_exts:
            return None

        filename_no_ext = path.stem

        metadata = FileMetadata(
            path=path,
            filename_no_ext=filename_no_ext,
        )

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return metadata

        # skip symbol extraction for very large files (likely bundled/minified)
        if len(content) > self.MAX_SCAN_BYTES:
            return metadata

        if ext in self.PYTHON_EXTS:
            self._scan_python(content, metadata)
        elif ext in self.TS_JS_EXTS:
            self._scan_typescript(content, metadata)
        elif ext in self.RUST_EXTS:
            self._scan_rust(content, metadata)

        return metadata

    def scan_directory(
        self,
        root: Path,
        extensions: set[str] | None = None,
        exclude_dirs: set[str] | None = None,
    ) -> list[FileMetadata]:
        """Recursively scan a directory for source files."""
        if extensions is None:
            extensions = self.PYTHON_EXTS | self.TS_JS_EXTS | self.RUST_EXTS

        if exclude_dirs is None:
            exclude_dirs = {
                "node_modules",
                ".git",
                "__pycache__",
                ".venv",
                "venv",
                "target",
                "dist",
                "build",
                ".next",
            }

        results: list[FileMetadata] = []

        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                # check if any parent is in exclude_dirs
                if any(part in exclude_dirs for part in path.parts):
                    continue
                metadata = self.scan(path)
                if metadata:
                    results.append(metadata)

        return results

    def _scan_python(self, content: str, metadata: FileMetadata) -> None:
        """Extract symbols from Python code."""
        try:
            tree = ast.parse(content)
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
            metadata.exported_symbols.append(name)
            metadata.symbol_info.append(
                SymbolInfo(
                    name=name,
                    line=get_line_number(match.start()),
                    kind=kind,
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
