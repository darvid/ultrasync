"""Tests for file scanner module."""

from pathlib import Path
from textwrap import dedent

import pytest

from ultrasync.file_scanner import FileMetadata, FileScanner, SymbolInfo


class TestSymbolInfo:
    """Test SymbolInfo dataclass."""

    def test_basic_creation(self):
        sym = SymbolInfo(name="foo", line=10, kind="function")
        assert sym.name == "foo"
        assert sym.line == 10
        assert sym.kind == "function"
        assert sym.end_line is None

    def test_with_end_line(self):
        sym = SymbolInfo(name="MyClass", line=5, kind="class", end_line=50)
        assert sym.end_line == 50


class TestFileMetadata:
    """Test FileMetadata dataclass."""

    def test_to_embedding_text(self):
        meta = FileMetadata(
            path=Path("/project/src/utils.py"),
            filename_no_ext="utils",
            exported_symbols=["foo", "bar"],
            component_names=["MyComponent"],
            top_comments=["Utility functions"],
        )
        text = meta.to_embedding_text()
        assert "/project/src/utils.py" in text
        assert "utils" in text
        assert "foo" in text
        assert "bar" in text
        assert "MyComponent" in text
        assert "Utility functions" in text

    def test_empty_metadata(self):
        meta = FileMetadata(
            path=Path("/test.py"),
            filename_no_ext="test",
        )
        text = meta.to_embedding_text()
        assert "/test.py" in text
        assert "test" in text


class TestFileScannerPython:
    """Test Python file scanning."""

    @pytest.fixture
    def scanner(self) -> FileScanner:
        return FileScanner()

    def test_scan_function(self, scanner: FileScanner, tmp_path: Path):
        """Test scanning Python function definitions."""
        py_file = tmp_path / "funcs.py"
        py_file.write_text(
            dedent("""
            def public_func():
                pass

            def another_public():
                return 42

            def _private_func():
                pass
        """).strip()
        )

        meta = scanner.scan(py_file)
        assert meta is not None
        assert "public_func" in meta.exported_symbols
        assert "another_public" in meta.exported_symbols
        assert "_private_func" not in meta.exported_symbols

    def test_scan_class(self, scanner: FileScanner, tmp_path: Path):
        """Test scanning Python class definitions."""
        py_file = tmp_path / "classes.py"
        py_file.write_text(
            dedent("""
            class MyClass:
                def method(self):
                    pass

            class AnotherClass:
                pass
        """).strip()
        )

        meta = scanner.scan(py_file)
        assert meta is not None
        assert "MyClass" in meta.exported_symbols
        assert "AnotherClass" in meta.exported_symbols

    def test_scan_constants(self, scanner: FileScanner, tmp_path: Path):
        """Test scanning Python constant assignments."""
        py_file = tmp_path / "consts.py"
        py_file.write_text(
            dedent("""
            PUBLIC_CONST = 42
            ANOTHER = "value"
            _private = "hidden"
        """).strip()
        )

        meta = scanner.scan(py_file)
        assert meta is not None
        assert "PUBLIC_CONST" in meta.exported_symbols
        assert "ANOTHER" in meta.exported_symbols
        assert "_private" not in meta.exported_symbols

    def test_scan_docstring(self, scanner: FileScanner, tmp_path: Path):
        """Test extracting module docstring."""
        py_file = tmp_path / "documented.py"
        py_file.write_text(
            dedent('''
            """This is the module docstring.

            More details here.
            """

            def foo():
                pass
        ''').strip()
        )

        meta = scanner.scan(py_file)
        assert meta is not None
        assert "This is the module docstring." in meta.top_comments

    def test_scan_async_function(self, scanner: FileScanner, tmp_path: Path):
        """Test scanning async function definitions."""
        py_file = tmp_path / "async_funcs.py"
        py_file.write_text(
            dedent("""
            async def fetch_data():
                pass

            async def process():
                return await fetch_data()
        """).strip()
        )

        meta = scanner.scan(py_file)
        assert meta is not None
        assert "fetch_data" in meta.exported_symbols
        assert "process" in meta.exported_symbols

        # check kind is async function
        async_sym = next(s for s in meta.symbol_info if s.name == "fetch_data")
        assert async_sym.kind == "async function"

    def test_scan_component_class(self, scanner: FileScanner, tmp_path: Path):
        """Test detecting component-like classes."""
        py_file = tmp_path / "component.py"
        py_file.write_text(
            dedent("""
            class MyComponent:
                def render(self):
                    return "<div/>"

            class RegularClass:
                def other_method(self):
                    pass
        """).strip()
        )

        meta = scanner.scan(py_file)
        assert meta is not None
        assert "MyComponent" in meta.component_names
        assert "RegularClass" not in meta.component_names

    def test_scan_symbol_info_lines(self, scanner: FileScanner, tmp_path: Path):
        """Test that symbol info includes line numbers."""
        py_file = tmp_path / "lines.py"
        py_file.write_text(
            dedent("""
            def first():
                pass

            def second():
                x = 1
                y = 2
                return x + y
        """).strip()
        )

        meta = scanner.scan(py_file)
        assert meta is not None

        first_sym = next(s for s in meta.symbol_info if s.name == "first")
        second_sym = next(s for s in meta.symbol_info if s.name == "second")

        assert first_sym.line == 1
        assert second_sym.line == 4
        assert second_sym.end_line == 7

    def test_scan_nonexistent_file(self, scanner: FileScanner, tmp_path: Path):
        """Test scanning nonexistent file returns None."""
        result = scanner.scan(tmp_path / "nonexistent.py")
        assert result is None

    def test_scan_syntax_error(self, scanner: FileScanner, tmp_path: Path):
        """Test scanning file with syntax errors."""
        py_file = tmp_path / "broken.py"
        py_file.write_text("def broken(:\n    pass")

        meta = scanner.scan(py_file)
        # should return metadata but with empty symbols
        assert meta is not None
        assert meta.exported_symbols == []


class TestFileScannerTypeScript:
    """Test TypeScript/JavaScript file scanning."""

    @pytest.fixture
    def scanner(self) -> FileScanner:
        return FileScanner()

    def test_scan_exports(self, scanner: FileScanner, tmp_path: Path):
        """Test scanning TypeScript exports."""
        ts_file = tmp_path / "utils.ts"
        ts_file.write_text(
            dedent("""
            export function helper() {
                return 42;
            }

            export const VALUE = "test";

            export class MyClass {
                method() {}
            }
        """).strip()
        )

        meta = scanner.scan(ts_file)
        assert meta is not None
        assert "helper" in meta.exported_symbols
        assert "VALUE" in meta.exported_symbols
        assert "MyClass" in meta.exported_symbols

    def test_scan_default_export(self, scanner: FileScanner, tmp_path: Path):
        """Test scanning default exports."""
        ts_file = tmp_path / "component.tsx"
        ts_file.write_text(
            dedent("""
            export default function MainComponent() {
                return <div />;
            }
        """).strip()
        )

        meta = scanner.scan(ts_file)
        assert meta is not None
        assert "MainComponent" in meta.exported_symbols

    def test_scan_react_components(self, scanner: FileScanner, tmp_path: Path):
        """Test detecting React components."""
        tsx_file = tmp_path / "Button.tsx"
        tsx_file.write_text(
            dedent("""
            export function Button({ onClick }) {
                return <button onClick={onClick}>Click</button>;
            }

            export const IconButton = ({ icon }) => {
                return <Button>{icon}</Button>;
            }
        """).strip()
        )

        meta = scanner.scan(tsx_file)
        assert meta is not None
        assert "Button" in meta.component_names
        assert "IconButton" in meta.component_names

    def test_scan_interface_and_type(
        self, scanner: FileScanner, tmp_path: Path
    ):
        """Test scanning TypeScript interfaces and types."""
        ts_file = tmp_path / "types.ts"
        ts_file.write_text(
            dedent("""
            export interface User {
                id: string;
                name: string;
            }

            export type Status = "active" | "inactive";
        """).strip()
        )

        meta = scanner.scan(ts_file)
        assert meta is not None
        assert "User" in meta.exported_symbols
        assert "Status" in meta.exported_symbols

    def test_scan_top_comment(self, scanner: FileScanner, tmp_path: Path):
        """Test extracting top-level comments."""
        ts_file = tmp_path / "commented.ts"
        ts_file.write_text(
            dedent("""
            // This is the main utility module
            export function util() {}
        """).strip()
        )

        meta = scanner.scan(ts_file)
        assert meta is not None
        assert "This is the main utility module" in meta.top_comments


class TestFileScannerRust:
    """Test Rust file scanning."""

    @pytest.fixture
    def scanner(self) -> FileScanner:
        return FileScanner()

    def test_scan_pub_items(self, scanner: FileScanner, tmp_path: Path):
        """Test scanning public Rust items."""
        rs_file = tmp_path / "lib.rs"
        rs_file.write_text(
            dedent("""
            pub struct MyStruct {
                field: i32,
            }

            pub enum Status {
                Active,
                Inactive,
            }

            pub fn public_func() -> i32 {
                42
            }

            fn private_func() {}
        """).strip()
        )

        meta = scanner.scan(rs_file)
        assert meta is not None
        assert "MyStruct" in meta.exported_symbols
        assert "Status" in meta.exported_symbols
        assert "public_func" in meta.exported_symbols
        assert "private_func" not in meta.exported_symbols

    def test_scan_pub_const(self, scanner: FileScanner, tmp_path: Path):
        """Test scanning public constants."""
        rs_file = tmp_path / "consts.rs"
        rs_file.write_text(
            dedent("""
            pub const MAX_SIZE: usize = 1024;
            pub static GLOBAL: &str = "test";
        """).strip()
        )

        meta = scanner.scan(rs_file)
        assert meta is not None
        assert "MAX_SIZE" in meta.exported_symbols
        assert "GLOBAL" in meta.exported_symbols

    def test_scan_impl_blocks(self, scanner: FileScanner, tmp_path: Path):
        """Test detecting impl blocks."""
        rs_file = tmp_path / "impl.rs"
        rs_file.write_text(
            dedent("""
            pub struct MyType;

            impl MyType {
                pub fn new() -> Self {
                    MyType
                }
            }
        """).strip()
        )

        meta = scanner.scan(rs_file)
        assert meta is not None
        assert "MyType" in meta.component_names

    def test_scan_doc_comments(self, scanner: FileScanner, tmp_path: Path):
        """Test extracting doc comments."""
        rs_file = tmp_path / "documented.rs"
        rs_file.write_text(
            dedent("""
            //! This is the crate documentation

            pub fn foo() {}
        """).strip()
        )

        meta = scanner.scan(rs_file)
        assert meta is not None
        assert "This is the crate documentation" in meta.top_comments


class TestFileScannerDirectory:
    """Test directory scanning."""

    @pytest.fixture
    def scanner(self) -> FileScanner:
        return FileScanner()

    def test_scan_directory(self, scanner: FileScanner, tmp_path: Path):
        """Test scanning a directory recursively."""
        # create structure
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("def main(): pass")
        (src / "utils.py").write_text("def helper(): pass")

        sub = src / "sub"
        sub.mkdir()
        (sub / "nested.py").write_text("def nested(): pass")

        results = scanner.scan_directory(tmp_path)
        assert len(results) == 3

        paths = [str(m.path) for m in results]
        assert any("main.py" in p for p in paths)
        assert any("utils.py" in p for p in paths)
        assert any("nested.py" in p for p in paths)

    def test_scan_directory_excludes(
        self, scanner: FileScanner, tmp_path: Path
    ):
        """Test that excluded directories are skipped."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("def main(): pass")

        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "installed.py").write_text("def installed(): pass")

        node = tmp_path / "node_modules"
        node.mkdir()
        (node / "package.js").write_text("export function pkg() {}")

        results = scanner.scan_directory(tmp_path)
        assert len(results) == 1
        assert "main.py" in str(results[0].path)

    def test_scan_directory_filter_extensions(
        self, scanner: FileScanner, tmp_path: Path
    ):
        """Test filtering by extension."""
        (tmp_path / "code.py").write_text("x = 1")
        (tmp_path / "code.ts").write_text("const x = 1")
        (tmp_path / "code.rs").write_text("let x = 1;")

        # only Python
        results = scanner.scan_directory(tmp_path, extensions={".py"})
        assert len(results) == 1
        assert results[0].path.suffix == ".py"

        # Python and TypeScript
        results = scanner.scan_directory(tmp_path, extensions={".py", ".ts"})
        assert len(results) == 2


class TestFileScannerReDoSProtection:
    """Test that regex patterns don't cause catastrophic backtracking.

    These tests use pathological inputs that would cause exponential time
    complexity if the regex patterns are vulnerable to ReDoS attacks.
    Each test should complete in under 1 second - if they hang, the
    pattern needs to be fixed.
    """

    @pytest.fixture
    def scanner(self) -> FileScanner:
        return FileScanner()

    def test_typescript_component_pattern_no_backtracking(
        self, scanner: FileScanner, tmp_path: Path
    ):
        """Pathological input that triggers backtracking in component regex.

        Pattern like 'export const Foo = aaaa...aaaa' with no arrow function
        could cause backtracking if using [^;]* or similar greedy patterns.
        """
        # create pathological content: lots of 'a's that don't end with =>
        pathological = "export const Component = " + "a" * 10000 + ";"

        tsx_file = tmp_path / "pathological.tsx"
        tsx_file.write_text(pathological)

        import time

        start = time.perf_counter()
        meta = scanner.scan(tsx_file)
        elapsed = time.perf_counter() - start

        # should complete in well under 1 second (actual: ~1ms)
        assert elapsed < 1.0, f"scan took {elapsed:.2f}s - possible ReDoS!"
        assert meta is not None

    def test_typescript_export_pattern_no_backtracking(
        self, scanner: FileScanner, tmp_path: Path
    ):
        """Pathological input for export detection pattern."""
        # nested/repeated patterns that could trigger backtracking
        pathological = "export " + "{ a, " * 1000 + "z }"

        ts_file = tmp_path / "exports.ts"
        ts_file.write_text(pathological)

        import time

        start = time.perf_counter()
        meta = scanner.scan(ts_file)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"scan took {elapsed:.2f}s - possible ReDoS!"
        assert meta is not None

    def test_rust_impl_pattern_no_backtracking(
        self, scanner: FileScanner, tmp_path: Path
    ):
        """Pathological input for Rust impl block detection."""
        # many impl-like patterns that don't fully match
        pathological = "\n".join([f"impl_{i}" for i in range(5000)])

        rs_file = tmp_path / "impls.rs"
        rs_file.write_text(pathological)

        import time

        start = time.perf_counter()
        meta = scanner.scan(rs_file)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"scan took {elapsed:.2f}s - possible ReDoS!"
        assert meta is not None

    def test_python_class_pattern_no_backtracking(
        self, scanner: FileScanner, tmp_path: Path
    ):
        """Pathological input for Python class detection."""
        # many class-like patterns
        pathological = "\n".join([f"class_{i} = None" for i in range(5000)])

        py_file = tmp_path / "classes.py"
        py_file.write_text(pathological)

        import time

        start = time.perf_counter()
        meta = scanner.scan(py_file)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"scan took {elapsed:.2f}s - possible ReDoS!"
        assert meta is not None
