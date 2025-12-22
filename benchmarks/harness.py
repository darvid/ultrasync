"""Benchmark harness for running Claude with different MCP configs."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from benchmarks.metrics import MetricsExtractor, ToolCallMetrics


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""

    name: str  # e.g., "with-ultrasync" or "baseline"
    mcp_config_path: Path
    description: str = ""

    # claude CLI flags
    max_turns: int = 15
    model: str = "sonnet"  # sonnet, opus, haiku
    timeout_seconds: int = 120

    # environment overrides
    env_overrides: dict[str, str] = field(default_factory=dict)


@dataclass
class TestCase:
    """A single benchmark test case."""

    id: str
    name: str
    prompt: str
    category: str
    difficulty: str = "medium"
    expected_files: list[str] = field(default_factory=list)
    expected_symbols: list[str] = field(default_factory=list)
    expected_concepts: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    test_case: TestCase
    config_name: str
    success: bool
    error: str | None = None

    # timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    elapsed_seconds: float = 0.0

    # raw output
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0

    # parsed metrics
    metrics: ToolCallMetrics | None = None

    # transcript (if available)
    transcript: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "test_case_id": self.test_case.id,
            "test_case_name": self.test_case.name,
            "config_name": self.config_name,
            "success": self.success,
            "error": self.error,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "elapsed_seconds": self.elapsed_seconds,
            "exit_code": self.exit_code,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "stdout_length": len(self.stdout),
            "stderr_length": len(self.stderr),
        }


class BenchmarkHarness:
    """Runs benchmarks comparing Claude with/without ultrasync."""

    def __init__(
        self,
        project_path: Path,
        ultrasync_dir: Path | None = None,
        output_dir: Path | None = None,
    ):
        self.project_path = Path(project_path).resolve()
        self.ultrasync_dir = (
            Path(ultrasync_dir).resolve()
            if ultrasync_dir
            else Path(__file__).parent.parent.resolve()
        )
        self.output_dir = (
            Path(output_dir).resolve()
            if output_dir
            else Path(__file__).parent / "results"
        )
        self.metrics_extractor = MetricsExtractor()

        # create output dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_test_cases(self, cases_dir: Path | None = None) -> list[TestCase]:
        """Load test cases from YAML files."""
        if cases_dir is None:
            cases_dir = Path(__file__).parent / "cases"

        cases = []
        for yaml_file in sorted(cases_dir.glob("*.yaml")):
            with open(yaml_file) as f:
                data = yaml.safe_load(f)

            for case_data in data.get("cases", []):
                cases.append(
                    TestCase(
                        id=case_data["id"],
                        name=case_data["name"],
                        prompt=case_data["prompt"].strip(),
                        category=case_data.get("category", "general"),
                        difficulty=case_data.get("difficulty", "medium"),
                        expected_files=case_data.get("expected_files", []),
                        expected_symbols=case_data.get("expected_symbols", []),
                        expected_concepts=case_data.get(
                            "expected_concepts", []
                        ),
                    )
                )

        return cases

    def get_default_configs(self) -> tuple[RunConfig, RunConfig]:
        """Get default with-ultrasync and baseline configs."""
        config_dir = Path(__file__).parent / "config"

        with_ultrasync = RunConfig(
            name="with-ultrasync",
            mcp_config_path=config_dir / "with-ultrasync.json",
            description="Claude with ultrasync MCP server enabled",
            env_overrides={
                "ULTRASYNC_DIR": str(self.ultrasync_dir),
                "ULTRASYNC_PROJECT_ROOT": str(self.project_path),
            },
        )

        baseline = RunConfig(
            name="baseline",
            mcp_config_path=config_dir / "baseline.json",
            description="Claude without ultrasync (built-in tools only)",
        )

        return with_ultrasync, baseline

    def _prepare_mcp_config(self, config: RunConfig) -> Path:
        """Prepare MCP config with env vars substituted."""
        with open(config.mcp_config_path) as f:
            config_content = f.read()

        # substitute env vars
        for key, value in config.env_overrides.items():
            config_content = config_content.replace(f"${{{key}}}", value)

        # write to temp file
        temp_config = self.output_dir / f".{config.name}-config.json"
        with open(temp_config, "w") as f:
            f.write(config_content)

        return temp_config

    async def run_single(
        self,
        test_case: TestCase,
        config: RunConfig,
        session_id: str | None = None,
        resume: bool = False,
    ) -> BenchmarkResult:
        """Run a single test case with a specific config.

        Args:
            test_case: The test case to run
            config: The run configuration
            session_id: Optional session ID for session persistence
            resume: If True, resume existing session (keeps MCP warm)
        """
        result = BenchmarkResult(
            test_case=test_case,
            config_name=config.name,
            success=False,
        )

        # prepare config file with env substitution
        mcp_config = self._prepare_mcp_config(config)

        # build claude command
        # use stream-json + verbose to get full transcript with tool calls
        # strict-mcp-config ignores user-level MCP servers for clean comparison
        cmd = [
            "claude",
            "--mcp-config",
            str(mcp_config),
            "--strict-mcp-config",
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--max-turns",
            str(config.max_turns),
            "--model",
            config.model,
        ]

        # add session management flags
        if session_id:
            if resume:
                cmd.extend(["--resume", session_id])
            else:
                cmd.extend(["--session-id", session_id])

        cmd.extend(["-p", test_case.prompt])

        # run with timeout
        start_time = time.perf_counter()
        result.start_time = datetime.now()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_path,
                env={**os.environ, **config.env_overrides},
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=config.timeout_seconds,
                )
                result.stdout = stdout.decode("utf-8", errors="replace")
                result.stderr = stderr.decode("utf-8", errors="replace")
                result.exit_code = proc.returncode or 0
                result.success = result.exit_code == 0

            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                result.error = f"Timeout after {config.timeout_seconds}s"
                result.success = False

        except Exception as e:
            result.error = str(e)
            result.success = False

        result.end_time = datetime.now()
        result.elapsed_seconds = time.perf_counter() - start_time

        # parse metrics from output
        if result.stdout:
            try:
                result.transcript = self._parse_transcript(result.stdout)
                result.metrics = self.metrics_extractor.extract(
                    result.transcript
                )
            except Exception as e:
                result.error = (
                    result.error or ""
                ) + f"; metrics parse error: {e}"

        return result

    def _parse_transcript(self, stdout: str) -> list[dict[str, Any]]:
        """Parse JSON transcript from claude output."""
        # claude --output-format json outputs a JSON object
        try:
            data = json.loads(stdout)
            if isinstance(data, dict):
                return data.get("messages", [data])
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            # try line-by-line JSON (stream format)
            messages = []
            for line in stdout.strip().split("\n"):
                line = line.strip()
                if line:
                    try:
                        messages.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            return messages

    async def run_comparison(
        self,
        test_cases: list[TestCase] | None = None,
        configs: tuple[RunConfig, RunConfig] | None = None,
        parallel: bool = False,
        use_session_resumption: bool = True,
    ) -> dict[str, list[BenchmarkResult]]:
        """Run all test cases with both configs and return results.

        Args:
            test_cases: Test cases to run (loads from cases/ if None)
            configs: Tuple of (with-ultrasync, baseline) configs
            parallel: Not yet implemented
            use_session_resumption: If True, reuse sessions to avoid
                MCP cold start on every test (fairer comparison)
        """
        import uuid

        if test_cases is None:
            test_cases = self.load_test_cases()

        if configs is None:
            configs = self.get_default_configs()

        results: dict[str, list[BenchmarkResult]] = {
            configs[0].name: [],
            configs[1].name: [],
        }

        # create run directory
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = self.output_dir / f"run-{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # generate session IDs for each config (for session resumption)
        session_ids: dict[str, str] = {}
        if use_session_resumption:
            for config in configs:
                session_ids[config.name] = str(uuid.uuid4())

        print(f"🚀 Starting benchmark run: {run_id}")
        print(f"   Project: {self.project_path}")
        print(f"   Test cases: {len(test_cases)}")
        print(f"   Configs: {configs[0].name}, {configs[1].name}")
        if use_session_resumption:
            print("   Session resumption: enabled (MCP stays warm)")
        print()

        # warmup phase if using session resumption
        if use_session_resumption:
            print("⏳ Warming up sessions (one-time MCP cold start)...")
            for config in configs:
                print(f"  → {config.name}...", end=" ", flush=True)
                warmup_case = TestCase(
                    id="warmup",
                    name="Warmup",
                    prompt="Say 'ready' in one word.",
                    category="warmup",
                )
                warmup_result = await self.run_single(
                    warmup_case,
                    config,
                    session_id=session_ids[config.name],
                    resume=False,  # first run creates session
                )
                status = "✓" if warmup_result.success else "✗"
                print(f"{status} ({warmup_result.elapsed_seconds:.1f}s)")
            print()

        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] {test_case.name}")

            for config in configs:
                print(f"  → {config.name}...", end=" ", flush=True)

                # use session resumption if enabled
                session_id = session_ids.get(config.name)
                result = await self.run_single(
                    test_case,
                    config,
                    session_id=session_id,
                    resume=use_session_resumption,  # resume after warmup
                )
                results[config.name].append(result)

                status = "✓" if result.success else "✗"
                print(f"{status} ({result.elapsed_seconds:.1f}s)")

                # save individual result
                result_file = run_dir / config.name / f"{test_case.id}.json"
                result_file.parent.mkdir(parents=True, exist_ok=True)
                with open(result_file, "w") as f:
                    json.dump(result.to_dict(), f, indent=2)

            print()

        # save summary
        summary = {
            "run_id": run_id,
            "project_path": str(self.project_path),
            "test_cases": len(test_cases),
            "results": {
                name: [r.to_dict() for r in run_results]
                for name, run_results in results.items()
            },
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"✅ Results saved to: {run_dir}")
        return results

    def run_sync(
        self,
        test_cases: list[TestCase] | None = None,
        configs: tuple[RunConfig, RunConfig] | None = None,
    ) -> dict[str, list[BenchmarkResult]]:
        """Synchronous wrapper for run_comparison."""
        return asyncio.run(self.run_comparison(test_cases, configs))


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark Claude with/without ultrasync"
    )
    parser.add_argument(
        "project_path",
        type=Path,
        help="Path to the project to benchmark against",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        help="Path to test cases directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for results",
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Only run cases matching this pattern",
    )
    args = parser.parse_args()

    harness = BenchmarkHarness(
        project_path=args.project_path,
        output_dir=args.output,
    )

    test_cases = harness.load_test_cases(args.cases)

    if args.filter:
        test_cases = [
            c
            for c in test_cases
            if args.filter.lower() in c.id.lower()
            or args.filter.lower() in c.name.lower()
        ]

    if not test_cases:
        print("No test cases found!")
        return

    await harness.run_comparison(test_cases)


if __name__ == "__main__":
    asyncio.run(main())
