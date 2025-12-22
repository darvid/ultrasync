"""Generate comparison reports from benchmark results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.harness import BenchmarkResult
from benchmarks.metrics import ToolCallMetrics, compare_metrics


@dataclass
class ComparisonReport:
    """A comparison report between two benchmark runs."""

    run_id: str
    generated_at: datetime
    test_case_count: int

    # per-config aggregates
    with_ultrasync_summary: dict[str, Any]
    baseline_summary: dict[str, Any]

    # comparison metrics
    comparison: dict[str, Any]

    # per-test-case breakdown
    test_cases: list[dict[str, Any]]

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Benchmark Report: {self.run_id}",
            "",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Test Cases: {self.test_case_count}",
            "",
            "## Summary",
            "",
            "| Metric | With Ultrasync | Baseline | Δ |",
            "|--------|----------------|----------|---|",
        ]

        # summary table
        us = self.with_ultrasync_summary
        bl = self.baseline_summary

        def delta(
            us_val: float, bl_val: float, lower_better: bool = True
        ) -> str:
            if bl_val == 0:
                return "N/A"
            pct = ((us_val - bl_val) / bl_val) * 100
            arrow = "↓" if (pct < 0) == lower_better else "↑"
            return f"{arrow} {abs(pct):.0f}%"

        # helper to shorten access
        us_tc = us.get("avg_tool_calls", 0)
        bl_tc = bl.get("avg_tool_calls", 0)
        us_fa = us.get("avg_files_accessed", 0)
        bl_fa = bl.get("avg_files_accessed", 0)
        us_tm = us.get("avg_time", 0)
        bl_tm = bl.get("avg_time", 0)
        us_sr = us.get("success_rate", 0)
        bl_sr = bl.get("success_rate", 0)

        lines.extend(
            [
                f"| Avg Tool Calls | {us_tc:.1f} | {bl_tc:.1f} | "
                f"{delta(us_tc, bl_tc)} |",
                f"| Avg Files | {us_fa:.1f} | {bl_fa:.1f} | "
                f"{delta(us_fa, bl_fa, False)} |",
                f"| Avg Time (s) | {us_tm:.1f} | {bl_tm:.1f} | "
                f"{delta(us_tm, bl_tm)} |",
                f"| Success Rate | {us_sr:.0%} | {bl_sr:.0%} | "
                f"{delta(1 - us_sr, 1 - bl_sr)} |",
            ]
        )

        lines.extend(
            [
                "",
                "## Key Findings",
                "",
            ]
        )

        # highlight key differences
        tool_reduction = self.comparison.get("tool_call_reduction", 0)
        if tool_reduction > 0:
            lines.append(
                f"- **{tool_reduction:.0f}% fewer tool calls** with ultrasync"
            )
        elif tool_reduction < 0:
            pct = abs(tool_reduction)
            lines.append(f"- **{pct:.0f}% more tool calls** with ultrasync")

        lines.extend(
            [
                "",
                "## Tool Usage Breakdown",
                "",
                "### With Ultrasync",
                "```",
            ]
        )

        for tool, count in sorted(
            us.get("tool_breakdown", {}).items(),
            key=lambda x: -x[1],
        ):
            lines.append(f"  {tool}: {count}")

        lines.extend(
            [
                "```",
                "",
                "### Baseline",
                "```",
            ]
        )

        for tool, count in sorted(
            bl.get("tool_breakdown", {}).items(),
            key=lambda x: -x[1],
        ):
            lines.append(f"  {tool}: {count}")

        lines.extend(
            [
                "```",
                "",
                "## Per-Test Case Results",
                "",
                "| Test | Cat | US | BL | US# | BL# | Win |",
                "|------|-----|----|----|-----|-----|-----|",
            ]
        )

        for tc in self.test_cases:
            us_time = tc.get("with_ultrasync", {}).get("time", 0)
            bl_time = tc.get("baseline", {}).get("time", 0)
            us_tools = tc.get("with_ultrasync", {}).get("tool_calls", 0)
            bl_tools = tc.get("baseline", {}).get("tool_calls", 0)

            # determine winner
            us_score = 0
            if us_time < bl_time:
                us_score += 1
            if us_tools < bl_tools:
                us_score += 1

            if us_score >= 2:
                winner = "US"
            elif us_score == 0:
                winner = "BL"
            else:
                winner = "="

            name = tc["name"][:20]
            cat = tc["category"][:4]
            lines.append(
                f"| {name} | {cat} | {us_time:.1f} | {bl_time:.1f} | "
                f"{us_tools} | {bl_tools} | {winner} |"
            )

        lines.extend(
            [
                "",
                "---",
                "",
                "*US = With Ultrasync, BL = Baseline*",
            ]
        )

        return "\n".join(lines)

    def to_json(self) -> str:
        """Generate JSON report."""
        return json.dumps(
            {
                "run_id": self.run_id,
                "generated_at": self.generated_at.isoformat(),
                "test_case_count": self.test_case_count,
                "with_ultrasync": self.with_ultrasync_summary,
                "baseline": self.baseline_summary,
                "comparison": self.comparison,
                "test_cases": self.test_cases,
            },
            indent=2,
        )


class ComparisonReporter:
    """Generate comparison reports from benchmark results."""

    def generate_report(
        self,
        results: dict[str, list[BenchmarkResult]],
        run_id: str | None = None,
    ) -> ComparisonReport:
        """Generate a comparison report from results."""
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

        with_ultrasync = results.get("with-ultrasync", [])
        baseline = results.get("baseline", [])

        # compute summaries
        us_summary = self._compute_summary(with_ultrasync)
        bl_summary = self._compute_summary(baseline)

        # compute comparison
        us_metrics = [r.metrics for r in with_ultrasync if r.metrics]
        bl_metrics = [r.metrics for r in baseline if r.metrics]
        comparison = (
            compare_metrics(us_metrics, bl_metrics)
            if us_metrics and bl_metrics
            else {}
        )

        # per-test-case breakdown
        test_cases = self._build_test_case_breakdown(with_ultrasync, baseline)

        return ComparisonReport(
            run_id=run_id,
            generated_at=datetime.now(),
            test_case_count=len(with_ultrasync),
            with_ultrasync_summary=us_summary,
            baseline_summary=bl_summary,
            comparison=comparison.get("comparison", {}),
            test_cases=test_cases,
        )

    def _compute_summary(
        self,
        results: list[BenchmarkResult],
    ) -> dict[str, Any]:
        """Compute summary statistics for a set of results."""
        if not results:
            return {}

        times = [r.elapsed_seconds for r in results]
        successes = [r.success for r in results]
        tool_calls = [
            sum(r.metrics.tool_calls.values()) if r.metrics else 0
            for r in results
        ]
        files_accessed = [
            len(r.metrics.files_accessed) if r.metrics else 0 for r in results
        ]

        # aggregate tool breakdown
        tool_breakdown: dict[str, int] = {}
        for r in results:
            if r.metrics:
                for tool, count in r.metrics.tool_calls.items():
                    tool_breakdown[tool] = tool_breakdown.get(tool, 0) + count

        return {
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "success_rate": sum(successes) / len(successes),
            "avg_tool_calls": sum(tool_calls) / len(tool_calls),
            "avg_files_accessed": sum(files_accessed) / len(files_accessed),
            "tool_breakdown": tool_breakdown,
        }

    def _build_test_case_breakdown(
        self,
        with_ultrasync: list[BenchmarkResult],
        baseline: list[BenchmarkResult],
    ) -> list[dict[str, Any]]:
        """Build per-test-case breakdown."""
        # index baseline by test case id
        baseline_by_id = {r.test_case.id: r for r in baseline}

        breakdown = []
        for us_result in with_ultrasync:
            tc = us_result.test_case
            bl_result = baseline_by_id.get(tc.id)

            entry = {
                "id": tc.id,
                "name": tc.name,
                "category": tc.category,
                "difficulty": tc.difficulty,
                "with_ultrasync": {
                    "success": us_result.success,
                    "time": us_result.elapsed_seconds,
                    "tool_calls": (
                        sum(us_result.metrics.tool_calls.values())
                        if us_result.metrics
                        else 0
                    ),
                    "files_accessed": (
                        len(us_result.metrics.files_accessed)
                        if us_result.metrics
                        else 0
                    ),
                },
            }

            if bl_result:
                entry["baseline"] = {
                    "success": bl_result.success,
                    "time": bl_result.elapsed_seconds,
                    "tool_calls": (
                        sum(bl_result.metrics.tool_calls.values())
                        if bl_result.metrics
                        else 0
                    ),
                    "files_accessed": (
                        len(bl_result.metrics.files_accessed)
                        if bl_result.metrics
                        else 0
                    ),
                }

            breakdown.append(entry)

        return breakdown

    def save_report(
        self,
        report: ComparisonReport,
        output_dir: Path,
        formats: list[str] | None = None,
    ) -> list[Path]:
        """Save report to files."""
        if formats is None:
            formats = ["md", "json"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = []

        if "md" in formats:
            md_path = output_dir / f"report-{report.run_id}.md"
            md_path.write_text(report.to_markdown())
            saved.append(md_path)

        if "json" in formats:
            json_path = output_dir / f"report-{report.run_id}.json"
            json_path.write_text(report.to_json())
            saved.append(json_path)

        return saved


def load_results_from_dir(
    run_dir: Path,
) -> dict[str, list[BenchmarkResult]]:
    """Load results from a benchmark run directory."""
    from benchmarks.harness import BenchmarkResult, TestCase

    results: dict[str, list[BenchmarkResult]] = {}

    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No summary.json found in {run_dir}")

    with open(summary_path) as f:
        summary = json.load(f)

    # reconstruct results from summary
    for config_name, config_results in summary.get("results", {}).items():
        results[config_name] = []

        for r in config_results:
            # minimal reconstruction - metrics aren't fully preserved
            tc = TestCase(
                id=r["test_case_id"],
                name=r["test_case_name"],
                prompt="",  # not preserved
                category="",  # not preserved
            )

            result = BenchmarkResult(
                test_case=tc,
                config_name=config_name,
                success=r["success"],
                error=r.get("error"),
                elapsed_seconds=r["elapsed_seconds"],
                exit_code=r["exit_code"],
            )

            # reconstruct metrics if available
            if r.get("metrics"):
                from collections import Counter

                m = r["metrics"]
                result.metrics = ToolCallMetrics(
                    tool_calls=Counter(m.get("tool_calls", {})),
                    files_accessed=set(m.get("files_accessed", [])),
                    turn_count=m.get("turn_count", 0),
                    ultrasync_search_count=m.get("ultrasync_search_count", 0),
                    grep_count=m.get("grep_count", 0),
                    glob_count=m.get("glob_count", 0),
                    read_count=m.get("read_count", 0),
                )

            results[config_name].append(result)

    return results
