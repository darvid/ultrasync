#!/usr/bin/env python3
"""CLI for running ultrasync benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def cmd_run(args: argparse.Namespace) -> int:
    """Run benchmarks."""
    from benchmarks.harness import BenchmarkHarness
    from benchmarks.reporter import ComparisonReporter

    harness = BenchmarkHarness(
        project_path=args.project,
        output_dir=args.output,
    )

    test_cases = harness.load_test_cases(args.cases)

    # filter if requested
    if args.filter:
        test_cases = [
            c
            for c in test_cases
            if args.filter.lower() in c.id.lower()
            or args.filter.lower() in c.name.lower()
            or args.filter.lower() in c.category.lower()
        ]

    if args.category:
        test_cases = [c for c in test_cases if c.category == args.category]

    if args.difficulty:
        test_cases = [c for c in test_cases if c.difficulty == args.difficulty]

    if not test_cases:
        print("❌ No test cases found matching filters!")
        return 1

    print(f"📋 Found {len(test_cases)} test cases")

    # run benchmarks
    results = asyncio.run(harness.run_comparison(test_cases))

    # generate report
    reporter = ComparisonReporter()
    report = reporter.generate_report(results)

    # save reports
    output_dir = args.output or (Path(__file__).parent / "results")
    saved = reporter.save_report(report, output_dir)

    print()
    print("📊 Reports generated:")
    for path in saved:
        print(f"   {path}")

    # print quick summary
    print()
    print("=" * 60)
    print(report.to_markdown().split("## Key Findings")[0])

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Generate report from existing results."""
    from benchmarks.reporter import ComparisonReporter, load_results_from_dir

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        return 1

    results = load_results_from_dir(run_dir)

    reporter = ComparisonReporter()
    report = reporter.generate_report(results, run_id=run_dir.name)

    if args.format == "md":
        print(report.to_markdown())
    else:
        print(report.to_json())

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List available test cases."""
    from benchmarks.harness import BenchmarkHarness

    harness = BenchmarkHarness(project_path=Path.cwd())
    cases = harness.load_test_cases(args.cases)

    # group by category
    by_category: dict[str, list] = {}
    for case in cases:
        by_category.setdefault(case.category, []).append(case)

    print(f"📋 {len(cases)} test cases available:\n")

    for category, cat_cases in sorted(by_category.items()):
        print(f"  [{category}]")
        for case in cat_cases:
            diff_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(
                case.difficulty, "⚪"
            )
            print(f"    {diff_emoji} {case.id}: {case.name}")
        print()

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark Claude with/without ultrasync MCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks against current project
  python -m benchmarks.cli run .

  # Run only discovery tests
  python -m benchmarks.cli run . --category discovery

  # Run a specific test
  python -m benchmarks.cli run . --filter find_mcp

  # List available tests
  python -m benchmarks.cli list

  # Generate report from previous run
  python -m benchmarks.cli report benchmarks/results/run-20250122-143000
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # run command
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument(
        "project",
        type=Path,
        help="Project directory to benchmark against",
    )
    run_parser.add_argument(
        "--cases",
        type=Path,
        help="Directory containing test case YAML files",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for results",
    )
    run_parser.add_argument(
        "--filter",
        type=str,
        help="Filter test cases by id/name pattern",
    )
    run_parser.add_argument(
        "--category",
        type=str,
        choices=["discovery", "understanding", "debugging", "modification"],
        help="Filter by category",
    )
    run_parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard"],
        help="Filter by difficulty",
    )
    run_parser.set_defaults(func=cmd_run)

    # report command
    report_parser = subparsers.add_parser(
        "report", help="Generate report from results"
    )
    report_parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to benchmark run directory",
    )
    report_parser.add_argument(
        "--format",
        choices=["md", "json"],
        default="md",
        help="Output format",
    )
    report_parser.set_defaults(func=cmd_report)

    # list command
    list_parser = subparsers.add_parser("list", help="List test cases")
    list_parser.add_argument(
        "--cases",
        type=Path,
        help="Directory containing test case YAML files",
    )
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
