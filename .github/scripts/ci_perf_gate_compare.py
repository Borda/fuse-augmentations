r"""CI performance gate comparison script for fuse-augmentations.

Compares a current benchmark score JSON against a committed baseline JSON.

Bootstrap protocol (no baseline present):
    Gate passes neutrally; prints instructions to commit the artifact as baseline.

Gate protocol (baseline present):
    Fails with exit code 1 when real_score < baseline_score * threshold.

Usage::

    # Bootstrap run (baseline not yet committed):
    python tasks/ci_perf_gate_compare.py \\
        --current ci_score.json \\
        --baseline .github/perf_baseline/ci_baseline_score.json

    # Gated comparison with job summary:
    python tasks/ci_perf_gate_compare.py \\
        --current ci_score.json \\
        --baseline .github/perf_baseline/ci_baseline_score.json \\
        --threshold 0.95 \\
        --summary-file "$GITHUB_STEP_SUMMARY"

    # Dry-check with fake data:
    echo '{"real_score": 1.75, "theoretical_target": 2.375}' > /tmp/fake_current.json
    echo '{"real_score": 1.70, "theoretical_target": 2.375}' > /tmp/fake_baseline.json
    python tasks/ci_perf_gate_compare.py \\
        --current /tmp/fake_current.json \\
        --baseline /tmp/fake_baseline.json \\
        --threshold 0.95

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_json(path: str) -> dict | None:
    """Load a JSON file; return None if path does not exist."""
    p = Path(path)
    if not p.exists():
        return None
    with p.open() as fh:
        return json.load(fh)  # type: ignore[return-value]


def _format_summary(
    current: dict,
    baseline: dict | None,
    threshold: float,
    passed: bool,
) -> str:
    """Render a GitHub job summary markdown table."""
    real_score: float = current["real_score"]
    theoretical: object = current.get("theoretical_target", "N/A")

    if baseline is None:
        baseline_score_str = "N/A (no baseline yet)"
        delta_str = "N/A"
        status = "⚪ NEUTRAL — no baseline; first run records score only"
    else:
        b_score: float = baseline["real_score"]
        delta: float = real_score - b_score
        delta_pct: float = (real_score / b_score - 1.0) * 100.0
        baseline_score_str = f"{b_score:.4f}"
        delta_str = f"{delta:+.4f} ({delta_pct:+.1f}%)"
        status = "✅ PASSED" if passed else "❌ FAILED — regression exceeds threshold"

    lines = [
        "## Perf Regression Gate",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Real Score | `{real_score:.4f}` |",
        f"| Theoretical Target | `{theoretical}` |",
        f"| Baseline Score | `{baseline_score_str}` |",
        f"| Delta vs Baseline | `{delta_str}` |",
        f"| Regression Threshold | `{threshold:.0%}` |",
        f"| Status | {status} |",
    ]

    if baseline is None:
        lines += [
            "",
            "> **Bootstrap mode**: No baseline file found.",
            "> Download this run's `ci_score.json` artifact, then bootstrap the gate:",
            "> ```bash",
            "> cp ci_score.json .github/perf_baseline/ci_baseline_score.json",
            "> git add .github/perf_baseline/ci_baseline_score.json",
            "> git commit -m 'chore: bootstrap CI perf baseline'",
            "> ```",
        ]
    elif not passed:
        min_score: float = baseline["real_score"] * threshold
        lines += [
            "",
            f"> **Failure**: `real_score={real_score:.4f}` is below the minimum "
            f"`{min_score:.4f}` (= baseline `{baseline['real_score']:.4f}` x `{threshold}`). "
            "Investigate the regression in `src/` or `experiments/` before merging.",
        ]

    return "\n".join(lines) + "\n"


def main() -> None:
    """Parse arguments, evaluate gate, optionally write summary, exit with result."""
    parser = argparse.ArgumentParser(
        description="Performance regression gate for fuse-augmentations CI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--current",
        required=True,
        metavar="PATH",
        help="Current score JSON (ci_score.json produced by the benchmark step).",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        metavar="PATH",
        help="Committed baseline JSON (.github/perf_baseline/ci_baseline_score.json).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        metavar="RATIO",
        help="Minimum allowed ratio current/baseline (default: 0.95 = 5%% regression allowed).",
    )
    parser.add_argument(
        "--summary-file",
        metavar="PATH",
        help="Append a markdown summary table to this file (pass $GITHUB_STEP_SUMMARY in CI).",
    )
    args = parser.parse_args()

    current = _load_json(args.current)
    if current is None:
        print(f"ERROR: current score file not found: {args.current}", file=sys.stderr)
        sys.exit(2)

    baseline = _load_json(args.baseline)
    real_score: float = current["real_score"]

    # Evaluate gate result before writing summary (summary needs `passed`).
    if baseline is None:
        passed = True
        gate_result = "NEUTRAL"
        min_score_val = None
    else:
        b_score: float = baseline["real_score"]
        min_score_val = b_score * args.threshold
        passed = real_score >= min_score_val
        gate_result = "PASSED" if passed else "FAILED"

    # Always write summary before any exit — ensures it appears even on gate failure.
    if args.summary_file:
        summary = _format_summary(current, baseline, args.threshold, passed)
        with Path(args.summary_file).open("a") as fh:
            fh.write(summary)

    # Print human-readable results to stdout.
    print(f"real_score={real_score:.4f}")
    if baseline is not None and min_score_val is not None:
        b: float = baseline["real_score"]
        print(f"baseline_score={b:.4f}")
        print(f"min_allowed_score={min_score_val:.4f}  (baseline x {args.threshold})")
        print(f"delta={real_score - b:+.4f}  ({(real_score / b - 1.0) * 100.0:+.1f}%)")
    else:
        print(f"baseline_score=N/A  (no file at {args.baseline})")
        print("ACTION: commit downloaded artifact as .github/perf_baseline/ci_baseline_score.json")

    print(f"GATE: {gate_result}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
