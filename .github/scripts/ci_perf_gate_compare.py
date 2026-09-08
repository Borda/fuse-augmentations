r"""CI performance gate comparison script for fuse-augmentations.

Compares a current benchmark score against a dynamically computed baseline
(the median of the last N `main` commits -- see ci_perf_baseline_aggregate.py).
Fails with exit code 1 when real_score < baseline_score * threshold.

There is no bootstrap/no-baseline case: the baseline is always freshly
computed from actual git history by the workflow's benchmark-history +
aggregate-baseline jobs before this script runs, so it can never be missing
or stale the way a committed baseline JSON could be.

Usage::

    python .github/scripts/ci_perf_gate_compare.py \\
        --current ci_score.json \\
        --baseline-score 1.6100 \\
        --threshold 0.90 \\
        --summary-file "$GITHUB_STEP_SUMMARY"

    # Dry-check with fake data:
    echo '{"real_score": 1.75, "theoretical_target": 2.375}' > /tmp/fake_current.json
    python .github/scripts/ci_perf_gate_compare.py \\
        --current /tmp/fake_current.json \\
        --baseline-score 1.70 \\
        --threshold 0.95

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_json(path: str) -> dict:
    """Load a JSON file."""
    with Path(path).open() as fh:
        return json.load(fh)  # type: ignore[no-any-return]


def _format_summary(current: dict, baseline_score: float, threshold: float, passed: bool) -> str:
    """Render a GitHub job summary markdown table."""
    real_score: float = current["real_score"]
    theoretical: object = current.get("theoretical_target", "N/A")
    delta: float = real_score - baseline_score
    delta_pct: float = (real_score / baseline_score - 1.0) * 100.0
    status = "✅ PASSED" if passed else "❌ FAILED — regression exceeds threshold"

    lines = [
        "## Perf Regression Gate",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Real Score | `{real_score:.4f}` |",
        f"| Theoretical Target | `{theoretical}` |",
        f"| Dynamic Baseline (median of last N main commits) | `{baseline_score:.4f}` |",
        f"| Delta vs Baseline | `{delta:+.4f} ({delta_pct:+.1f}%)` |",
        f"| Regression Threshold | `{threshold:.0%}` |",
        f"| Status | {status} |",
    ]

    if not passed:
        min_score: float = baseline_score * threshold
        lines += [
            "",
            (
                f"> **Failure**: `real_score={real_score:.4f}` is below the minimum "
                f"`{min_score:.4f}` (= dynamic baseline `{baseline_score:.4f}` x `{threshold}`). "
                "Investigate the regression in `src/` or `experiments/` before merging."
            ),
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
        "--baseline-score",
        required=True,
        type=float,
        metavar="SCORE",
        help="Dynamically computed baseline real_score (output of ci_perf_baseline_aggregate.py).",
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
    real_score: float = current["real_score"]

    min_score_val = args.baseline_score * args.threshold
    passed = real_score >= min_score_val
    gate_result = "PASSED" if passed else "FAILED"

    # Always write summary before any exit — ensures it appears even on gate failure.
    if args.summary_file:
        summary = _format_summary(current, args.baseline_score, args.threshold, passed)
        with Path(args.summary_file).open("a") as fh:
            fh.write(summary)

    print(f"real_score={real_score:.4f}")
    print(f"baseline_score={args.baseline_score:.4f}")
    print(f"min_allowed_score={min_score_val:.4f}  (baseline x {args.threshold})")
    print(f"delta={real_score - args.baseline_score:+.4f}  ({(real_score / args.baseline_score - 1.0) * 100.0:+.1f}%)")
    print(f"GATE: {gate_result}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
