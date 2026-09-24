r"""CI performance gate comparison script for fuse-augmentations.

Compares a current benchmark score against a dynamically computed baseline
(the median of the last N `main` commits -- see ci_perf_baseline_aggregate.py).
Fails with exit code 1 when real_score < baseline_score * threshold.

There is no bootstrap/no-baseline case: the baseline is always freshly
computed from actual git history by the workflow's benchmark-history +
aggregate-baseline jobs before this script runs, so it can never be missing
or stale the way a committed baseline JSON could be.

The CLI uses ``fire``, which ships in the ``cli`` extra::

    pip install "fuse-augmentations[cli]"

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

import json
import math
import sys
from pathlib import Path


def _load_json(path: str) -> dict:
    """Load a JSON file, exiting 2 when it is missing or malformed.

    Exit 2, not 1: callers distinguish "the gate failed" (1) from "the script could not run" (2), and collapsing the two
    makes a missing input look like a performance regression.

    """
    try:
        with Path(path).open() as fh:
            return json.load(fh)  # type: ignore[no-any-return]
    except FileNotFoundError:
        print(f"ERROR: current score file not found: {path}", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as exc:
        print(f"ERROR: current score file is not valid JSON: {path} ({exc})", file=sys.stderr)
        sys.exit(2)


def _format_summary(
    current: dict,
    baseline_score: float,
    threshold: float,
    passed: bool,
    efficiency: float | None = None,
    min_efficiency: float | None = None,
) -> str:
    """Render the gate summary with the actual failed checks named separately."""
    real_score: float = current["real_score"]
    theoretical: object = current.get("theoretical_target", "N/A")
    delta: float = real_score - baseline_score
    delta_pct: float = (real_score / baseline_score - 1.0) * 100.0
    status = "✅ PASSED" if passed else "❌ FAILED"

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
    ]

    if efficiency is not None and min_efficiency is not None:
        floor_status = "✅" if efficiency >= min_efficiency else "❌"
        lines.append(
            f"| Efficiency vs Target (absolute floor) | `{efficiency:.3f}` >= `{min_efficiency:.3f}` {floor_status} |"
        )

    lines.append(f"| Status | {status} |")

    if not passed:
        lines.append("")
        min_score: float = baseline_score * threshold
        if real_score < min_score:
            lines.append(
                f"> **Failure**: rolling ratio — `real_score={real_score:.4f}` is below "
                f"`{min_score:.4f}` (= dynamic baseline `{baseline_score:.4f}` x `{threshold}`)."
            )
        if efficiency is not None and min_efficiency is not None and efficiency < min_efficiency:
            lines.append(
                f"> **Failure**: absolute efficiency floor — `{efficiency:.3f}` is below `{min_efficiency:.3f}`."
            )
        lines.append("Investigate the regression in `src/` or `experiments/` before merging.")

    return "\n".join(lines) + "\n"


def main(
    current: str,
    baseline_score: float,
    threshold: float = 0.90,
    min_efficiency: float | None = None,
    summary_file: str | None = None,
) -> None:
    """Evaluate the perf gate, optionally write a job summary, exit 1 on regression.

    Two independent checks, both of which must pass. The rolling ``threshold`` check is relative and
    catches a sudden step regression. The ``min_efficiency`` check is absolute and catches slow drift
    that the rolling baseline cannot see at all: because the baseline is recomputed from recent
    ``main`` every run, the reference follows the code, so a sequence of individually-tolerated
    regressions ratchets the bar down indefinitely with every run green.

    Args:
        current: Current score JSON (``ci_score.json`` produced by the benchmark step).
        baseline_score: Dynamically computed baseline real_score (output of ci_perf_baseline_aggregate.py).
        threshold: Minimum allowed ratio current/baseline (0.90 = 10% regression allowed).
        min_efficiency: Absolute floor on ``real_score / theoretical_target``. ``theoretical_target``
            is derived from operation counts, not wall time, so it is hardware-independent and a
            usable anchor. Note it shifts if the benchmark case bank changes, which silently
            re-baselines this floor -- revisit the value when cases are added or removed.
        summary_file: Append a markdown summary table here (pass ``$GITHUB_STEP_SUMMARY`` in CI).

    Examples:
        ```pycon
        >>> main(current="ci_score.json", baseline_score=1.61, threshold=0.90)  # doctest: +SKIP

        ```

    """
    # fire infers the type from the literal, so an empty $BASELINE_SCORE arrives as "" and would
    # raise an uncaught TypeError below. Coerce once, then check the domain fire cannot know about.
    try:
        baseline_score = float(baseline_score)
    except (TypeError, ValueError):
        print(f"ERROR: --baseline-score must be a number, got {baseline_score!r}", file=sys.stderr)
        sys.exit(2)
    if not math.isfinite(baseline_score) or baseline_score <= 0:
        print(f"ERROR: --baseline-score must be finite and positive, got {baseline_score}", file=sys.stderr)
        sys.exit(2)

    current_score = _load_json(current)
    try:
        real_score = float(current_score["real_score"])
    except (KeyError, TypeError, ValueError):
        print(f"ERROR: current score must contain a numeric real_score, got {current_score!r}", file=sys.stderr)
        sys.exit(2)
    if not math.isfinite(real_score) or real_score <= 0:
        print(f"ERROR: real_score must be finite and positive, got {real_score}", file=sys.stderr)
        sys.exit(2)

    min_score_val = baseline_score * threshold
    passed = real_score >= min_score_val
    efficiency: float | None = None

    if min_efficiency is not None:
        raw_target = current_score.get("theoretical_target")
        try:
            theoretical = float(raw_target)
        except (TypeError, ValueError):
            theoretical = float("nan")
        if not math.isfinite(theoretical) or theoretical <= 0:
            print(f"ERROR: --min-efficiency given but theoretical_target is unusable: {raw_target!r}", file=sys.stderr)
            sys.exit(2)
        efficiency = real_score / theoretical
        if efficiency < min_efficiency:
            passed = False

    gate_result = "PASSED" if passed else "FAILED"

    # Always write summary before any exit — ensures it appears even on gate failure.
    if summary_file:
        summary = _format_summary(current_score, baseline_score, threshold, passed, efficiency, min_efficiency)
        with Path(summary_file).open("a") as fh:
            fh.write(summary)

    print(f"real_score={real_score:.4f}")
    print(f"baseline_score={baseline_score:.4f}")
    print(f"min_allowed_score={min_score_val:.4f}  (baseline x {threshold})")
    print(f"delta={real_score - baseline_score:+.4f}  ({(real_score / baseline_score - 1.0) * 100.0:+.1f}%)")
    if efficiency is not None:
        print(f"efficiency={efficiency:.4f}  (min_efficiency {min_efficiency})")
    print(f"GATE: {gate_result}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
