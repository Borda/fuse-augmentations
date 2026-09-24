r"""Aggregate a dynamic CI perf baseline from N historical benchmark samples.

Reads one float score per ``<n>.txt`` file in a directory (each written by a
``benchmark-history`` matrix job that ran ``experiments/optimize_score.py``
against a recent commit on ``main``), takes the median, and writes it to an
output file plus a GitHub job summary table. Median, not mean, so a single
flaky historical run can't swing the number the live PR gate is compared
against -- same reasoning already applied inside ``optimize_score.py``'s own
per-case timing.

No state is persisted anywhere: this recomputes the reference from actual git
history on every gate run, so it can never go stale the way a committed
baseline JSON can.

The CLI uses ``fire``, which ships in the ``cli`` extra::

    pip install "fuse-augmentations[cli]"

Usage::

    python .github/scripts/ci_perf_baseline_aggregate.py \\
        --score-dir history/ \\
        --out avg.txt \\
        --min-samples 3 \\
        --expected-samples 5 \\
        --summary-file "$GITHUB_STEP_SUMMARY"

"""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path


def _parse_score(text: str, source: str) -> float | None:
    """Return a usable score from ``text``, or ``None`` when it cannot be trusted.

    A single unparsable or non-finite sample must not kill the job: doing so would defeat both the
    ``min_samples`` tolerance below and the ``continue-on-error`` tolerance one job upstream. A
    non-positive value is rejected for the same reason a NaN is -- ``0.0`` reaches the gate's delta
    arithmetic and raises ``ZeroDivisionError`` there, misattributing a bad input as a gate crash.

    Args:
        text: Raw file contents, already stripped.
        source: Sample name, used in the warning.

    Returns:
        The parsed score, or ``None`` when it is unparsable, non-finite, or non-positive.

    Examples:
        ```pycon
        >>> _parse_score("1.75", "0")
        1.75
        >>> _parse_score("nan", "0") is None
        True

        ```

    """
    try:
        value = float(text)
    except ValueError:
        print(f"::warning::sample {source} is not a number ({text!r}) -- excluded", file=sys.stderr)
        return None
    if not math.isfinite(value) or value <= 0:
        print(f"::warning::sample {source} is not a positive finite score ({value}) -- excluded", file=sys.stderr)
        return None
    return value


def main(
    score_dir: str,
    out: str,
    min_samples: int = 3,
    expected_samples: int | None = None,
    summary_file: str | None = None,
) -> None:
    """Read all ``<n>.txt`` scores in ``score_dir``, write the median to ``out``, exit 1 if too few.

    Args:
        score_dir: Directory of downloaded ``<n>.txt`` score files.
        out: Output file for the aggregated score.
        min_samples: Minimum number of historical samples required to trust the median.
        expected_samples: How many benchmark legs the workflow dispatched. Passed explicitly rather
            than inferred from the downloaded files: a leg killed by the job timeout uploads no
            artifact at all, so counting files would shrink both sides of the comparison equally and
            never report the degradation.
        summary_file: Append a markdown table here (pass ``$GITHUB_STEP_SUMMARY`` in CI).

    Examples:
        ```pycon
        >>> main(score_dir="history", out="avg.txt")  # doctest: +SKIP

        ```

    """
    score_files = sorted(Path(score_dir).glob("*.txt"))
    samples: list[tuple[str, float]] = []
    for f in score_files:
        text = f.read_text().strip()
        if text:
            value = _parse_score(text, f.stem)
            if value is not None:
                samples.append((f.stem, value))

    if expected_samples is not None and len(samples) < expected_samples:
        print(
            f"::warning::degraded baseline: {len(samples)}/{expected_samples} benchmark legs produced a usable "
            f"score. The median is built from fewer, older commits than intended -- check the benchmark-history "
            f"job logs for timed-out or failed legs.",
            file=sys.stderr,
        )

    if len(samples) < min_samples:
        print(
            f"ERROR: only {len(samples)}/{len(score_files)} historical benchmark jobs produced a score, "
            f"need at least {min_samples}. Check the benchmark-history job logs.",
            file=sys.stderr,
        )
        sys.exit(1)

    values = [v for _, v in samples]
    median_score = statistics.median(values)

    Path(out).write_text(f"{median_score:.4f}\n")

    print(f"samples={dict(samples)}")
    print(f"median_baseline_score={median_score:.4f}")

    if summary_file:
        ordered = sorted(samples, key=lambda item: item[0])
        lines = [
            "## Dynamic Perf Baseline (last N main commits)",
            "",
            "| Commits back (HEAD~n) | real_score |",
            "|---|---|",
            *[f"| {n} | `{v:.4f}` |" for n, v in ordered],
            "",
            f"**Median baseline**: `{median_score:.4f}` (from {len(samples)}"
            + (f"/{expected_samples}" if expected_samples is not None else "")
            + " samples)",
            "",
        ]
        with Path(summary_file).open("a") as fh:
            fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
