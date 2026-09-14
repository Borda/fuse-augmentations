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

Usage::

    python .github/scripts/ci_perf_baseline_aggregate.py \\
        --dir history/ \\
        --out avg.txt \\
        --min-samples 3 \\
        --summary-file "$GITHUB_STEP_SUMMARY"

"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path


def main() -> None:
    """Read all ``<n>.txt`` scores in --dir, write the median to --out, exit 1 if too few."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, metavar="PATH", help="Directory of downloaded <n>.txt score files.")
    parser.add_argument("--out", required=True, metavar="PATH", help="Output file for the aggregated score.")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        metavar="N",
        help="Minimum number of historical samples required to trust the median (default: 3).",
    )
    parser.add_argument("--summary-file", metavar="PATH", help="Append a markdown table (e.g. $GITHUB_STEP_SUMMARY).")
    args = parser.parse_args()

    score_files = sorted(Path(args.dir).glob("*.txt"))
    samples: list[tuple[str, float]] = []
    for f in score_files:
        text = f.read_text().strip()
        if text:
            samples.append((f.stem, float(text)))

    if len(samples) < args.min_samples:
        print(
            f"ERROR: only {len(samples)}/{len(score_files)} historical benchmark jobs produced a score, "
            f"need at least {args.min_samples}. Check the benchmark-history job logs.",
            file=sys.stderr,
        )
        sys.exit(1)

    values = [v for _, v in samples]
    median_score = statistics.median(values)

    Path(args.out).write_text(f"{median_score:.4f}\n")

    print(f"samples={dict(samples)}")
    print(f"median_baseline_score={median_score:.4f}")

    if args.summary_file:
        ordered = sorted(samples, key=lambda item: item[0])
        lines = [
            "## Dynamic Perf Baseline (last N main commits)",
            "",
            "| Commits back (HEAD~n) | real_score |",
            "|---|---|",
            *[f"| {n} | `{v:.4f}` |" for n, v in ordered],
            "",
            f"**Median baseline**: `{median_score:.4f}` (from {len(samples)} samples)",
            "",
        ]
        with Path(args.summary_file).open("a") as fh:
            fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
