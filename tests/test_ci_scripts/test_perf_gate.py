"""Subprocess tests for the two CI perf-gate scripts and the bench-harness statistic they consume.

``.github/scripts/`` is excluded from pytest collection (see ``norecursedirs`` in ``pyproject.toml``),
so these scripts have no automated coverage even though a non-zero exit on either one blocks a PR merge.
This module invokes them as real subprocesses (``sys.executable`` + a repo-root-relative path -- never
a hardcoded absolute path, so the suite is portable across CI runners and local checkouts).

Covers, per the PR #18 review (finding F15):

1. ``ci_perf_baseline_aggregate.py`` exits 1 when usable samples fall below ``--min-samples``.
2. ``ci_perf_baseline_aggregate.py`` computes the correct median and excludes a blank sample file
   rather than parsing it as ``0.0``.
3. ``ci_perf_gate_compare.py``'s PASS/FAIL boundary at ``real_score == baseline_score * threshold``.
4. ``ci_perf_gate_compare.py`` rejects a non-finite ``--baseline-score`` (``nan``/``inf``) with the
   script-error exit code, not a silent gate decision.
5. ``experiments/optimize_score.py``'s ``_bench`` returns the median of per-batch averages, not a
   plain mean or a per-call median -- a silent revert to the old statistic must fail this test.

Both scripts use ``fire`` for their CLI (imported lazily under ``if __name__ == "__main__":``), which
ships in the ``cli`` extra. Every subprocess-invoking test is skipped, not hard-failed, when ``fire``
is not importable in this environment.

"""

from __future__ import annotations

import importlib.util
import json
import statistics
import subprocess
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AGGREGATE_SCRIPT = _REPO_ROOT / ".github" / "scripts" / "ci_perf_baseline_aggregate.py"
_GATE_SCRIPT = _REPO_ROOT / ".github" / "scripts" / "ci_perf_gate_compare.py"
_OPTIMIZE_SCORE_PATH = _REPO_ROOT / "experiments" / "optimize_score.py"

try:
    import fire  # noqa: F401

    _FIRE_AVAILABLE = True
except ImportError:
    _FIRE_AVAILABLE = False

_skip_no_fire = pytest.mark.skipif(
    not _FIRE_AVAILABLE, reason="fire (the `cli` extra) is not importable in this environment"
)


def _run_script(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run ``script`` as a subprocess with the current interpreter and return the completed process."""
    return subprocess.run(  # noqa: S603 - the interpreter and script path are project-controlled
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _load_optimize_score() -> types.ModuleType:
    """Import ``experiments/optimize_score.py`` by path -- ``experiments/`` has no ``__init__.py``."""
    spec = importlib.util.spec_from_file_location("optimize_score", _OPTIMIZE_SCORE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    _optimize_score: types.ModuleType | None = _load_optimize_score()
    _OPTIMIZE_SCORE_IMPORT_ERROR: str | None = None
except ImportError as exc:
    # Pre-existing local environment issue (e.g. a scipy binary incompatibility pulled in via kornia /
    # albumentations), not something a fix to the perf-gate scripts should be blocked on. Meaningful
    # in CI where the full dependency set imports cleanly; skipped here rather than erroring.
    _optimize_score = None
    _OPTIMIZE_SCORE_IMPORT_ERROR = str(exc)

_skip_no_optimize_score = pytest.mark.skipif(
    _optimize_score is None,
    reason=f"experiments/optimize_score.py did not import: {_OPTIMIZE_SCORE_IMPORT_ERROR}",
)


@_skip_no_fire
class TestAggregateMinSamples:
    """``ci_perf_baseline_aggregate.py`` exits 1 when too few usable samples are present."""

    def test_exits_1_below_min_samples(self, tmp_path: Path):
        """Fewer usable samples than ``--min-samples`` fails the job instead of aggregating a weak baseline.

        Only one score file is provided against ``--min-samples 3``. Silently proceeding here would let a near-empty
        history window set the PR gate's baseline.

        """
        (tmp_path / "1.txt").write_text("2.5")
        out = tmp_path / "avg.txt"

        result = _run_script(
            _AGGREGATE_SCRIPT,
            "--score-dir",
            str(tmp_path),
            "--out",
            str(out),
            "--min-samples",
            "3",
        )

        assert result.returncode == 1
        assert "need at least 3" in result.stderr
        assert not out.exists()


@_skip_no_fire
class TestAggregateMedian:
    """``ci_perf_baseline_aggregate.py`` computes the correct median and excludes blank samples."""

    def test_median_excludes_blank_sample(self, tmp_path: Path):
        """A blank ``<n>.txt`` is dropped, not parsed as ``0.0``, and the median is of the usable samples.

        Files: ``1.txt=1.0``, ``2.txt=2.0``, ``3.txt=3.0``, ``4.txt=`` (blank). If the blank file were
        counted as ``0.0`` the median over four samples would be ``1.5``; excluding it correctly gives the
        median of ``[1.0, 2.0, 3.0]``, which is ``2.0``.

        """
        (tmp_path / "1.txt").write_text("1.0")
        (tmp_path / "2.txt").write_text("2.0")
        (tmp_path / "3.txt").write_text("3.0")
        (tmp_path / "4.txt").write_text("")
        out = tmp_path / "avg.txt"

        result = _run_script(
            _AGGREGATE_SCRIPT,
            "--score-dir",
            str(tmp_path),
            "--out",
            str(out),
            "--min-samples",
            "1",
        )

        assert result.returncode == 0
        assert out.read_text().strip() == "2.0000"
        assert "median_baseline_score=2.0000" in result.stdout


@_skip_no_fire
class TestGateCompareThresholdBoundary:
    """``ci_perf_gate_compare.py``'s PASS/FAIL boundary at ``real_score == baseline_score * threshold``."""

    @pytest.mark.parametrize(
        ("real_score", "expected_exit"),
        [
            pytest.param(3.0, 0, id="at-boundary-passes"),
            pytest.param(2.999, 1, id="just-below-boundary-fails"),
        ],
    )
    def test_boundary_exit_code(self, tmp_path: Path, real_score: float, expected_exit: int):
        """``real_score`` exactly at ``baseline_score * threshold`` passes; just under it fails.

        ``baseline_score=4.0`` and ``threshold=0.75`` are both exactly representable in binary
        floating point, so ``min_score = 3.0`` holds exactly and the boundary case is not a
        floating-point rounding artifact.

        """
        current = tmp_path / "current.json"
        current.write_text(json.dumps({"real_score": real_score}))

        result = _run_script(
            _GATE_SCRIPT,
            "--current",
            str(current),
            "--baseline-score",
            "4.0",
            "--threshold",
            "0.75",
        )

        assert result.returncode == expected_exit


@_skip_no_fire
class TestGateCompareNonFiniteBaseline:
    """``ci_perf_gate_compare.py`` rejects a non-finite ``--baseline-score`` as a script error."""

    @pytest.mark.parametrize("baseline_score", ["nan", "inf"], ids=["nan", "inf"])
    def test_non_finite_baseline_exits_2(self, tmp_path: Path, baseline_score: str):
        """A ``nan`` or ``inf`` ``--baseline-score`` exits 2 (script/input error), never a silent gate result.

        ``real_score >= baseline_score * threshold`` is well-defined-looking but meaningless once ``baseline_score`` is
        non-finite -- every comparison against ``nan`` is ``False`` and every comparison against ``inf`` trivially
        fails, so an unguarded gate would either always fail or misreport a crash as a regression. This must be caught
        explicitly before that arithmetic runs.

        """
        current = tmp_path / "current.json"
        current.write_text(json.dumps({"real_score": 1.5}))

        result = _run_script(
            _GATE_SCRIPT,
            "--current",
            str(current),
            "--baseline-score",
            baseline_score,
        )

        assert result.returncode == 2
        assert "finite and positive" in result.stderr


@_skip_no_optimize_score
class TestBenchStatistic:
    """``_bench`` returns the median of per-batch averages, not a plain mean or a per-call median."""

    def test_bench_returns_median_of_batch_averages(self, monkeypatch: pytest.MonkeyPatch):
        """``_bench``'s return value matches ``median(batch_means)``, distinct from ``mean(batch_means)``.

        ``time.perf_counter`` is replaced (via the module's own ``time`` name, not the global stdlib module) with a
        deterministic sequence: most batches "take" 1ms, a fifth of them "take" 100ms -- an outlier pattern designed so
        the median and the mean of the batch averages disagree sharply. A silent regression from ``statistics.median``
        to ``statistics.mean`` in ``_bench`` (or a revert to timing individual calls instead of batches) changes the
        returned value and fails this test.

        """
        module = _optimize_score
        n_batches = module.NUM_BATCHES
        batch_size = module.BATCH_SIZE
        n_slow = max(1, n_batches // 5)
        elapsed_seconds = [0.001] * (n_batches - n_slow) + [0.1] * n_slow

        perf_values: list[float] = []
        for elapsed in elapsed_seconds:
            perf_values.extend([0.0, elapsed])
        counter = iter(perf_values)
        fake_time = types.SimpleNamespace(perf_counter=lambda: next(counter))
        monkeypatch.setattr(module, "time", fake_time)

        result = module._bench(lambda _tensor: None)

        expected_batch_means_ms = [elapsed * 1000.0 / batch_size for elapsed in elapsed_seconds]
        assert result == pytest.approx(statistics.median(expected_batch_means_ms))
        assert result != pytest.approx(statistics.mean(expected_batch_means_ms))
