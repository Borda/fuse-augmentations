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
6. The optimization score's default three complete measurements emit their median on the
   unchanged two-line standard-output contract.

Both scripts use ``fire`` for their CLI (imported lazily under ``if __name__ == "__main__":``), which
ships in the ``cli`` extra. Every subprocess-invoking test is skipped, not hard-failed, when ``fire``
is not importable in this environment.

"""

from __future__ import annotations

import ast
import copy
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _benchmark_function(name: str, namespace: dict[str, object]):
    """Load one benchmark function while replacing unavailable binary dependencies at its boundary."""
    tree = ast.parse(_OPTIMIZE_SCORE_PATH.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(  # noqa: S102 - only a named function from this project-controlled benchmark is compiled
        compile(ast.Module(body=[function], type_ignores=[]), str(_OPTIMIZE_SCORE_PATH), "exec"), namespace
    )
    return namespace[name]


def _run_script(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run ``script`` as a subprocess with the current interpreter and return the completed process."""
    return subprocess.run(  # noqa: S603 - the interpreter and script path are project-controlled
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_collection_preserves_parent_thread_environment():
    """Importing the test module must not pin thread settings in the pytest process."""
    inherited = {
        "OMP_NUM_THREADS": "7",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "9",
        "NUMEXPR_NUM_THREADS": "10",
    }
    result = subprocess.run(  # noqa: S603 - the interpreter and test module path are project-controlled
        [
            sys.executable,
            "-c",
            (
                "import json, os, runpy, sys; "
                "runpy.run_path(sys.argv[1]); "
                "print(json.dumps({name: os.environ[name] for name in json.loads(sys.argv[2])}))"
            ),
            str(Path(__file__).resolve()),
            json.dumps(list(inherited)),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **inherited},
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == inherited


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

    @pytest.mark.parametrize(
        ("real_score", "floor", "expected_causes"),
        [
            pytest.param(2.5, 0.2, {"rolling ratio"}, id="ratio-only"),
            pytest.param(3.2, 0.4, {"absolute efficiency floor"}, id="floor-only"),
            pytest.param(2.5, 0.4, {"rolling ratio", "absolute efficiency floor"}, id="both"),
        ],
    )
    def test_failed_summary_names_only_failed_checks(
        self, tmp_path: Path, real_score: float, floor: float, expected_causes: set[str]
    ):
        """The summary names each failed threshold without blaming a passing one."""
        current = tmp_path / "current.json"
        current.write_text(json.dumps({"real_score": real_score, "theoretical_target": 10.0}))
        summary = tmp_path / "summary.md"

        result = _run_script(
            _GATE_SCRIPT,
            "--current",
            str(current),
            "--baseline-score",
            "4.0",
            "--threshold",
            "0.75",
            "--min-efficiency",
            str(floor),
            "--summary-file",
            str(summary),
        )

        assert result.returncode == 1
        text = summary.read_text(encoding="utf-8")
        causes = {cause for cause in ("rolling ratio", "absolute efficiency floor") if f"**Failure**: {cause}" in text}
        assert causes == expected_causes


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


class TestBenchStatistic:
    """``_bench`` returns the median of per-batch averages, not a plain mean or a per-call median."""

    def test_bench_returns_median_of_batch_averages(self):
        """``_bench``'s return value matches ``median(batch_means)``, distinct from ``mean(batch_means)``.

        The isolated child supplies deterministic batch times, leaving collection's process-wide OpenCV and PyTorch
        thread settings untouched.

        """
        code = """
import importlib.util
import json
import sys
import types

spec = importlib.util.spec_from_file_location("optimize_score", sys.argv[1])
module = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(module)
except ImportError as exc:
    print(f"IMPORT_ERROR: {exc}")
    sys.exit(3)
n_batches = module.NUM_BATCHES
batch_size = module.BATCH_SIZE
n_slow = max(1, n_batches // 5)
elapsed_seconds = [0.001] * (n_batches - n_slow) + [0.1] * n_slow
counter = iter(value for elapsed in elapsed_seconds for value in (0.0, elapsed))
module.time = types.SimpleNamespace(perf_counter=lambda: next(counter))
actual = module._bench(lambda _tensor: None)
print(json.dumps({"actual": actual, "n_batches": n_batches, "batch_size": batch_size}))
"""
        result = subprocess.run(  # noqa: S603 - the interpreter and benchmark path are project-controlled
            [sys.executable, "-c", code, str(_OPTIMIZE_SCORE_PATH)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 3 and result.stdout.startswith("IMPORT_ERROR:"):
            pytest.skip(result.stdout.strip())
        assert result.returncode == 0, result.stderr

        measured = json.loads(result.stdout.splitlines()[-1])
        n_slow = max(1, measured["n_batches"] // 5)
        elapsed_seconds = [0.001] * (measured["n_batches"] - n_slow) + [0.1] * n_slow
        batch_means_ms = [elapsed * 1000.0 / measured["batch_size"] for elapsed in elapsed_seconds]
        assert measured["actual"] == pytest.approx(statistics.median(batch_means_ms))
        assert measured["actual"] != pytest.approx(statistics.mean(batch_means_ms))


class TestScoreRepetitions:
    """Whole-score repetitions report one median without changing the CLI score lines."""

    def test_main_uses_default_three_complete_scores(self):
        """The default emits the median of three complete measurements, not the last one."""
        code = """
import contextlib
import importlib.util
import io
import json
import sys

spec = importlib.util.spec_from_file_location("optimize_score", sys.argv[1])
module = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(module)
except ImportError as exc:
    print(f"IMPORT_ERROR: {exc}")
    sys.exit(3)
samples = iter([(1.0, 2.0, []), (5.0, 2.0, []), (3.0, 2.0, [])])
calls = []
def measure():
    calls.append(1)
    return next(samples)
module._measure_once = measure
output = io.StringIO()
with contextlib.redirect_stdout(output):
    module.main()
print(json.dumps({"output": output.getvalue(), "calls": len(calls)}))
"""
        result = subprocess.run(  # noqa: S603 - the interpreter and benchmark path are project-controlled
            [sys.executable, "-c", code, str(_OPTIMIZE_SCORE_PATH)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 3 and result.stdout.startswith("IMPORT_ERROR:"):
            pytest.skip(result.stdout.strip())
        assert result.returncode == 0, result.stderr
        measured = json.loads(result.stdout)
        assert measured == {"output": "real_score=3.0000\ntheoretical_target=2.0000\n", "calls": 3}


def test_albumentations_benchmark_reseeds_native_and_fused_copies():
    """Each whole-score pass seeds both Albumentations copies independently of global NumPy state."""
    native_seeds: list[int | None] = []
    fused_seeds: list[int | None] = []

    class Transform:
        def __init__(self):
            self.seed: int | None = None

        def set_random_seed(self, seed: int):
            self.seed = seed

    class NativeCompose:
        def __init__(self, transforms, seed=None):
            native_seeds.append(seed)
            for transform in transforms:
                transform.set_random_seed(seed)

        def __call__(self, **_kwargs):
            return None

    class FusedCompose:
        def __init__(self, transforms, reorder=None):
            fused_seeds.extend(transform.seed for transform in transforms if isinstance(transform, Transform))

        def __call__(self, *_args, **_kwargs):
            return None

    class TensorNative:
        def __call__(self, *_args):
            return None

    namespace = {
        "A": SimpleNamespace(Compose=NativeCompose),
        "FuseCompose": FusedCompose,
        "_ALBU_CASES": [("a01_rotate", 1, [Transform()])],
        "_CASES": [],
        "_MIXED_AGR_CASES": [("d01_mixed", 3, [Transform()], [object()], [object()])],
        "_IMAGE_NDARRAY": object(),
        "_IMAGE_TENSOR": object(),
        "_bench": lambda _pipeline: 1.0,
        "_bench_albu": lambda _pipeline: 1.0,
        "copy": copy,
        "K": SimpleNamespace(AugmentationSequential=lambda *_transforms: TensorNative()),
        "np": SimpleNamespace(random=SimpleNamespace(seed=lambda _seed: None)),
        "ReorderPolicy": SimpleNamespace(AGGRESSIVE="aggressive"),
        "statistics": statistics,
        "torch": SimpleNamespace(manual_seed=lambda _seed: None),
        "tv": SimpleNamespace(Compose=lambda _transforms: TensorNative()),
    }
    measure = _benchmark_function("_measure_once", namespace)

    first = measure()
    second = measure()

    assert native_seeds == [0, 0, 0, 0]
    assert fused_seeds == [0, 0, 0, 0]
    assert first == second
    assert {(case["case"], case["backend"]) for case in first[2]} == {
        ("a01_rotate", "albumentations"),
        ("d01_mixed", "kornia"),
        ("d01_mixed", "torchvision"),
        ("d01_mixed", "albumentations"),
    }


def test_score_details_include_repetitions_cases_and_runner(tmp_path: Path, monkeypatch, capsys):
    """The optional report preserves every case and runner while stdout stays gate-parseable."""
    first_cases = [{"case": "a01_rotate", "backend": "albumentations", "native_ms": 2.0, "fused_ms": 1.0, "boost": 2.0}]
    second_cases = [
        {"case": "a01_rotate", "backend": "albumentations", "native_ms": 6.0, "fused_ms": 2.0, "boost": 3.0}
    ]
    samples = iter([(1.0, 2.0, first_cases), (3.0, 2.0, second_cases)])
    details = tmp_path / "details.json"
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("RUNNER_NAME", "runner-17")
    monkeypatch.setenv("RUNNER_OS", "Linux")
    monkeypatch.setenv("GITHUB_JOB", "benchmark-pr")
    monkeypatch.setenv("BENCH_SOURCE_SHA", "history-commit")
    namespace = {
        "_measure_once": lambda: next(samples),
        "json": json,
        "os": os,
        "Path": Path,
        "statistics": statistics,
        "sys": sys,
    }
    _benchmark_function("_write_details", namespace)
    main = _benchmark_function("main", namespace)

    main(repetitions=2, details_json=str(details), summary_file=str(summary))

    assert capsys.readouterr().out == "real_score=2.0000\ntheoretical_target=2.0000\n"
    report = json.loads(details.read_text(encoding="utf-8"))
    assert report["runner"] == {"name": "runner-17", "os": "Linux", "job": "benchmark-pr"}
    assert report["source_sha"] == "history-commit"
    assert [trial["score"] for trial in report["repetitions"]] == [1.0, 3.0]
    assert [trial["cases"] for trial in report["repetitions"]] == [first_cases, second_cases]
    assert "| albumentations | a01_rotate | 4.0000 | 1.5000 | 2.500x |" in summary.read_text(encoding="utf-8")
