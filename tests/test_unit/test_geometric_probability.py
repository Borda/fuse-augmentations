"""Unit tests for per-transform geometric probability on ``from_params``.

``from_params`` exposes ``hflip_p``/``vflip_p`` but historically applied rotation and scale to every sample.
``rotation_p`` and ``scale_p`` extend the same convention to the geometric ranges, on both the backend-free engine and
the ``backend=`` path that builds ``TransformSpec`` objects.

The backward-compatibility direction matters as much as the new behaviour: the default of ``1.0`` has to leave existing
seeded pipelines bit-for-bit unchanged, which is what pins the split into separate transforms to the case where a
probability was actually lowered.

"""

from __future__ import annotations

import pytest
import torch

from fuse_augmentations.compose import FusedCompose

IMAGE_SHAPE = (4, 3, 16, 16)


@pytest.fixture
def image() -> torch.Tensor:
    """Return a deterministic ``(4, 3, 16, 16)`` float32 batch with structure a warp can move."""
    values = torch.linspace(0.0, 1.0, steps=IMAGE_SHAPE[2] * IMAGE_SHAPE[3])
    return values.reshape(1, 1, *IMAGE_SHAPE[2:]).expand(IMAGE_SHAPE).contiguous()


def _run_seeded(seed: int, **kwargs: object) -> torch.Tensor:
    """Run a ``from_params`` pipeline under a caller-owned generator so results are comparable."""
    generator = torch.Generator().manual_seed(seed)
    pipe = FusedCompose.from_params(generator=generator, **kwargs)
    image = torch.linspace(0.0, 1.0, steps=IMAGE_SHAPE[2] * IMAGE_SHAPE[3])
    image = image.reshape(1, 1, *IMAGE_SHAPE[2:]).expand(IMAGE_SHAPE).contiguous()
    return torch.as_tensor(pipe(image))


class TestRotationProbability:
    """``rotation_p`` on the backend-free engine."""

    def test_zero_probability_leaves_the_image_unchanged(self, image: torch.Tensor) -> None:
        """``rotation_p=0.0`` never applies the rotation, so the output equals the input.

        A probability that is accepted but ignored is the failure this guards: the parameter would
        look wired up while every sample still got rotated.

        """
        pipe = FusedCompose.from_params(rotation=(-45.0, 45.0), rotation_p=0.0)

        out = torch.as_tensor(pipe(image))

        torch.testing.assert_close(out, image)

    def test_full_probability_reproduces_the_previous_default(self, image: torch.Tensor) -> None:
        """``rotation_p=1.0`` matches an identically seeded call that does not pass it at all.

        This is the compatibility guard: the default must not change what an existing seeded
        pipeline produces, which it would if the parameter altered the number or order of draws.

        """
        without = _run_seeded(7, rotation=(-45.0, 45.0))
        with_default = _run_seeded(7, rotation=(-45.0, 45.0), rotation_p=1.0)

        torch.testing.assert_close(with_default, without)

    def test_intermediate_probability_applies_to_some_samples_only(self) -> None:
        """A probability below one leaves part of the batch untransformed.

        Per-sample gating is the point of the parameter — a pipeline that applied the gate once per batch would either
        rotate everything or nothing and still pass a shape-only assertion.

        """
        generator = torch.Generator().manual_seed(3)
        pipe = FusedCompose.from_params(rotation=(30.0, 45.0), rotation_p=0.5, generator=generator)
        image = torch.rand(32, 3, 16, 16)

        out = torch.as_tensor(pipe(image))

        # An ungated sample still passes through the shared warp with an identity matrix, so it is
        # equal to its input only up to resampling error -- hence a tolerance rather than equality.
        unchanged = torch.tensor([
            bool(torch.allclose(out[idx], image[idx], atol=1e-4)) for idx in range(image.shape[0])
        ])
        assert bool(unchanged.any())
        assert not bool(unchanged.all())


class TestScaleProbability:
    """``scale_p`` on the backend-free engine."""

    def test_zero_probability_leaves_the_image_unchanged(self, image: torch.Tensor) -> None:
        """``scale_p=0.0`` never applies the scale range.

        Scale shares one probability across ``scale``/``scale_x``/``scale_y``, so this also pins that the shared gate is
        actually consulted rather than defaulted per key.

        """
        pipe = FusedCompose.from_params(scale=(0.5, 1.5), scale_p=0.0)

        out = torch.as_tensor(pipe(image))

        torch.testing.assert_close(out, image)

    def test_per_axis_ranges_share_the_probability(self, image: torch.Tensor) -> None:
        """``scale_x`` and ``scale_y`` are gated by ``scale_p`` too, not left at 1.0.

        The per-axis ranges are the scale family under a different spelling; gating only the uniform ``scale`` would
        make the parameter's effect depend on which spelling the caller used.

        """
        pipe = FusedCompose.from_params(scale_x=(0.5, 1.5), scale_y=(0.5, 1.5), scale_p=0.0)

        out = torch.as_tensor(pipe(image))

        torch.testing.assert_close(out, image)

    def test_rotation_still_applies_when_only_scale_is_disabled(self, image: torch.Tensor) -> None:
        """Disabling scale does not disable rotation in the same pipeline.

        The two ranges are gated independently, so a shared gate — the simplest wrong implementation, since both live in
        one direct-parameter transform by default — would show up here as an unchanged image.

        """
        generator = torch.Generator().manual_seed(11)
        pipe = FusedCompose.from_params(
            rotation=(40.0, 45.0),
            scale=(0.5, 1.5),
            scale_p=0.0,
            rotation_p=1.0,
            generator=generator,
        )

        out = torch.as_tensor(pipe(image))

        assert not bool(torch.equal(out, image))


class TestBackendSpecPath:
    """``rotation_p``/``scale_p`` reaching the ``TransformSpec`` builder used by ``backend=``."""

    @pytest.mark.parametrize(
        ("kwargs", "operation", "expected"),
        [
            pytest.param({"rotation": (-30.0, 30.0), "rotation_p": 0.25}, "rotation", 0.25, id="rotation"),
            pytest.param({"scale": (0.5, 1.5), "scale_p": 0.75}, "scale", 0.75, id="scale"),
            pytest.param({"rotation": (-30.0, 30.0)}, "rotation", 1.0, id="rotation-default"),
        ],
    )
    def test_probability_lands_on_the_spec(
        self,
        kwargs: dict[str, object],
        operation: str,
        expected: float,
    ) -> None:
        """The geometric spec carries the requested probability instead of a hardcoded 1.0.

        The spec builder is the only place the ``backend=`` path can express a per-op probability, so this is where the
        parameter has to arrive for that path to honour it at all.

        """
        specs = FusedCompose._geometric_kwargs_to_specs(**kwargs)

        match = next(spec for spec in specs if spec.operation == operation)
        assert match.prob == expected


class TestProbabilityValidation:
    """Bounds and mutual-exclusivity checks for the new parameters."""

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            pytest.param("rotation_p", 1.5, id="rotation-above-one"),
            pytest.param("rotation_p", -0.1, id="rotation-below-zero"),
            pytest.param("scale_p", 2.0, id="scale-above-one"),
        ],
    )
    def test_out_of_range_probability_raises_at_construction(self, name: str, value: float) -> None:
        """A probability outside ``[0, 1]`` raises when the pipeline is built, not at the first call.

        Construction time is where the caller can still see which parameter was wrong; deferring to the first draw would
        surface it inside a data loader worker instead.

        """
        with pytest.raises(ValueError, match=r"must be in \[0.0, 1.0\]"):
            FusedCompose.from_params(rotation=(-30.0, 30.0), **{name: value})

    def test_probability_conflicts_with_the_specs_overload(self) -> None:
        """Passing ``rotation_p`` alongside ``specs=`` raises, like every other keyword parameter.

        ``specs=`` already carries its own per-op probabilities, so accepting both would leave two sources for the same
        value with no rule for which wins.

        """
        from fuse_augmentations.types import TransformSpec

        with pytest.raises(ValueError, match="mutually exclusive"):
            FusedCompose.from_params(
                specs=[TransformSpec(operation="rotation", params={"degrees": (-30.0, 30.0)})],
                rotation_p=0.5,
            )
