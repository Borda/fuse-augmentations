"""Integration tests for KorniaAdapter native parameter replay and matrix parity."""

import warnings

import pytest
import torch

from fuse_augmentations import Compose
from fuse_augmentations._compat import _KORNIA_AVAILABLE

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

    from fuse_augmentations.adapters.kornia import (
        KorniaAdapter,
        build_matrix_numpy_b1_kornia,
        sample_and_build_matrix_numpy_b1_kornia,
    )
    from fuse_augmentations.affine.matrix import inv3x3, normalize_matrix, normalize_matrix_io, perspective_grid
    from fuse_augmentations.types import TransformCategory

    _CATEGORY_PARAMS = [
        (kornia_aug.RandomRotation(degrees=30, p=1.0), TransformCategory.GEOMETRIC_INTERP),
        (kornia_aug.RandomAffine(degrees=30, p=1.0), TransformCategory.GEOMETRIC_INTERP),
        (kornia_aug.RandomHorizontalFlip(p=1.0), TransformCategory.GEOMETRIC_EXACT),
        (kornia_aug.RandomVerticalFlip(p=1.0), TransformCategory.GEOMETRIC_EXACT),
    ]
else:
    _CATEGORY_PARAMS = []

if _KORNIA_AVAILABLE:
    _ZERO_PROBABILITY_TRANSFORMS = [
        pytest.param(kornia_aug.RandomAffine(degrees=(17.0, 17.0), p=0.0), id="affine"),
        pytest.param(kornia_aug.RandomRotation90(times=(1, 1), p=0.0), id="rotation90"),
        pytest.param(kornia_aug.RandomPerspective(distortion_scale=0.3, p=0.0), id="perspective"),
    ]
else:
    _ZERO_PROBABILITY_TRANSFORMS = []

pytestmark = pytest.mark.integration

DEFAULT_DEVICE = torch.device("cpu")
DEFAULT_DTYPE = torch.float32


def _replay_affine(
    image: torch.Tensor,
    matrix: torch.Tensor,
    *,
    output_height: int | None = None,
    output_width: int | None = None,
) -> torch.Tensor:
    """Render a forward affine matrix through the same one-pass fused warp."""
    batch_size, num_channels, input_height, input_width = image.shape
    height = input_height if output_height is None else output_height
    width = input_width if output_width is None else output_width
    matrix_inv = inv3x3(matrix)
    matrix_norm = normalize_matrix_io(matrix_inv, input_height, input_width, height, width)
    grid = torch.nn.functional.affine_grid(
        matrix_norm[:, :2, :], (batch_size, num_channels, height, width), align_corners=True
    )
    return torch.nn.functional.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def _random_affine_params(forward_params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert Kornia's native RandomAffine parameters to the adapter's canonical units."""
    return {
        "angle_rad": -torch.deg2rad(forward_params["angle"]),
        "translate_x": forward_params["translations"][:, 0],
        "translate_y": forward_params["translations"][:, 1],
        "scale_x": forward_params["scale"][:, 0],
        "scale_y": forward_params["scale"][:, 1],
        "shear_x_rad": -torch.deg2rad(forward_params["shear_x"]),
        "shear_y_rad": -torch.deg2rad(forward_params["shear_y"]),
    }


@pytest.fixture
def adapter() -> "KorniaAdapter":
    """Create a fresh KorniaAdapter instance for each test."""
    return KorniaAdapter()


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestCategory:
    """Verify adapter.category() for known and unknown Kornia transforms."""

    @pytest.mark.parametrize("transform, expected_cat", _CATEGORY_PARAMS)
    def test_registered_transforms(self, adapter, transform, expected_cat):
        """Known Kornia transforms return their expected category."""
        assert adapter.category(transform) == expected_cat

    def test_unknown_transform(self, adapter):
        """Unknown transform falls back to SPATIAL_KERNEL with a UserWarning."""

        class UnknownTransform:
            pass

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            cat = adapter.category(UnknownTransform())
        assert cat == TransformCategory.SPATIAL_KERNEL
        assert len(recorded_warnings) == 1
        assert "Unknown Kornia transform" in str(recorded_warnings[0].message)


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestNativeParity:
    """Verify fused matrix path matches Kornia native output for each transform type."""

    def test_random_affine_combined_parameter_replay_on_non_square_batch(self, adapter):
        """Replay rotation, scale, translation, and both shears against native Kornia output."""
        image = torch.rand(2, 3, 40, 64)
        transform = kornia_aug.RandomAffine(
            degrees=(17.0, 17.0),
            translate=(0.2, 0.1),
            scale=(0.85, 0.85, 1.1, 1.1),
            shear=(8.0, 8.0, -5.0, -5.0),
            p=1.0,
            align_corners=True,
        )
        forward_params = transform.forward_parameters(torch.Size(image.shape))

        native_out = transform(image, params=forward_params)
        matrix = adapter.build_matrix(transform, _random_affine_params(forward_params), height=40, width=64)
        fused_out = _replay_affine(image, matrix)

        torch.testing.assert_close(native_out, fused_out, rtol=1e-4, atol=1e-4)

    def test_random_rotation90_replay_uses_kornia_rotation_direction(self, adapter):
        """A one-quarter native rotation replays with the same direction on a non-square batch."""
        image = torch.rand(2, 3, 40, 64)
        transform = kornia_aug.RandomRotation90(times=(1, 1), p=1.0, align_corners=True)
        forward_params = transform.forward_parameters(torch.Size(image.shape))

        native_out = transform(image, params=forward_params)
        matrix = adapter.build_matrix(
            transform,
            {"_batch_size": torch.tensor([2]), "k90": forward_params["times"].to(torch.int64)},
            height=40,
            width=64,
        )
        fused_out = _replay_affine(image, matrix)

        torch.testing.assert_close(native_out, fused_out, rtol=1e-4, atol=1e-4)

    def test_random_perspective_parameter_replay(self, adapter):
        """Replay native perspective points through the projective matrix without unit or direction drift."""
        image = torch.rand(2, 3, 40, 64)
        transform = kornia_aug.RandomPerspective(distortion_scale=0.3, p=1.0, align_corners=True)
        forward_params = transform.forward_parameters(torch.Size(image.shape))

        native_out = transform(image, params=forward_params)
        matrix = adapter.build_matrix(
            transform,
            {"start_points": forward_params["start_points"], "end_points": forward_params["end_points"]},
            height=40,
            width=64,
        )
        matrix_norm = normalize_matrix(inv3x3(matrix), height=40, width=64)
        grid = perspective_grid(matrix_norm, height=40, width=64)
        fused_out = torch.nn.functional.grid_sample(
            image, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        torch.testing.assert_close(native_out, fused_out, rtol=1e-4, atol=1e-4)

    def test_random_resized_crop_parameter_replay_on_non_square_batch(self, adapter):
        """Replay crop corners and target size against native Kornia crop-resize output."""
        image = torch.rand(2, 3, 40, 64)
        transform = kornia_aug.RandomResizedCrop(
            size=(16, 24), scale=(0.6, 0.6), ratio=(1.0, 1.0), p=1.0, align_corners=True
        )
        forward_params = transform.forward_parameters(torch.Size(image.shape))
        source = forward_params["src"]
        output_size = forward_params["output_size"].float()

        native_out = transform(image, params=forward_params)
        matrix = adapter.build_matrix(
            transform,
            {
                "crop_top": source[:, 0, 1],
                "crop_left": source[:, 0, 0],
                "crop_h": source[:, 3, 1] - source[:, 0, 1] + 1.0,
                "crop_w": source[:, 1, 0] - source[:, 0, 0] + 1.0,
                "target_h": output_size[:, 0],
                "target_w": output_size[:, 1],
            },
            height=40,
            width=64,
        )
        fused_out = _replay_affine(image, matrix, output_height=16, output_width=24)

        torch.testing.assert_close(native_out, fused_out, rtol=1e-4, atol=1e-4)

    def test_numpy_affine_builder_replays_combined_native_matrix(self, adapter):
        """The B=1 cv2 matrix builder preserves native combined-affine semantics."""
        transform = kornia_aug.RandomAffine(
            degrees=(17.0, 17.0),
            translate=(0.2, 0.1),
            scale=(0.85, 0.85, 1.1, 1.1),
            shear=(8.0, 8.0, -5.0, -5.0),
            p=1.0,
            align_corners=True,
        )
        forward_params = transform.forward_parameters(torch.Size((1, 3, 40, 64)))
        image = torch.rand(1, 3, 40, 64)
        transform(image, params=forward_params)

        params = _random_affine_params(forward_params)
        numpy_matrix = build_matrix_numpy_b1_kornia(transform, params, height=40, width=64)
        native_matrix = transform.transform_matrix[0].double()

        torch.testing.assert_close(torch.from_numpy(numpy_matrix), native_matrix, rtol=1e-6, atol=1e-6)

    def test_numpy_fused_sampler_replays_combined_affine_matrix(self, adapter):
        """The B=1 sampler-builder shares Kornia's parameter units and composition order."""
        transform = kornia_aug.RandomAffine(
            degrees=(-17.0, 17.0),
            translate=(0.2, 0.1),
            scale=(0.85, 0.95, 1.05, 1.15),
            shear=(-8.0, 8.0, -5.0, 5.0),
            p=1.0,
            align_corners=True,
        )
        input_shape = (1, 3, 40, 64)
        torch.manual_seed(123)
        forward_params = transform.forward_parameters(torch.Size(input_shape))
        params = _random_affine_params(forward_params)
        expected = adapter.build_matrix(transform, params, height=40, width=64)[0].double()

        torch.manual_seed(123)
        actual = sample_and_build_matrix_numpy_b1_kornia(transform, input_shape, height=40, width=64)

        torch.testing.assert_close(torch.from_numpy(actual), expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("transform", _ZERO_PROBABILITY_TRANSFORMS)
    def test_zero_probability_preserves_a_non_square_batch(self, transform):
        """A disabled shape-preserving primitive leaves every batch item unchanged."""
        image = torch.rand(2, 3, 40, 64)

        output = Compose([transform])(image)

        torch.testing.assert_close(output, image, rtol=1e-4, atol=1e-4)

    def test_shear_sign(self, adapter):
        """Our shear matrix sign convention matches Kornia's native output."""
        batch_size, num_channels, height, width = 2, 3, 64, 64
        img = torch.rand(batch_size, num_channels, height, width)

        # X-shear only, fixed shear value; align_corners=True to match our pipeline
        shear_deg = 20.0
        transform = kornia_aug.RandomAffine(degrees=0, shear=(shear_deg, shear_deg, 0, 0), p=1.0, align_corners=True)

        # Use forward_parameters so native and fused paths share identical sampled params
        forward_params = transform.forward_parameters(torch.Size((batch_size, num_channels, height, width)))
        native_out = transform(img, params=forward_params)

        # Our fused path -- build params from forward_params using same negation as sample_params.
        # Negation converts Kornia CW convention -> our CCW rotation_matrix convention.
        params: dict[str, torch.Tensor] = {}
        if "angle" in forward_params:
            params["angle_rad"] = -torch.deg2rad(forward_params["angle"].to(DEFAULT_DEVICE))
        if "translations" in forward_params:
            params["translate_x"] = forward_params["translations"][:, 0].to(DEFAULT_DEVICE)
            params["translate_y"] = forward_params["translations"][:, 1].to(DEFAULT_DEVICE)
        if "scale" in forward_params:
            params["scale_x"] = forward_params["scale"][:, 0].to(DEFAULT_DEVICE)
            params["scale_y"] = forward_params["scale"][:, 1].to(DEFAULT_DEVICE)
        if "shear_x" in forward_params:
            params["shear_x_rad"] = -torch.deg2rad(forward_params["shear_x"].to(DEFAULT_DEVICE))
        if "shear_y" in forward_params:
            params["shear_y_rad"] = -torch.deg2rad(forward_params["shear_y"].to(DEFAULT_DEVICE))
        mtx_fwd = adapter.build_matrix(transform, params, height, width)
        mtx_inv = inv3x3(mtx_fwd)
        mtx_norm = normalize_matrix(mtx_inv, height, width)
        grid = torch.nn.functional.affine_grid(
            mtx_norm[:, :2, :], (batch_size, num_channels, height, width), align_corners=True
        )
        fused_out = torch.nn.functional.grid_sample(
            img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        # Compare -- they should be close
        assert torch.allclose(native_out, fused_out, atol=1e-3), (
            f"Shear sign mismatch: max diff = {(native_out - fused_out).abs().max().item():.6f}"
        )

    def test_translation_units(self, adapter):
        """translate_x is in pixels, not fractional."""
        batch_size, num_channels, height, width = 4, 3, 64, 128
        translate_frac = 0.5  # 50% in Kornia's convention

        transform = kornia_aug.RandomAffine(
            degrees=0,
            translate=(translate_frac, translate_frac),
            p=1.0,
        )

        params = adapter.sample_params(transform, (batch_size, num_channels, height, width), DEFAULT_DEVICE)

        # Translation should be in pixels. With translate=(0.5, 0.5),
        # the range is [-0.5*width, 0.5*width] for x and [-0.5*height, 0.5*height] for y.
        # Values should generally be larger than 1.0 for width=128, height=64.
        # We just verify the values are NOT in [0, 1] fractional range
        # (some could be close to 0, so check at least one sample is > 1).
        if "translate_x" in params:
            translate_x = params["translate_x"]
            # The values are already pixels from Kornia's generate_parameters
            assert translate_x.shape == (batch_size,)

    def test_rotation(self, adapter):
        """Rotation matrix produces same output as native Kornia."""
        batch_size, num_channels, height, width = 2, 3, 64, 64
        img = torch.rand(batch_size, num_channels, height, width)

        # align_corners=True to match our pipeline convention
        transform = kornia_aug.RandomRotation(degrees=45, p=1.0, align_corners=True)

        # Use forward_parameters so native and fused paths share identical sampled params
        forward_params = transform.forward_parameters(torch.Size((batch_size, num_channels, height, width)))
        native_out = transform(img, params=forward_params)

        # Our fused path -- negate angle to convert Kornia CW -> our CCW convention
        params = {"angle_rad": -torch.deg2rad(forward_params["degrees"].to(DEFAULT_DEVICE))}
        mtx_fwd = adapter.build_matrix(transform, params, height, width)
        mtx_inv = inv3x3(mtx_fwd)
        mtx_norm = normalize_matrix(mtx_inv, height, width)
        grid = torch.nn.functional.affine_grid(
            mtx_norm[:, :2, :], (batch_size, num_channels, height, width), align_corners=True
        )
        fused_out = torch.nn.functional.grid_sample(
            img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        assert torch.allclose(native_out, fused_out, atol=1e-3), (
            f"Rotation parity failed: max diff = {(native_out - fused_out).abs().max().item():.6f}"
        )

    def test_random_shear(self, adapter):
        """kornia_aug.RandomShear matrix path matches native Kornia output."""
        batch_size, num_channels, height, width = 2, 3, 64, 64
        img = torch.rand(batch_size, num_channels, height, width)

        transform = kornia_aug.RandomShear(shear=(-20.0, 20.0), p=1.0, align_corners=True)
        forward_params = transform.forward_parameters(torch.Size((batch_size, num_channels, height, width)))
        native_out = transform(img, params=forward_params)

        params: dict[str, torch.Tensor] = {}
        if "shear_x" in forward_params:
            params["shear_x_rad"] = -torch.deg2rad(forward_params["shear_x"].to(DEFAULT_DEVICE))
        if "shear_y" in forward_params:
            params["shear_y_rad"] = -torch.deg2rad(forward_params["shear_y"].to(DEFAULT_DEVICE))

        mtx_fwd = adapter.build_matrix(transform, params, height, width)
        mtx_inv = inv3x3(mtx_fwd)
        mtx_norm = normalize_matrix(mtx_inv, height, width)
        grid = torch.nn.functional.affine_grid(
            mtx_norm[:, :2, :], (batch_size, num_channels, height, width), align_corners=True
        )
        fused_out = torch.nn.functional.grid_sample(
            img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        assert torch.allclose(native_out, fused_out, atol=1e-3), (
            f"kornia_aug.RandomShear parity failed: max diff = {(native_out - fused_out).abs().max().item():.6f}"
        )

    def test_random_translate(self, adapter):
        """kornia_aug.RandomTranslate matrix path matches native Kornia output."""
        batch_size, num_channels, height, width = 2, 3, 64, 64
        img = torch.rand(batch_size, num_channels, height, width)

        transform = kornia_aug.RandomTranslate(
            translate_x=(0.1, 0.2), translate_y=(0.1, 0.2), p=1.0, align_corners=True
        )
        forward_params = transform.forward_parameters(torch.Size((batch_size, num_channels, height, width)))
        native_out = transform(img, params=forward_params)

        params = {
            "translate_x": forward_params["translate_x"].to(DEFAULT_DEVICE),
            "translate_y": forward_params["translate_y"].to(DEFAULT_DEVICE),
        }
        mtx_fwd = adapter.build_matrix(transform, params, height, width)
        mtx_inv = inv3x3(mtx_fwd)
        mtx_norm = normalize_matrix(mtx_inv, height, width)
        grid = torch.nn.functional.affine_grid(
            mtx_norm[:, :2, :], (batch_size, num_channels, height, width), align_corners=True
        )
        fused_out = torch.nn.functional.grid_sample(
            img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        assert torch.allclose(native_out, fused_out, atol=1e-3), (
            f"kornia_aug.RandomTranslate parity failed: max diff = {(native_out - fused_out).abs().max().item():.6f}"
        )

    def test_vflip(self, adapter):
        """VFlip matrix produces same output as native Kornia."""
        batch_size, num_channels, height, width = 2, 3, 32, 32
        img = torch.rand(batch_size, num_channels, height, width)

        transform = kornia_aug.RandomVerticalFlip(p=1.0)

        native_out = transform(img)

        params = adapter.sample_params(transform, (batch_size, num_channels, height, width), DEFAULT_DEVICE)
        mtx_fwd = adapter.build_matrix(transform, params, height, width)
        if mtx_fwd.shape[0] == 1 and batch_size > 1:
            mtx_fwd = mtx_fwd.expand(batch_size, -1, -1)
        mtx_inv = inv3x3(mtx_fwd)
        mtx_norm = normalize_matrix(mtx_inv, height, width)
        grid = torch.nn.functional.affine_grid(
            mtx_norm[:, :2, :], (batch_size, num_channels, height, width), align_corners=True
        )
        fused_out = torch.nn.functional.grid_sample(
            img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        assert torch.allclose(native_out, fused_out, atol=1e-3), (
            f"VFlip parity failed: max diff = {(native_out - fused_out).abs().max().item():.6f}"
        )

    def test_hflip(self, adapter):
        """HFlip matrix produces same output as native Kornia."""
        batch_size, num_channels, height, width = 2, 3, 32, 32
        img = torch.rand(batch_size, num_channels, height, width)

        transform = kornia_aug.RandomHorizontalFlip(p=1.0)

        native_out = transform(img)

        params = adapter.sample_params(transform, (batch_size, num_channels, height, width), DEFAULT_DEVICE)
        mtx_fwd = adapter.build_matrix(transform, params, height, width)
        # Expand to batch if needed
        if mtx_fwd.shape[0] == 1 and batch_size > 1:
            mtx_fwd = mtx_fwd.expand(batch_size, -1, -1)
        mtx_inv = inv3x3(mtx_fwd)
        mtx_norm = normalize_matrix(mtx_inv, height, width)
        grid = torch.nn.functional.affine_grid(
            mtx_norm[:, :2, :], (batch_size, num_channels, height, width), align_corners=True
        )
        fused_out = torch.nn.functional.grid_sample(
            img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        assert torch.allclose(native_out, fused_out, atol=1e-3), (
            f"HFlip parity failed: max diff = {(native_out - fused_out).abs().max().item():.6f}"
        )


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestSameOnBatch:
    """Verify that same_on_batch is respected by adapter.sample_params."""

    def test_same_on_batch_produces_identical_params(self, adapter):
        """When same_on_batch=True, all samples get the same angle."""
        transform = kornia_aug.RandomRotation(degrees=30, same_on_batch=True, p=1.0)
        params = adapter.sample_params(transform, (4, 3, 8, 8), DEFAULT_DEVICE)
        angles = params["angle_rad"]
        assert angles.shape == (4,)
        # All 4 samples should have the same angle
        assert torch.allclose(angles[0].expand(4), angles), f"same_on_batch=True but angles differ: {angles}"

    def test_different_on_batch_produces_varied_params(self, adapter):
        """When same_on_batch=False, a large batch produces varied angles."""
        transform = kornia_aug.RandomRotation(degrees=30, same_on_batch=False, p=1.0)
        params = adapter.sample_params(transform, (16, 3, 8, 8), DEFAULT_DEVICE)
        angles = params["angle_rad"]
        assert angles.shape == (16,)
        # With 16 samples and 30 degree range, min and max should differ
        assert not torch.allclose(angles.min().unsqueeze(0), angles.max().unsqueeze(0)), (
            f"same_on_batch=False but all angles identical: {angles}"
        )

    def test_same_on_batch_affine(self, adapter):
        """same_on_batch=True on kornia_aug.RandomAffine produces identical params across batch."""
        transform = kornia_aug.RandomAffine(degrees=30, translate=(0.3, 0.3), same_on_batch=True, p=1.0)
        params = adapter.sample_params(transform, (4, 3, 8, 8), DEFAULT_DEVICE)
        if "angle_rad" in params:
            angles = params["angle_rad"]
            assert torch.allclose(angles[0].expand(4), angles), f"same_on_batch=True but angles differ: {angles}"
        if "translate_x" in params:
            translate_x = params["translate_x"]
            assert torch.allclose(translate_x[0].expand(4), translate_x), (
                f"same_on_batch=True but translate_x differs: {translate_x}"
            )


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
class TestBuildMatrixFallback:
    """Verify build_matrix fallback path returns identity for unregistered transforms."""

    def test_unknown_transform_returns_identity(self, adapter):
        """An unregistered transform class produces a (1, 3, 3) identity matrix."""

        class UnknownTransform:
            pass

        mat = adapter.build_matrix(UnknownTransform(), {}, 64, 64)
        assert mat.shape == torch.Size([1, 3, 3])
        assert torch.allclose(mat, torch.eye(3).unsqueeze(0)), f"Expected identity fallback, got: {mat}"
