"""Regression tests pinning the Albumentations quarter-turn matrix direction.

The build_matrix quarter-turn branch (RandomRotate90 / D4 r90/r270) must warp in the
same direction as native ``np.rot90(+k)`` and the adapter's own ``exact_apply`` path.
The matrix path only executes when a quarter-turn is fused with an interpolating
transform, so single-transform pipelines never exercise it — these tests pin it
directly and through mixed fusion.

"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE
from fuse_augmentations.affine.matrix import inv3x3, normalize_matrix_io

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as A

    from fuse_augmentations import Compose, ReorderPolicy
    from fuse_augmentations.adapters.albumentations import (
        AlbumentationsAdapter,
        _apply_d4_element,
        _d4_matrix,
    )

pytestmark = pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")


def _smooth_image(height: int = 32, width: int = 32) -> torch.Tensor:
    """Smooth asymmetric image so rotation direction is observable."""
    yy, xx = torch.meshgrid(torch.linspace(0, 3.14, height), torch.linspace(0, 3.14, width), indexing="ij")
    plane = 0.5 + 0.3 * torch.sin(3 * xx) * torch.cos(2 * yy) + 0.15 * xx / 3.14
    return plane.reshape(1, 1, height, width).expand(1, 3, height, width).contiguous()


def _warp_forward(image: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Warp with a forward pixel-space matrix using the fused-segment grid convention."""
    batch, channels, height, width = image.shape
    theta_full = normalize_matrix_io(inv3x3(matrix), height, width, height, width)
    grid = F.affine_grid(theta_full[:, :2, :], size=(batch, channels, height, width), align_corners=True)
    return F.grid_sample(image, grid, mode="bilinear", align_corners=True)


class TestQuarterTurnMatrixDirection:
    @pytest.mark.parametrize(
        "k90",
        [
            pytest.param(0, id="k0-identity"),
            pytest.param(1, id="k1-ccw"),
            pytest.param(2, id="k2-half"),
            pytest.param(3, id="k3-cw"),
        ],
    )
    def test_rotate90_matrix_warp_matches_torch_rot90(self, k90):
        """build_matrix(RandomRotate90) warp equals native np.rot90(+k) direction."""
        image = _smooth_image()
        params = {
            "_batch_size": torch.tensor([1]),
            "k90": torch.tensor([k90], dtype=torch.int64),
        }
        matrix = AlbumentationsAdapter.build_matrix(A.RandomRotate90(p=1.0), params, 32, 32)

        warped = _warp_forward(image, matrix)

        expected = torch.rot90(image, k=k90, dims=(-2, -1))
        torch.testing.assert_close(warped, expected, atol=1e-5, rtol=0)

    @pytest.mark.parametrize(
        "elem",
        [
            pytest.param("e", id="identity"),
            pytest.param("r90", id="r90"),
            pytest.param("r180", id="r180"),
            pytest.param("r270", id="r270"),
            pytest.param("h", id="hflip"),
            pytest.param("v", id="vflip"),
            pytest.param("t", id="transpose"),
            pytest.param("hvt", id="anti-transpose"),
        ],
    )
    def test_d4_matrix_warp_matches_exact_apply(self, elem):
        """_d4_matrix warp equals _apply_d4_element for every D4 group element."""
        image = _smooth_image()
        from fuse_augmentations.adapters.albumentations import _D4_ELEM_TO_CODE

        code = torch.tensor([_D4_ELEM_TO_CODE[elem]], dtype=torch.int64)
        matrix = _d4_matrix(code, height=32, width=32, device=image.device, dtype=torch.float32)

        warped = _warp_forward(image, matrix)

        expected = _apply_d4_element(image, elem)
        torch.testing.assert_close(warped, expected, atol=1e-5, rtol=0)


def _fused_and_affine_only(factor: int, execution: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Affine+RandomRotate90 fused vs the affine-only pipeline with a pinned factor."""
    image = _smooth_image()
    affine = A.Affine(rotate=(20.0, 20.0), p=1.0)
    with patch.object(A.RandomRotate90, "get_params", return_value={"factor": factor}):
        fused = Compose([affine, A.RandomRotate90(p=1.0)], reorder=ReorderPolicy.NONE, execution=execution)(image)
    affine_only = Compose([affine], reorder=ReorderPolicy.NONE, execution=execution)(image)
    return fused, affine_only


class TestMixedFusionQuarterTurnDirection:
    @pytest.mark.parametrize(
        "factor",
        [
            pytest.param(1, id="factor1"),
            pytest.param(3, id="factor3"),
        ],
    )
    def test_fused_torch_execution_matches_native_direction(self, factor):
        """Torch execution: fused Affine+RandomRotate90 equals rot90(+factor) of affine-only output."""
        fused, affine_only = _fused_and_affine_only(factor, "torch")

        # rot90 is a lossless permutation, so it commutes with the shared single warp.
        expected = torch.rot90(affine_only, k=factor, dims=(-2, -1))
        torch.testing.assert_close(fused, expected, atol=1e-5, rtol=0)

    @pytest.mark.parametrize(
        "factor",
        [
            pytest.param(1, id="factor1"),
            pytest.param(3, id="factor3"),
        ],
    )
    def test_fused_cv2_execution_matches_native_direction(self, factor):
        """cv2 execution: mean-abs bound — border pixels round differently on the composed grid
        (isolated outliers up to ~3e-2 max), while a direction regression shifts the whole image
        to ~6e-2 MEAN — 60x above this bound."""
        fused, affine_only = _fused_and_affine_only(factor, "cv2")

        expected = torch.rot90(affine_only, k=factor, dims=(-2, -1))
        assert (fused - expected).abs().mean().item() < 1e-3
