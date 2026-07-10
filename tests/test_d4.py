"""Symbolic-exactness / D4-group execution tests.

Covers the two halves of the feature:

- ``classify_d4_batch`` / ``apply_d4_image`` matrix primitives (exact-integer
  classification of the eight dihedral-group elements, axis-swap gating on
  non-square images, and lossless application via ``flip``/``rot90``).
- ``FusedAffineSegment`` dispatch: an interpolating chain that composes to a D4
  element executes losslessly (no ``grid_sample``) and matches the equivalent
  single tensor op bit-for-bit, while a non-D4 chain (including a sub-degree-off
  near-quarter-turn) stays on the grid path.

Requires kornia.

"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from typing_extensions import Self

from fuse_augmentations import Compose
from fuse_augmentations._compat import _KORNIA_AVAILABLE
from fuse_augmentations.affine.matrix import (
    apply_d4_image,
    classify_d4_batch,
    hflip_matrix,
    vflip_matrix,
)

if _KORNIA_AVAILABLE:
    import kornia.augmentation as ka

pytestmark = pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required")

BATCH, CHANNELS, SIDE = 2, 3, 16

_D4_NAMES = ("identity", "hflip", "vflip", "rot180", "rot90", "rot270", "transpose", "anti_transpose")


def _tensor_op(name: str, image: torch.Tensor) -> torch.Tensor:
    """Apply the reference tensor op for a D4 name via flip/rot90 primitives."""
    ops = {
        "identity": lambda x: x,
        "hflip": lambda x: x.flip(dims=[-1]),
        "vflip": lambda x: x.flip(dims=[-2]),
        "rot180": lambda x: torch.rot90(x, k=2, dims=[-2, -1]),
        "rot90": lambda x: torch.rot90(x, k=1, dims=[-2, -1]),
        "rot270": lambda x: torch.rot90(x, k=3, dims=[-2, -1]),
        "transpose": lambda x: x.transpose(-2, -1),
        "anti_transpose": lambda x: torch.rot90(x, k=1, dims=[-2, -1]).flip(dims=[-1]),
    }
    return ops[name](image)


class _GridSampleSpy:
    """Context-managed spy that counts ``F.grid_sample`` calls."""

    def __init__(self) -> None:
        """Initialise with a zero call count."""
        self.calls = 0
        self._orig = F.grid_sample

    def __enter__(self) -> Self:
        """Patch ``F.grid_sample`` with a counting wrapper."""

        def _counting(*args: object, **kwargs: object) -> torch.Tensor:
            self.calls += 1
            return self._orig(*args, **kwargs)  # type: ignore[arg-type]

        F.grid_sample = _counting  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: object) -> None:
        """Restore the original ``F.grid_sample``."""
        F.grid_sample = self._orig  # type: ignore[assignment]


class TestClassifyD4Batch:
    """``classify_d4_batch`` exact-integer classification of D4 forward matrices."""

    @pytest.mark.parametrize("name", _D4_NAMES, ids=list(_D4_NAMES))
    def test_canonical_matrix_classifies_to_its_name(self, name: str) -> None:
        """Each canonical D4 forward matrix classifies back to its own op name."""
        from fuse_augmentations.affine.matrix import _d4_forward_matrix

        matrix = _d4_forward_matrix(name, SIDE, SIDE).unsqueeze(0).expand(BATCH, -1, -1)
        assert classify_d4_batch(matrix, SIDE, SIDE) == name

    def test_identity_matrix_classifies_identity(self) -> None:
        """A batched identity matrix classifies as ``identity``."""
        eye = torch.eye(3).unsqueeze(0).expand(BATCH, -1, -1)
        assert classify_d4_batch(eye, SIDE, SIDE) == "identity"

    def test_hflip_matrix_helper_classifies_hflip(self) -> None:
        """The production ``hflip_matrix`` builder classifies as ``hflip``."""
        matrix = hflip_matrix(width=SIDE, batch_size=BATCH, device=torch.device("cpu"), dtype=torch.float32)
        assert classify_d4_batch(matrix, SIDE, SIDE) == "hflip"

    def test_vflip_matrix_helper_classifies_vflip(self) -> None:
        """The production ``vflip_matrix`` builder classifies as ``vflip``."""
        matrix = vflip_matrix(height=SIDE, batch_size=BATCH, device=torch.device("cpu"), dtype=torch.float32)
        assert classify_d4_batch(matrix, SIDE, SIDE) == "vflip"

    def test_non_d4_rotation_returns_none(self) -> None:
        """A 45-degree rotation matrix is not a D4 element -> ``None``."""
        angle = torch.tensor(torch.pi / 4)
        cos, sin = torch.cos(angle), torch.sin(angle)
        matrix = torch.eye(3).unsqueeze(0)
        matrix[0, 0, 0], matrix[0, 0, 1] = cos, -sin
        matrix[0, 1, 0], matrix[0, 1, 1] = sin, cos
        assert classify_d4_batch(matrix, SIDE, SIDE) is None

    def test_extra_translation_beyond_border_form_returns_none(self) -> None:
        """A flip with an added integer shift is not the border-preserving D4 form."""
        matrix = hflip_matrix(width=SIDE, batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        matrix[0, 0, 2] += 3.0  # net translation now differs from the canonical hflip
        assert classify_d4_batch(matrix, SIDE, SIDE) is None

    @pytest.mark.parametrize("name", ["rot90", "rot270", "transpose", "anti_transpose"], ids=lambda n: n)
    def test_axis_swap_ops_rejected_on_non_square(self, name: str) -> None:
        """Axis-swapping D4 ops are rejected on non-square images (shape change)."""
        from fuse_augmentations.affine.matrix import _d4_forward_matrix

        matrix = _d4_forward_matrix(name, height=8, width=16).unsqueeze(0)
        assert classify_d4_batch(matrix, height=8, width=16) is None

    def test_mixed_batch_elements_returns_none(self) -> None:
        """A batch where samples land on different D4 elements is not dispatchable."""
        hflip = hflip_matrix(width=SIDE, batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        vflip = vflip_matrix(height=SIDE, batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        matrix = torch.cat([hflip, vflip], dim=0)
        assert classify_d4_batch(matrix, SIDE, SIDE) is None


class TestApplyD4Image:
    """``apply_d4_image`` losslessly reproduces each D4 tensor op."""

    @pytest.mark.parametrize("name", _D4_NAMES, ids=list(_D4_NAMES))
    def test_matches_reference_tensor_op(self, name: str) -> None:
        """``apply_d4_image`` equals the reference flip/rot90 op exactly (atol=0)."""
        image = torch.rand(BATCH, CHANNELS, SIDE, SIDE)
        assert torch.equal(apply_d4_image(image, name), _tensor_op(name, image))


class TestFusedAffineSegmentD4Dispatch:
    """``FusedAffineSegment`` routes composed-D4 interpolating chains losslessly."""

    def test_hflip_rot90_vflip_chain_matches_single_op_without_grid_sample(self) -> None:
        """A hflip+vflip+90-degree chain composes to one D4 op, applied with zero grid_sample."""
        torch.manual_seed(0)
        image = torch.rand(BATCH, CHANNELS, SIDE, SIDE)
        pipe = Compose([
            ka.RandomHorizontalFlip(p=1.0),
            ka.RandomVerticalFlip(p=1.0),
            ka.RandomRotation(degrees=(90.0, 90.0), p=1.0, align_corners=True),
        ])
        with _GridSampleSpy() as spy:
            out = pipe(image.clone())
        expected = _tensor_op("rot270", image)  # hflip . vflip . rot90 == rot270
        assert spy.calls == 0
        assert torch.equal(out, expected)

    def test_interp_rot180_uses_zero_grid_sample(self) -> None:
        """A lone interpolating 180-degree rotation chain runs on the exact path."""
        torch.manual_seed(0)
        image = torch.rand(BATCH, CHANNELS, SIDE, SIDE)
        pipe = Compose([
            ka.RandomRotation(degrees=(180.0, 180.0), p=1.0, align_corners=True),
            ka.RandomHorizontalFlip(p=1.0),
        ])
        with _GridSampleSpy() as spy:
            out = pipe(image.clone())
        expected = _tensor_op("vflip", image)  # rot180 . hflip == vflip
        assert spy.calls == 0
        assert torch.equal(out, expected)

    def test_non_d4_chain_still_uses_grid_sample(self) -> None:
        """A non-D4 (45-degree) interpolating chain stays on the grid_sample path.

        Uses ``batch>1`` so the torch grid path runs (not the B=1 cv2 fast path); a
        non-D4 composition must fall through to ``grid_sample``.

        """
        torch.manual_seed(0)
        image = torch.rand(BATCH, CHANNELS, SIDE, SIDE)
        pipe = Compose([
            ka.RandomRotation(degrees=(45.0, 45.0), p=1.0, align_corners=True),
            ka.RandomHorizontalFlip(p=1.0),
        ])
        with _GridSampleSpy() as spy:
            pipe(image.clone())
        assert spy.calls >= 1


class TestNearD4Rotations:
    """Sub-degree-off rotations must NOT snap to an exact D4 op; exact quarter-turns must."""

    @pytest.mark.parametrize("side", [16, 64, 256, 512], ids=lambda s: f"{s}px")
    def test_exact_quarter_turn_uses_exact_path(self, side: int) -> None:
        """An exactly-90-degree composed chain runs on the exact path (no grid_sample)."""
        torch.manual_seed(0)
        image = torch.rand(BATCH, CHANNELS, side, side)
        pipe = Compose([
            ka.RandomHorizontalFlip(p=1.0),
            ka.RandomRotation(degrees=(90.0, 90.0), p=1.0, align_corners=True),
        ])
        with _GridSampleSpy() as spy:
            out = pipe(image.clone())
        assert spy.calls == 0
        assert torch.equal(out, image.transpose(-2, -1))  # hflip . rot90 == transpose

    @pytest.mark.parametrize("side", [16, 64, 256, 512], ids=lambda s: f"{s}px")
    @pytest.mark.parametrize("deg", [89.99, 90.01], ids=lambda d: f"{d}deg")
    def test_near_quarter_turn_stays_on_grid_path(self, deg: float, side: int) -> None:
        """A sub-degree-off rotation stays on grid_sample and is NOT snapped to rot90."""
        torch.manual_seed(0)
        image = torch.rand(BATCH, CHANNELS, side, side)
        pipe = Compose([
            ka.RandomHorizontalFlip(p=1.0),
            ka.RandomRotation(degrees=(deg, deg), p=1.0, align_corners=True),
        ])
        with _GridSampleSpy() as spy:
            out = pipe(image.clone())
        assert spy.calls >= 1
        assert not torch.equal(out, image.transpose(-2, -1))

    def test_classify_rejects_near_90_matrix_scale_free(self) -> None:
        """`classify_d4_batch` rejects an 89.99-degree matrix at a large size (scale-free eps)."""
        torch.manual_seed(0)
        side = 256
        image = torch.rand(1, CHANNELS, side, side)
        pipe = Compose([ka.RandomRotation(degrees=(89.99, 89.99), p=1.0, align_corners=True)])
        pipe(image.clone())
        assert classify_d4_batch(pipe.transform_matrix, side, side) is None


class TestFusedAffineSegmentD4Aux:
    """D4 dispatch routes auxiliary targets exactly and never raises."""

    def test_d4_chain_routes_mask_exactly(self) -> None:
        """A composed-D4 chain routes a mask losslessly through the same op."""
        torch.manual_seed(0)
        image = torch.rand(BATCH, CHANNELS, SIDE, SIDE)
        mask = torch.randint(0, 4, (BATCH, 1, SIDE, SIDE)).float()
        pipe = Compose(
            [ka.RandomHorizontalFlip(p=1.0), ka.RandomRotation(degrees=(90.0, 90.0), p=1.0, align_corners=True)],
            data_keys=["input", "mask"],
        )
        out_image, out_mask = pipe(image.clone(), mask.clone())
        expected_mask = _tensor_op("transpose", mask)  # hflip . rot90 == transpose
        assert torch.equal(out_mask, expected_mask)
        assert out_image.shape == (BATCH, CHANNELS, SIDE, SIDE)

    def test_d4_chain_routes_keypoints_via_exact_matrix(self) -> None:
        """A composed-D4 chain routes keypoints through the exact forward matrix."""
        torch.manual_seed(0)
        image = torch.rand(BATCH, CHANNELS, SIDE, SIDE)
        keypoints = torch.tensor([[[2.0, 3.0], [10.0, 5.0]]]).expand(BATCH, -1, -1).clone()
        pipe = Compose(
            [ka.RandomHorizontalFlip(p=1.0), ka.RandomRotation(degrees=(90.0, 90.0), p=1.0, align_corners=True)],
            data_keys=["input", "keypoints"],
        )
        _, out_kp = pipe(image.clone(), keypoints)
        # transpose maps (x, y) -> (y, x)
        expected = keypoints.flip(dims=[-1])
        assert torch.allclose(out_kp, expected, atol=1e-4)


class TestExactSegmentNonFlipAux:
    """A non-flip exact op routes a mask losslessly; box/keypoint aux never raises."""

    def test_rotate90_routes_mask_without_raising(self) -> None:
        """RandomRotation90 with a mask returns image and mask (no RuntimeError)."""
        torch.manual_seed(0)
        image = torch.rand(BATCH, CHANNELS, SIDE, SIDE)
        mask = torch.randint(0, 4, (BATCH, 1, SIDE, SIDE)).float()
        pipe = Compose([ka.RandomRotation90(times=(1, 1), p=1.0)], data_keys=["input", "mask"])
        out_image, out_mask = pipe(image.clone(), mask.clone())
        assert out_image.shape == (BATCH, CHANNELS, SIDE, SIDE)
        assert out_mask.shape == (BATCH, 1, SIDE, SIDE)
        # Image and mask share the identical rotation -> mask equals rot90 of input.
        assert torch.equal(out_mask, torch.rot90(mask, k=1, dims=[-2, -1]))

    def test_standalone_rotate90_with_keypoints_does_not_raise(self) -> None:
        """A standalone RandomRotation90 + keypoints routes via the grid path, no raise."""
        torch.manual_seed(0)
        image = torch.rand(BATCH, CHANNELS, SIDE, SIDE)
        keypoints = torch.tensor([[[2.0, 3.0]]]).expand(BATCH, -1, -1).clone()
        pipe = Compose([ka.RandomRotation90(times=(1, 1), p=1.0)], data_keys=["input", "keypoints"])
        out_image, out_kp = pipe(image.clone(), keypoints)
        assert out_image.shape == (BATCH, CHANNELS, SIDE, SIDE)
        assert out_kp.shape == (BATCH, 1, 2)

    def test_all_exact_chain_with_keypoints_does_not_raise(self) -> None:
        """An all-exact flip+rot90 chain + keypoints routes without raising (grid fallback)."""
        torch.manual_seed(0)
        image = torch.rand(BATCH, CHANNELS, SIDE, SIDE)
        keypoints = torch.tensor([[[2.0, 3.0], [10.0, 5.0]]]).expand(BATCH, -1, -1).clone()
        pipe = Compose(
            [ka.RandomHorizontalFlip(p=1.0), ka.RandomRotation90(times=(1, 1), p=1.0)],
            data_keys=["input", "keypoints"],
        )
        _, out_kp = pipe(image.clone(), keypoints)
        assert out_kp.shape == (BATCH, 2, 2)

    def test_all_exact_chain_routes_keypoints_through_fused_segment(self) -> None:
        """An all-exact chain + coord aux is built as a FusedAffineSegment (grid path)."""
        pipe = Compose(
            [ka.RandomHorizontalFlip(p=1.0), ka.RandomRotation90(times=(1, 1), p=1.0)],
            data_keys=["input", "keypoints"],
        )
        assert type(pipe._segments[0]).__name__ == "FusedAffineSegment"

    def test_all_exact_chain_mask_only_stays_exact_segment(self) -> None:
        """An all-exact chain with only a mask keeps the lossless ExactAffineSegment."""
        pipe = Compose(
            [ka.RandomHorizontalFlip(p=1.0), ka.RandomRotation90(times=(1, 1), p=1.0)],
            data_keys=["input", "mask"],
        )
        assert type(pipe._segments[0]).__name__ == "ExactAffineSegment"
