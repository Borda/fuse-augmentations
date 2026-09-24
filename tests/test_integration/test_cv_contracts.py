"""Regression contracts for CV targets, gradients, hooks, and passthrough safety.

Each case uses ``Compose`` as a caller does.  The assertions target silent failures that can retain plausible output
shapes while corrupting labels or changing PyTorch module semantics.

"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fuse_augmentations import Compose
from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE, _TORCHVISION_AVAILABLE

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as albu

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

if _TORCHVISION_AVAILABLE:
    from torchvision.transforms import v2 as torchvision_v2


pytestmark = pytest.mark.integration


def _image(batch_size: int = 2) -> torch.Tensor:
    """Return an asymmetric batch whose mutations and gradients are observable."""
    values = torch.arange(batch_size * 3 * 8 * 10, dtype=torch.float32)
    return values.reshape(batch_size, 3, 8, 10) / values.numel()


@pytest.mark.parametrize(
    ("data_key", "target"),
    [
        pytest.param("bbox_xyxy", torch.tensor([[[1.0, 1.0, 4.0, 5.0]]]), id="boxes-batch-one"),
        pytest.param("keypoints", torch.tensor([[[1.0, 1.0], [4.0, 5.0]]]), id="keypoints-batch-one"),
    ],
)
def test_compose_rejects_coordinate_target_batch_mismatch(data_key: str, target: torch.Tensor) -> None:
    """A B=1 coordinate table cannot silently label every image in a B=2 batch."""
    pipe = Compose.from_params(hflip_p=1.0, data_keys=["input", data_key])

    with pytest.raises(ValueError, match=rf"{data_key}.*batch"):
        pipe(_image(), target)


def test_compose_rejects_extra_bbox_columns_instead_of_dropping_labels() -> None:
    """A fifth bbox column must fail rather than discarding an embedded class ID."""
    boxes_with_labels = torch.tensor([
        [[1.0, 1.0, 4.0, 5.0, 42.0]],
        [[2.0, 2.0, 5.0, 6.0, 99.0]],
    ])
    pipe = Compose.from_params(hflip_p=1.0, data_keys=["input", "bbox_xyxy"])

    with pytest.raises(ValueError, match=r"bbox_xyxy.*trailing.*4"):
        pipe(_image(), boxes_with_labels)


def test_compose_preserves_valid_empty_bbox_tables() -> None:
    """Schema validation accepts zero instances when the batch and bbox columns match."""
    boxes = torch.empty(2, 0, 4)
    pipe = Compose.from_params(hflip_p=1.0, data_keys=["input", "bbox_xyxy"])

    _, transformed_boxes = pipe(_image(), boxes)

    assert transformed_boxes.shape == boxes.shape
    assert transformed_boxes.dtype is boxes.dtype


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        pytest.param(torch.zeros(1, 1, 8, 10), r"mask.*batch", id="batch-one"),
        pytest.param(torch.zeros(2, 1, 7, 10), r"mask.*spatial", id="wrong-height"),
    ],
)
def test_compose_rejects_mask_batch_and_spatial_mismatch(mask: torch.Tensor, message: str) -> None:
    """Masks must share both the image batch and spatial canvas before any warp runs."""
    pipe = Compose.from_params(hflip_p=1.0, data_keys=["input", "mask"])

    with pytest.raises(ValueError, match=message):
        pipe(_image(), mask)


def test_compose_rejects_rboxes_with_wrong_trailing_width() -> None:
    """Rotated boxes require exactly cx, cy, width, height, and angle."""
    incomplete_rboxes = torch.zeros(2, 1, 4)
    pipe = Compose.from_params(hflip_p=1.0, data_keys=["input", "rboxes"])

    with pytest.raises(ValueError, match=r"rboxes.*trailing.*5"):
        pipe(_image(), incomplete_rboxes)


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations required for native NumPy routing")
def test_native_numpy_multi_target_rejects_batched_boxes_for_one_image() -> None:
    """The native NumPy shortcut validates target batch cardinality before routing boxes."""
    image_hwc = np.zeros((8, 10, 3), dtype=np.uint8)
    two_box_batches = np.array(
        [
            [[1.0, 1.0, 4.0, 5.0]],
            [[2.0, 2.0, 5.0, 6.0]],
        ],
        dtype=np.float32,
    )
    pipe = Compose(
        [albu.Affine(rotate=(0.0, 0.0), scale=(1.0, 1.0), translate_percent=(0.0, 0.0), p=1.0)],
        data_keys=["input", "bbox_xyxy"],
        execution="cv2",
    )

    with pytest.raises(ValueError, match=r"bbox_xyxy.*batch"):
        pipe(image=image_hwc, bboxes=two_box_batches)


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations required for native NumPy routing")
def test_native_numpy_multi_target_preserves_unbatched_empty_boxes() -> None:
    """The supported HWC single-image API keeps an empty ``(0, 4)`` table unbatched."""
    image_hwc = np.zeros((8, 10, 3), dtype=np.uint8)
    boxes = np.empty((0, 4), dtype=np.float32)
    pipe = Compose(
        [albu.Affine(rotate=(0.0, 0.0), scale=(1.0, 1.0), translate_percent=(0.0, 0.0), p=1.0)],
        data_keys=["input", "bbox_xyxy"],
        execution="cv2",
    )

    output = pipe(image=image_hwc, bboxes=boxes)

    assert output["bboxes"].shape == boxes.shape
    assert output["bboxes"].dtype == np.float32


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations required for batched NumPy routing")
def test_batched_numpy_keyword_targets_fall_back_from_native_shortcut() -> None:
    """BHWC arrays route image batches and their box batches through the tensor-backed path."""
    image = np.stack([
        np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3),
        np.full((8, 10, 3), 255, dtype=np.uint8),
    ])
    boxes = np.array(
        [
            [[1.0, 1.0, 4.0, 5.0]],
            [[2.0, 2.0, 5.0, 6.0]],
        ],
        dtype=np.float32,
    )
    pipe = Compose(
        [albu.Affine(translate_px={"x": 2, "y": -1}, p=1.0)],
        data_keys=["input", "bbox_xyxy"],
        execution="cv2",
    )

    output = pipe(image=image, bboxes=boxes)

    expected_boxes = boxes + np.array([2.0, -1.0, 2.0, -1.0], dtype=np.float32)
    assert output["image"].shape == image.shape
    assert output["image"].dtype == image.dtype
    np.testing.assert_allclose(output["bboxes"], expected_boxes, rtol=0.0, atol=1e-5)


def test_compose_rejects_a_keypoint_flip_cycle() -> None:
    """A mirror schema must contain only fixed points and two-slot pairs."""
    with pytest.raises(ValueError, match=r"keypoint_flip_index.*involut"):
        Compose.from_params(
            hflip_p=1.0,
            data_keys=["input", "keypoints"],
            keypoint_flip_index=(1, 2, 0),
        )


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required for mixed reflected inverse coverage")
def test_inverse_restores_ordered_keypoints_for_mixed_reflection_batch() -> None:
    """Inverse applies the pair table only to the reflected sample and restores slot order."""
    torch.manual_seed(0)
    image = _image()
    keypoints = torch.tensor([
        [[1.0, 1.0], [8.0, 4.0]],
        [[2.0, 2.0], [7.0, 5.0]],
    ])
    pipe = Compose(
        [
            kornia_aug.RandomHorizontalFlip(p=0.5, same_on_batch=False),
            kornia_aug.RandomRotation(degrees=(10.0, 10.0), p=1.0, same_on_batch=False),
        ],
        data_keys=["input", "keypoints"],
        keypoint_flip_index=(1, 0),
    )

    (augmented_image, augmented_keypoints), matrix = pipe(image, keypoints, return_matrix=True)
    recovered_image, recovered_keypoints = pipe.inverse(augmented_image, augmented_keypoints, matrix=matrix)

    assert matrix is not None
    determinants = torch.linalg.det(matrix[:, :2, :2])
    assert bool((determinants < 0).any())
    assert bool((determinants > 0).any())
    assert recovered_image.shape == image.shape
    torch.testing.assert_close(recovered_keypoints, keypoints, rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="torchvision required for CPU fast-path autograd coverage")
@pytest.mark.parametrize(
    ("batch_size", "with_mask"),
    [
        pytest.param(1, False, id="b1-image-only"),
        pytest.param(2, False, id="b2-image-only-control"),
        pytest.param(1, True, id="b1-mask-control"),
    ],
)
def test_torchvision_multi_affine_preserves_input_gradient(
    batch_size: int,
    with_mask: bool,
) -> None:
    """Fixed two-affine CPU calls retain autograd for B=1 and established control routes."""
    image = _image(batch_size).detach().requires_grad_()
    transforms = [
        torchvision_v2.RandomAffine(degrees=(13.0, 13.0), translate=(0.0, 0.0), scale=(1.1, 1.1)),
        torchvision_v2.RandomAffine(degrees=(7.0, 7.0), translate=(0.0, 0.0), scale=(0.9, 0.9)),
    ]
    pipe = Compose(transforms, data_keys=["input", "mask"] if with_mask else None)

    output = pipe(image, torch.zeros_like(image[:, :1]))[0] if with_mask else pipe(image)
    assert output.requires_grad
    output.sum().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="torchvision required for public hook coverage")
@pytest.mark.parametrize(
    ("transforms", "scenario"),
    [
        pytest.param(lambda: [torchvision_v2.RandomHorizontalFlip(p=1.0)], "exact", id="exact"),
        pytest.param(lambda: [torchvision_v2.RandomAffine(degrees=(0.0, 0.0))], "fused", id="fused"),
        pytest.param(
            lambda: [
                torchvision_v2.RandomAffine(degrees=(0.0, 0.0)),
                torchvision_v2.RandomAffine(degrees=(0.0, 0.0)),
            ],
            "general",
            id="general",
        ),
    ],
)
def test_compose_honors_result_changing_public_hooks(transforms, scenario: str) -> None:
    """Exact, fused, and general tensor routes each dispatch one public pre/post hook."""
    image = torch.zeros(1, 3, 8, 10)
    pipe = Compose(transforms())
    events: list[str] = []

    def add_one(_module, args):
        events.append("pre")
        return (args[0] + 1.0,)

    def double_output(_module, _args, output):
        events.append("post")
        return output * 2.0

    pre_handle = pipe.register_forward_pre_hook(add_one)
    post_handle = pipe.register_forward_hook(double_output)
    try:
        output = pipe(image)
    finally:
        pre_handle.remove()
        post_handle.remove()

    assert scenario in {"exact", "fused", "general"}
    assert events == ["pre", "post"]
    torch.testing.assert_close(output, torch.full_like(image, 2.0))


@pytest.mark.parametrize("call_style", [pytest.param("keyword", id="keyword"), pytest.param("mixed", id="mixed")])
def test_multi_target_tensor_calls_honor_dict_result_replacement_hooks(call_style: str) -> None:
    """Tensor keyword and mixed calls preserve the public hook wrapper around dict outputs."""
    image = torch.zeros(1, 3, 8, 10)
    mask = torch.ones(1, 1, 8, 10)
    pipe = Compose.from_params(hflip_p=1.0, data_keys=["input", "mask"])
    events: list[str] = []

    def replace_image(_module, _args, output):
        events.append("post")
        assert isinstance(output, dict)
        return {**output, "image": torch.full_like(output["image"], 7.0)}

    handle = pipe.register_forward_hook(replace_image)
    try:
        output = pipe(image=image, mask=mask) if call_style == "keyword" else pipe(image, mask=mask)
    finally:
        handle.remove()

    assert events == ["post"]
    torch.testing.assert_close(output["image"], torch.full_like(image, 7.0))
    torch.testing.assert_close(output["mask"], mask)


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations required for passthrough policy coverage")
def test_unknown_spatial_passthrough_with_targets_fails_before_image_mutation() -> None:
    """An unclassified coordinate-changing transform must not execute beside targets."""

    class UnknownSpatialShift(albu.ImageOnlyTransform):
        """Move every image column while retaining a call counter for the refusal oracle."""

        def __init__(self) -> None:
            super().__init__(p=1.0)
            self.calls = 0

        def apply(self, image: np.ndarray, **params: object) -> np.ndarray:
            self.calls += 1
            return np.roll(image, 1, axis=1)

    transform = UnknownSpatialShift()
    image = _image(batch_size=1)
    original = image.clone()
    mask = torch.zeros(1, 1, 8, 10)
    with pytest.warns(UserWarning, match="Unknown Albumentations transform"):
        pipe = Compose([transform], data_keys=["input", "mask"])

    with pytest.raises(ValueError, match=r"(Unknown|unclassified).*auxiliary"):
        pipe(image, mask)

    assert transform.calls == 0
    torch.testing.assert_close(image, original)


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="albumentations required for passthrough policy coverage")
def test_known_gaussian_blur_keeps_auxiliary_mask_and_changes_image() -> None:
    """A registered coordinate-preserving blur remains usable with target routing."""
    image = torch.zeros(1, 3, 8, 10)
    image[..., 4, 5] = 1.0
    mask = torch.zeros(1, 1, 8, 10)
    mask[..., 3:6, 4:7] = 2.0
    pipe = Compose([albu.GaussianBlur(blur_limit=(3, 3), p=1.0)], data_keys=["input", "mask"])

    output, output_mask = pipe(image, mask)

    assert not torch.equal(output, image)
    torch.testing.assert_close(output_mask, mask, rtol=0.0, atol=0.0)
