"""Integration regressions for exact-operation matrices used in inference workflows."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fuse_augmentations import Compose
from fuse_augmentations._compat import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE
from fuse_augmentations.targets import transform_keypoints

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as A

    from fuse_augmentations.adapters.albumentations import _D4_ELEM_TO_CODE, AlbumentationsAdapter

if _KORNIA_AVAILABLE:
    import kornia.augmentation as K

    from fuse_augmentations.adapters.kornia import KorniaAdapter


pytestmark = pytest.mark.integration


def _marked_image(batch_size: int = 1, height: int = 7, width: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unique marker images and their image-centre coordinates."""
    image = torch.zeros(batch_size, 1, height, width)
    points = torch.empty(batch_size, 1, 2)
    for index in range(batch_size):
        x = float((index + 2) % width)
        y = float((index + 1) % height)
        image[index, 0, int(y), int(x)] = float(index + 1)
        points[index, 0] = torch.tensor([x, y])
    return image, points


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
def test_compose_singleton_rotation90_returns_matrix_matching_marked_pixel() -> None:
    """A singleton exact Kornia quarter turn exposes the matrix that moved its marker."""
    image = torch.arange(49, dtype=torch.float32).reshape(1, 1, 7, 7)
    source_points = torch.tensor([[[5.0, 1.0]]])
    pipe = Compose([K.RandomRotation90(times=(1, 1), p=1.0)])

    output, matrix = pipe(image, return_matrix=True)

    assert torch.equal(output, torch.rot90(image, 1, dims=[2, 3]))
    assert matrix is not None
    assert pipe.transform_matrix is not None
    torch.testing.assert_close(matrix, pipe.transform_matrix)
    mapped_points = transform_keypoints(source_points, matrix)
    mapped_x, mapped_y = mapped_points[0, 0].round().to(torch.int64).tolist()
    assert output[0, 0, mapped_y, mapped_x].item() == image[0, 0, 1, 5].item()


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
def test_compose_kornia_probability_batch_matrix_matches_each_marker() -> None:
    """Per-sample flip probabilities publish identity or mirror matrices for their own markers."""
    image, source_points = _marked_image(batch_size=4, height=3, width=5)
    pipe = Compose([K.RandomHorizontalFlip(p=0.5), K.RandomVerticalFlip(p=0.0)])
    torch.manual_seed(4)

    output, matrix = pipe(image, return_matrix=True)

    assert matrix is not None
    mapped_points = transform_keypoints(source_points, matrix).round().to(torch.int64)
    for index, (mapped_x, mapped_y) in enumerate(mapped_points[:, 0].tolist()):
        assert output[index, 0, mapped_y, mapped_x].item() == float(index + 1)
    assert torch.equal(matrix[:, 0, 0] == -1.0, torch.tensor([False, False, True, True]))


@pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="missing kornia")
def test_kornia_rotation90_params_match_per_batch_pixels_and_matrix() -> None:
    """Provided Kornia k90 values drive each exact image row and its forward matrix."""
    image, source_points = _marked_image(batch_size=2)
    transform = K.RandomRotation90(times=(0, 3), p=1.0)
    params = {
        "_batch_size": torch.tensor([2], dtype=torch.int64),
        "k90": torch.tensor([1, 3], dtype=torch.int64),
    }

    output = KorniaAdapter.exact_apply(transform, image, params=params)
    matrix = KorniaAdapter.build_matrix(transform, params, height=7, width=7)

    expected = torch.stack([torch.rot90(image[0], 1, dims=[1, 2]), torch.rot90(image[1], 3, dims=[1, 2])])
    assert torch.equal(output, expected)
    mapped_points = transform_keypoints(source_points, matrix).round().to(torch.int64)
    for index, (mapped_x, mapped_y) in enumerate(mapped_points[:, 0].tolist()):
        assert output[index, 0, mapped_y, mapped_x].item() == float(index + 1)


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
def test_albumentations_d4_params_match_asymmetric_rectangular_pixels_and_matrix() -> None:
    """Provided D4 codes control exact pixels and source-to-output matrices without a redraw."""
    image, source_points = _marked_image(batch_size=2, height=3, width=5)
    transform = A.D4(p=1.0)
    params = {
        "_batch_size": torch.tensor([2], dtype=torch.int64),
        "d4_code": torch.tensor([_D4_ELEM_TO_CODE["h"], _D4_ELEM_TO_CODE["v"]], dtype=torch.int64),
    }

    output = AlbumentationsAdapter.exact_apply(transform, image, params=params)
    matrix = AlbumentationsAdapter.build_matrix(transform, params, height=3, width=5)

    expected = torch.stack([image[0].flip(dims=[2]), image[1].flip(dims=[1])])
    assert torch.equal(output, expected)
    mapped_points = transform_keypoints(source_points, matrix).round().to(torch.int64)
    for index, (mapped_x, mapped_y) in enumerate(mapped_points[:, 0].tolist()):
        assert output[index, 0, mapped_y, mapped_x].item() == float(index + 1)


@pytest.mark.skipif(not _ALBUMENTATIONS_AVAILABLE, reason="missing albumentations")
def test_compose_albumentations_numpy_rotation90_returns_matrix_matching_marked_pixel() -> None:
    """Native NumPy exact calls return the actual matrix alongside the restored image dict."""
    image = np.zeros((7, 7, 1), dtype=np.uint8)
    image[1, 5, 0] = 255
    transform = A.RandomRotate90(p=1.0)
    transform.set_random_seed(17)
    torch.manual_seed(17)
    pipe = Compose([transform])

    result, matrix = pipe(image=image, return_matrix=True)

    assert matrix is not None
    assert pipe.transform_matrix is not None
    torch.testing.assert_close(matrix, pipe.transform_matrix)
    assert matrix.dtype == torch.float32
    source_point = torch.tensor([[[5.0, 1.0]]], dtype=torch.float32)
    mapped_x, mapped_y = transform_keypoints(source_point, matrix)[0, 0].round().to(torch.int64).tolist()
    assert isinstance(result, dict)
    assert result["image"][mapped_y, mapped_x, 0] == 255
