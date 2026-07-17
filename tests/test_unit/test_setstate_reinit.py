"""Cross-version unpickling rebuilds every derived dispatch attribute (CORE-1).

``FusedCompose._setup_instance`` derives roughly ten runtime dispatch attributes
(``_seg_dispatch_tags``, ``_multi_target``, ``_aux_keys``, the single-segment fast
paths, ``_is_albu_native``, ``_albu_seg_tags``, ``_output_converter``) from the core
pickled state. ``__setstate__`` previously rebuilt only three of them, so a pickle
produced before the others existed raised ``AttributeError`` on the first forward.
``__setstate__`` now calls the same ``_build_derived_state`` builder as construction,
so an older pickle (simulated here by stripping the derived attributes from the
restored state) is fully reconstructed.

Requires kornia.

"""

from __future__ import annotations

import pickle

import pytest
import torch

from fuse_augmentations import Compose, FusedCompose
from fuse_augmentations._compat import _KORNIA_AVAILABLE

if _KORNIA_AVAILABLE:
    import kornia.augmentation as kornia_aug

pytestmark = pytest.mark.skipif(not _KORNIA_AVAILABLE, reason="kornia required")

# Derived attributes rebuilt by ``_build_derived_state`` — absent from a pre-attr pickle.
_DERIVED_ATTRS = (
    "_seg_dispatch_tags",
    "_multi_target",
    "_aux_keys",
    "_single_exact_fast",
    "_single_fused_fast_seg",
    "_single_albu_direct_seg",
    "_is_albu_native",
    "_albu_seg_tags",
    "_output_converter",
)


def _image() -> torch.Tensor:
    """Return a deterministic ``(2, 3, 16, 16)`` float32 image batch."""
    gen = torch.Generator().manual_seed(0)
    return torch.rand(2, 3, 16, 16, generator=gen)


def _legacy_restore(pipe: FusedCompose, drop: tuple[str, ...]) -> FusedCompose:
    """Rebuild ``pipe`` from a state dict with ``drop`` attributes removed, as an old pickle would lack them."""
    state = dict(pipe.__dict__)
    for attr in drop:
        state.pop(attr, None)
    restored = FusedCompose.__new__(FusedCompose)
    restored.__setstate__(state)
    return restored


def test_pickle_roundtrip_forward_matches():
    """A standard pickle round-trip preserves the deterministic forward output."""
    pipe = Compose([kornia_aug.RandomHorizontalFlip(p=1.0)])
    image = _image()
    reloaded = pickle.loads(pickle.dumps(pipe))  # noqa: S301 -- trusted, self-produced bytes
    torch.testing.assert_close(reloaded(image), pipe(image))


def test_legacy_pickle_missing_derived_attrs_rebuilds_and_runs():
    """A pickle lacking every derived attribute is fully rebuilt and forwards without AttributeError."""
    pipe = Compose([kornia_aug.RandomHorizontalFlip(p=1.0)])
    image = _image()
    restored = _legacy_restore(pipe, _DERIVED_ATTRS)
    assert all(hasattr(restored, attr) for attr in _DERIVED_ATTRS)
    torch.testing.assert_close(restored(image), pipe(image))


def test_legacy_pickle_multi_target_rebuilt():
    """A multi-target pickle missing derived attrs rebuilds ``_multi_target``/``_aux_keys`` and routes the mask."""
    pipe = Compose([kornia_aug.RandomHorizontalFlip(p=1.0)], data_keys=["input", "mask"])
    image, mask = _image(), _image()[:, :1]
    restored = _legacy_restore(pipe, _DERIVED_ATTRS)
    assert restored._multi_target is True
    assert restored._aux_keys == ["mask"]
    out_image, out_mask = restored(image, mask)
    exp_image, exp_mask = pipe(image, mask)
    torch.testing.assert_close(out_image, exp_image)
    torch.testing.assert_close(out_mask, exp_mask)


def test_legacy_pickle_output_converter_rebuilt_from_backend_flag():
    """A pickle whose ``_output_converter`` is dropped rebuilds it from the retained ``_output_backend`` flag."""
    pipe = Compose([kornia_aug.RandomHorizontalFlip(p=1.0)], output_backend="numpy")
    image = _image()
    restored = _legacy_restore(pipe, ("_output_converter",))
    result = restored(image)
    # numpy output_backend converts to a channel-last ndarray; a lost converter would return a tensor.
    assert result.shape == (2, 16, 16, 3)
