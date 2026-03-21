# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `SegmentDescriptor` frozen dataclass (`@dataclass(frozen=True, slots=True)`) to `fuse_augmentations._types` for structured, machine-readable plan introspection. Fields: `kind: str`, `transforms: tuple[str, ...]`, `n_warps_saved: int`, `backend: str | None`. Provides a `to_dict()` method returning a JSON-serialisable `dict[str, object]`.
- Added `fusion_plan_descriptors` property to `FusedCompose` returning `list[SegmentDescriptor]` — one descriptor per pipeline segment. The existing `fusion_plan` string property is unchanged.
- Implemented `ReorderPolicy.AGGRESSIVE`: multi-pass bubble-sort variant of `reorder_pointwise`. Extends the fusion window maximally by moving all `POINTWISE` ops across multiple geometric-stretch boundaries in repeated passes until no further movement is possible. Previously raised `NotImplementedError`.
- Extended `GEOMETRIC_EXACT` dispatch: `RandomRotation90` (Kornia), `RandomRotate90` / `D4` / `Transpose` (Albumentations) are now registered as `GEOMETRIC_EXACT` and fused losslessly via `tensor.flip` chains rather than `grid_sample`.

### Changed

- Renamed `ExactSegment` to `ExactAffineSegment` for naming consistency with `FusedAffineSegment`. The old name is available as a deprecated alias and will be removed in v0.9.

### Deprecated

- `ExactSegment`: use `ExactAffineSegment` instead. Deprecated since v0.7, removal planned for v0.9.
