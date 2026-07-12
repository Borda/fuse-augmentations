---
title: Advanced API
description: Low-level segments, protocols, segmentation helpers, and experimental extension boundaries.
---

# Advanced API

Most applications should use `Compose`. The objects on this page expose implementation-level building blocks for research, inspection, and controlled integrations. Their contracts are narrower and more likely to evolve than the core pipeline API.

## Stability levels

| Level                 | Surface                                                                      | Guidance                                                          |
| --------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Public                | `Compose`, `TransformSpec`, descriptors, converters, target functions, enums | Preferred application API                                         |
| Advanced              | Segment classes, adapter/converter protocols, `build_segments`               | Use when the low-level contract is understood and pinned by tests |
| Experimental/internal | `_backend.register_adapter`, entry-point loading, private segment subclasses | Do not treat as stable production extension API                   |

Segment objects are `torch.nn.Module` implementations, but the top-level pipeline directly dispatches their `forward` methods for speed. Forward hooks registered on individual segment modules are therefore bypassed; register hooks on the pipeline or execute the segment directly when hook observation is required.

## Geometric segments

### `FusedAffineSegment`

Composes one compatible affine/exact run. Its adapter must classify transforms, sample canonical parameters, and build forward pixel matrices.

::: fuse_augmentations.FusedAffineSegment
    options:
        show_root_heading: true
        show_source: false

### `ExactAffineSegment`

Executes an exact-only run without interpolation. Supported discrete operations and auxiliary-target behavior depend on the adapter.

::: fuse_augmentations.ExactAffineSegment
    options:
        show_root_heading: true
        show_source: false

### `ProjectiveSegment`

Composes consecutive projective homographies. It is a separate segment from adjacent affine geometry.

::: fuse_augmentations.ProjectiveSegment
    options:
        show_root_heading: true
        show_source: false

### `CropResizeSegment`

Runs a standalone fixed-output crop-resize. On torch paths, an immediately preceding affine run may instead be represented by a private combined geo-crop segment; that private implementation is intentionally not an API object.

::: fuse_augmentations.CropResizeSegment
    options:
        show_root_heading: true
        show_source: false

## Color segments

`FusedColorSegment` accepts transforms that the adapter can express as homogeneous (4 \\times 4) color matrices. It cannot represent nonlinear saturation/hue or arbitrary intermediate clamps.

::: fuse_augmentations.FusedColorSegment
    options:
        show_root_heading: true
        show_source: false

## Segmentation helper

`build_segments` is exported for advanced use but is an implementation-oriented planner. It expects an adapter instance and returns a heterogeneous list of segment modules and passthrough transforms. Prefer `Compose` unless a custom planner is being tested directly.

::: fuse_augmentations.build_segments
    options:
        show_root_heading: true
        show_source: false

## Protocols

### Transform adapters

The protocol describes classification, parameter sampling, matrix building, exact execution, color-matrix building, and native passthrough. Conformance alone does not register an adapter with `Compose`.

::: fuse_augmentations.TransformAdapter
    options:
        show_root_heading: true
        show_source: false

### Backend converters

Converters translate the pipeline's primary output representation. Coordinate-target conversion is deliberately separate from image layout conversion.

::: fuse_augmentations.BackendConverter
    options:
        show_root_heading: true
        show_source: false

## Experimental adapter registration

The private `fuse_augmentations._backend` module contains `register_adapter` and lazy loading for the `fuse_augmentations.adapters` entry-point group. This surface is explicitly experimental.

The current registry provides detection metadata. A default third-party adapter registered as `Backend.UNKNOWN` is not yet an end-to-end `Compose` execution path: composition dispatch only constructs one of the three built-in adapters. A production plugin should wait for a public contract with an integration test covering registration, construction, forward execution, and serialization.

Do not map a third-party adapter to a built-in backend enum as a workaround. Composition would retrieve the built-in adapter, not the registered third-party instance.

## When to stay at the core API

Use `Compose` instead of direct segments when any of these are needed:

- mixed-backend routing;
- output conversion;
- `data_keys` validation and assembly;
- passthrough safety checks;
- structured plan descriptors;
- serialization compatibility;
- stable handling of future segment implementations.
