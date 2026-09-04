<!-- GENERATED FILE -- do not edit by hand.
     Regenerate with: python .github/scripts/measure_parity_tolerances.py --write-doc -->

# Measured fused-vs-native tolerances

Maximum absolute per-pixel difference between each backend's own compose and `fuse_augmentations.Compose` running the same deterministic transform. Every parameter range is collapsed to a single value and every probability is 1.0, so no sampling remains and the number is resampling and composition behaviour rather than two different random draws.

Images are float32 in `[0, 1]`, so a tolerance of `0.004` is roughly one step of an 8-bit level. These are measurements, not promises: they record what the current implementations do, and `.github/workflows/ci_parity-gate.yml` fails when one drifts past its recorded bound.

A single-op row compares one warp against one warp, so its difference is small by construction and a large value there would be a defect. A chain row compares one fused warp against one native warp per operation: resampling twice is not the same as resampling once, so a visible difference there is the fusion working as designed on a deliberately high-frequency test image. Both kinds still have to stay stable, which is what the gate checks -- the number, not its size, is the signal.

The prose companion to this table -- which surfaces are verified and where the boundaries are -- is `quality-and-fidelity.md`.

| Backend        | Operation           | Native passes | Max abs difference |
| -------------- | ------------------- | ------------- | ------------------ |
| albumentations | affine_chain        | 2             | 0.998238           |
| albumentations | brightness_contrast | 1             | 0.000000           |
| albumentations | hflip               | 1             | 0.000000           |
| albumentations | rotate              | 1             | 0.000005           |
| albumentations | scale               | 1             | 0.000007           |
| albumentations | translate           | 1             | 0.000000           |
| albumentations | vflip               | 1             | 0.000000           |
| kornia         | affine_chain        | 2             | 0.000008           |
| kornia         | hflip               | 1             | 0.000000           |
| kornia         | rotate              | 1             | 0.000000           |
| kornia         | vflip               | 1             | 0.000000           |
| torchvision    | affine_chain        | 2             | 0.701755           |
| torchvision    | hflip               | 1             | 0.000000           |
| torchvision    | rotate              | 1             | 0.000000           |
| torchvision    | vflip               | 1             | 0.000000           |
