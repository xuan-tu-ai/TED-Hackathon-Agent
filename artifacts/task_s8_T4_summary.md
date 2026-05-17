# Task Summary: T4 — Document revin, patchify, preprocessing, backbone, bottleneck

**Sprint:** Sprint 8
**Task:** T4

## Summary of Work
Added comprehensive Google Style docstrings to all 5 files in `layers/`. Each file has a module-level docstring explaining its role in the architecture, and every class/method has full Args/Returns sections with shape annotations.

## Files Modified
- [backend/app/anomaly_detection/layers/revin.py](../../backend/app/anomaly_detection/layers/revin.py) — Module docstring; `GlobalRevIN`, `PerPatchRevIN` already well-documented; added module-level overview
- [backend/app/anomaly_detection/layers/patchify.py](../../backend/app/anomaly_detection/layers/patchify.py) — Already decent; added module docstring
- [backend/app/anomaly_detection/layers/preprocessing.py](../../backend/app/anomaly_detection/layers/preprocessing.py) — Major rewrite: module-level docstring explaining V1–V6 vs V7 split; full Google Style for `RevIN`, `LogScaler`, `GlobalStatsStandardizer`
- [backend/app/anomaly_detection/layers/backbone.py](../../backend/app/anomaly_detection/layers/backbone.py) — Full rewrite: module docstring explaining MLP-Mixer architecture; `MixerBlock` and `MLPMixerBackbone` fully documented
- [backend/app/anomaly_detection/layers/bottleneck.py](../../backend/app/anomaly_detection/layers/bottleneck.py) — Major rewrite: all 5 classes (`MultitaskVAEHead`, `FactorizedVectorQuantizerEMA`, `ResidualFactorizedVectorQuantizerEMA`, `LegacyFactorizedVectorQuantizerEMA`, `MultiTaskVQVAEHead`) documented with VAE vs VQ vs RVQ differences explained

## Testing
- **Verification:** Python AST parse on all 5 files — all OK
- **Status:** Passed

## Additional Notes
- Documented the key architectural difference: VAE reparameterisation trick vs VQ nearest-neighbour lookup vs Residual VQ multi-level quantisation
- Explained cosine distance (Factorized) vs Euclidean distance (Legacy) choice and backward-compat reasons
