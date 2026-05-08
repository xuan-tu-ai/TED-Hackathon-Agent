import pytest
import torch

from app.anomaly_detection.layers.patchify import (
    Patchify,
    FlattenPatches,
    unfold_to_patches,
    flatten_to_tokens,
)


class TestPatchify:
    def test_output_shape(self):
        patchify = Patchify(patch_size=32, patch_stride=16)
        x = torch.randn(4, 6, 240)
        patches = patchify(x)
        assert patches.shape == (4, 6, 14, 32)

    def test_num_patches_correct(self):
        patchify = Patchify(patch_size=32, patch_stride=16)
        x = torch.randn(2, 3, 240)
        patches = patchify(x)
        assert patches.shape[2] == 14  # (240-32)//16 + 1 = 14

    def test_different_seq_len(self):
        patchify = Patchify(patch_size=16, patch_stride=8)
        x = torch.randn(2, 3, 128)
        patches = patchify(x)
        expected_patches = (128 - 16) // 8 + 1
        assert patches.shape == (2, 3, expected_patches, 16)

    def test_unfold_functional(self):
        patches = unfold_to_patches(torch.randn(2, 6, 240), patch_size=32, patch_stride=16)
        assert patches.shape == (2, 6, 14, 32)

    def test_3d_input_required(self):
        patchify = Patchify(patch_size=32, patch_stride=16)
        x_2d = torch.randn(4, 240)  # missing channel dim
        with pytest.raises(ValueError, match="Expected 3D"):
            patchify(x_2d)


class TestFlattenPatches:
    def test_output_shape(self):
        flatten = FlattenPatches(patch_size=32)
        patches = torch.randn(4, 6, 14, 32)
        tokens = flatten(patches)
        assert tokens.shape == (4, 84, 32)  # 6*14=84, 32 unchanged

    def test_patch_size_varied(self):
        flatten = FlattenPatches(patch_size=16)
        patches = torch.randn(4, 3, 20, 16)
        tokens = flatten(patches)
        assert tokens.shape == (4, 60, 16)  # 3*20=60

    def test_flatten_functional(self):
        patches = torch.randn(2, 3, 10, 16)
        tokens = flatten_to_tokens(patches, d_half=32)
        assert tokens.shape == (2, 30, 32)  # functional version still projects

    def test_reshape_only(self):
        flatten = FlattenPatches(patch_size=32)
        patches = torch.randn(4, 6, 14, 32)
        tokens = flatten(patches)
        assert tokens.shape[-1] == 32  # last dim preserved (no projection in class)