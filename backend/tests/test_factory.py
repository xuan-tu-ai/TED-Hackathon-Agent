import pytest
import torch

from app.anomaly_detection.models.factory import create_model


def create_glc_flow_v7(metadata=None):
    return create_model("glc_flow_v7", metadata=metadata)


class TestCreateModel:
    def test_create_model_glc_flow_v7(self):
        model = create_model("glc_flow_v7")
        assert model is not None

    def test_create_model_with_metadata(self):
        meta = {"d_model": 64, "transformer_layers": 2}
        model = create_model("glc_flow_v7", metadata=meta)
        assert model.d_model == 64
        assert model.transformer_layers == 2

    def test_create_model_invalid_name(self):
        with pytest.raises(ValueError):
            create_model("invalid_model")


class TestCreateGLCFlowV7:
    def test_create_default(self):
        model = create_glc_flow_v7()
        assert model is not None

    def test_model_trainable(self):
        model = create_glc_flow_v7()
        x = torch.randn(2, 6, 240)
        out = model({"input": x})
        assert "z_pooled" in out

    def test_count_parameters(self):
        model = create_glc_flow_v7()
        total = sum(p.numel() for p in model.parameters())
        assert total > 0

    def test_get_metadata(self):
        model = create_glc_flow_v7()
        meta = model.get_metadata()
        assert meta["name"] == "GLCFlowV7"
        assert "d_model" in meta