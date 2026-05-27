import pytest
import torch

from src.anomaly_detection.models.base import BaseAnomalyDetectionModel
from src.anomaly_detection.models.factory import create_model
from src.anomaly_detection.models.glc_flow_v7 import GLCFlowV7


def make_model() -> GLCFlowV7:
    return create_model("glc_flow_v7")


class TestBaseAnomalyDetectionModel:
    def test_checkpoint_load_save(self, tmp_path):
        model = make_model()
        folder = str(tmp_path)

        model.save_manual_checkpoint(epoch=1, val_loss=0.5, folder_path=folder, is_best=True)

        import os
        files = os.listdir(folder)
        assert len(files) > 0

    def test_manual_checkpoint_best(self, tmp_path):
        model = make_model()
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()

        model.save_manual_checkpoint(epoch=2, val_loss=0.3, folder_path=str(ckpt_dir), is_best=True)

        import os
        files = os.listdir(str(ckpt_dir))
        assert any("best" in f for f in files)

    def test_model_eval_mode(self):
        model = make_model()
        model.eval()
        x = torch.randn(2, 6, 240)
        with torch.no_grad():
            out = model({"input": x})
        assert "z_pooled" in out
