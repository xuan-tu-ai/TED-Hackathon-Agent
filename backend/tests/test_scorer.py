import pytest
import torch

from app.anomaly_detection.layers.scorer import MADThreshold, AnomalyScorer, PatchMasker, PerturbationLoop
from app.anomaly_detection.layers.cnf import CNF


class TestMADThreshold:
    def test_fit_sets_threshold(self):
        threshold = MADThreshold(k=3.0)
        scores = torch.tensor([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 10.0])
        threshold.fit(scores)
        assert threshold.threshold is not None
        assert threshold.median is not None
        assert threshold.mad is not None

    def test_predict_above_threshold(self):
        threshold = MADThreshold(k=3.0)
        scores = torch.tensor([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])
        threshold.fit(scores)
        preds = threshold.predict(scores)
        assert preds.dtype == torch.bool
        assert (preds.float().sum() < len(scores))  # not all should be flagged

    def test_robust_to_outliers(self):
        """MAD should be robust — outliers shouldn't move threshold much."""
        clean = torch.tensor([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])
        contaminated = torch.tensor([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 100.0])

        t_clean = MADThreshold(k=3.0)
        t_clean.fit(clean)

        t_contam = MADThreshold(k=3.0)
        t_contam.fit(contaminated)

        assert abs(t_clean.threshold - t_contam.threshold) < 1.0

    def test_predict_requires_fit(self):
        threshold = MADThreshold()
        with pytest.raises(RuntimeError, match="fit"):
            threshold.predict(torch.tensor([0.5]))


class TestAnomalyScorer:
    def test_score_shape(self):
        cnf = CNF(d_model=256, time_emb_dim=32, hidden_dim=256)
        scorer = AnomalyScorer(cnf)
        z_pooled = torch.randn(4, 256)
        scores, z_noise = scorer(z_pooled)
        assert scores.shape == (4,)
        assert z_noise.shape == (4, 256)

    def test_score_higher_for_random_than_zero(self):
        cnf = CNF(d_model=256, time_emb_dim=32, hidden_dim=256)
        scorer = AnomalyScorer(cnf)

        z_random = torch.randn(4, 256) * 10
        z_zero = torch.zeros(4, 256)

        score_random, _ = scorer(z_random)
        score_zero, _ = scorer(z_zero)

        assert score_random.shape == (4,)
        assert score_zero.shape == (4,)


class TestPatchMasker:
    def test_mask_single_patch(self):
        masker = PatchMasker(num_sensors=6, num_patches=14)
        raw = torch.randn(4, 6, 240)
        masked = masker.mask(raw, sensor=2, patch=5)
        start = 5 * 16
        end = start + 32
        assert masked[:, 2, start:end].abs().sum() < 1e-6
        assert masked[:, :2, :].sum() == raw[:, :2, :].sum()  # other sensors unchanged
        assert masked[:, 3:, :].sum() == raw[:, 3:, :].sum()

    def test_all_masks_count(self):
        masker = PatchMasker(num_sensors=6, num_patches=14)
        raw = torch.randn(2, 6, 240)
        masks = masker.all_masks(raw)
        assert len(masks) == 84  # 6 * 14

    def test_all_masks_different(self):
        masker = PatchMasker(num_sensors=6, num_patches=14)
        raw = torch.randn(2, 6, 240)
        masks = masker.all_masks(raw)
        for i in range(len(masks) - 1):
            s1, p1, m1 = masks[i]
            s2, p2, m2 = masks[i + 1]
            if s1 == s2 and p1 == p2:
                assert torch.allclose(m1, m2)


class TestPerturbationLoop:
    def test_localize_shape(self):
        cnf = CNF(d_model=256, time_emb_dim=32, hidden_dim=256)

        class DummyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(6 * 240, 256)

            def forward(self, x):
                B = x.shape[0]
                return self.linear(x.flatten(1))

        encoder = DummyEncoder()
        scorer = AnomalyScorer(cnf)
        masker = PatchMasker(num_sensors=6, num_patches=14)
        loop = PerturbationLoop(encoder, cnf, scorer, masker)

        raw = torch.randn(1, 6, 240)  # batch=1 to simplify
        with torch.no_grad():
            heatmap = loop.localize(raw)
        assert heatmap.shape == (6, 14)

    def test_localize_batch_returns_list(self):
        cnf = CNF(d_model=256, time_emb_dim=32, hidden_dim=256)

        class DummyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(6 * 240, 256)

            def forward(self, x):
                return self.linear(x.flatten(1))

        encoder = DummyEncoder()
        scorer = AnomalyScorer(cnf)
        masker = PatchMasker(num_sensors=6, num_patches=14)
        loop = PerturbationLoop(encoder, cnf, scorer, masker)

        batch = torch.randn(2, 6, 240)  # batch=2
        with torch.no_grad():
            heatmaps = loop.localize_batch(batch)
        assert len(heatmaps) == 2
        for h in heatmaps:
            assert h.shape == (6, 14)