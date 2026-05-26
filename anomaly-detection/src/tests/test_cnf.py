import pytest
import torch

from src.anomaly_detection.layers.cnf import VectorFieldMLP, CNF


class TestVectorFieldMLP:
    def test_output_shape(self):
        mlp = VectorFieldMLP(d_model=256, time_emb_dim=32, hidden_dim=256)
        z_and_t = torch.randn(4, 288)  # 256 + 32
        v = mlp(z_and_t)
        assert v.shape == (4, 256)

    def test_time_embedding_used(self):
        """Verify that different t values give different outputs."""
        mlp = VectorFieldMLP(d_model=256, time_emb_dim=32, hidden_dim=256)
        z = torch.randn(4, 256)

        t0 = torch.zeros(4, 1)
        t1 = torch.ones(4, 1)

        t0_emb = mlp.net[0].weight[:, :256]  # just need to go through full forward
        # Simpler: directly call with CNF's pattern
        from src.anomaly_detection.layers.time_embedding import SinusoidalTimeEmbedding
        time_emb = SinusoidalTimeEmbedding(dim=32)
        t0_emb = time_emb(t0)
        t1_emb = time_emb(t1)
        assert not torch.allclose(t0_emb, t1_emb)

        # Now test full vector field with time embedding
        cnf = CNF(d_model=256, time_emb_dim=32, hidden_dim=256)
        v0 = cnf(z, t0)
        v1 = cnf(z, t1)
        assert not torch.allclose(v0, v1, atol=1e-4)

    def test_different_z_give_different_v(self):
        cnf = CNF(d_model=256, time_emb_dim=32, hidden_dim=256)
        z1 = torch.randn(4, 256)
        z2 = torch.randn(4, 256)
        t = torch.ones(4, 1) * 0.5
        v1 = cnf(z1, t)
        v2 = cnf(z2, t)
        assert not torch.allclose(v1, v2)

    def test_no_nan_output(self):
        mlp = VectorFieldMLP(d_model=256, time_emb_dim=32, hidden_dim=256)
        z_and_t = torch.randn(4, 288)
        v = mlp(z_and_t)
        assert not torch.isnan(v).any()


class TestCNF:
    def test_forward_single_time(self):
        cnf = CNF(d_model=256, time_emb_dim=32, hidden_dim=256)
        z = torch.randn(4, 256)
        t = torch.rand(4, 1)
        v = cnf(z, t)
        assert v.shape == (4, 256)

    def test_ode_func_shape(self):
        cnf = CNF(d_model=256, time_emb_dim=32, hidden_dim=256)
        t = 0.5
        x = torch.randn(4, 256)
        dxdt = cnf.ode_func(t, x)
        assert dxdt.shape == (4, 256)

    def test_forward_ode_euler(self):
        cnf = CNF(d_model=256, time_emb_dim=32, hidden_dim=256)
        z = torch.randn(4, 256)
        z_final, log_det = cnf.forward_ode(z, t_start=1.0, t_end=0.0, method="euler", num_steps=50)
        assert z_final.shape == (4, 256)
        assert log_det.shape == (4,)

    def test_forward_ode_shapes_preserved(self):
        cnf = CNF(d_model=256, time_emb_dim=32, hidden_dim=256)
        z = torch.randn(2, 256)
        z_final, log_det = cnf.forward_ode(z, t_start=1.0, t_end=0.0, method="euler", num_steps=20)
        assert z_final.shape == z.shape
        assert log_det.shape[0] == z.shape[0]

    def test_log_likelihood_finite(self):
        cnf = CNF(d_model=256, time_emb_dim=32, hidden_dim=256)
        z = torch.randn(4, 256)
        ll, z_noise = cnf.compute_log_likelihood(z)
        assert ll.shape == (4,)
        assert not torch.isnan(ll).any()
        assert not torch.isnan(z_noise).any()

    def test_log_likelihood_different_for_different_z(self):
        cnf = CNF(d_model=256, time_emb_dim=32, hidden_dim=256)
        z1 = torch.randn(4, 256)
        z2 = torch.randn(4, 256)
        ll1, _ = cnf.compute_log_likelihood(z1)
        ll2, _ = cnf.compute_log_likelihood(z2)
        assert not torch.allclose(ll1, ll2)

    def test_d_model_varied(self):
        for d in [64, 128, 256]:
            cnf = CNF(d_model=d, time_emb_dim=32, hidden_dim=128)
            z = torch.randn(4, d)
            t = torch.rand(4, 1)
            v = cnf(z, t)
            assert v.shape == (4, d)

    def test_time_embedding_in_cnf(self):
        cnf = CNF(d_model=128, time_emb_dim=32, hidden_dim=256)
        assert hasattr(cnf, "time_embedding")

    def test_forward_ode_z_change_during_integration(self):
        """ODE should change z — noise should be different from data after integration."""
        cnf = CNF(d_model=256, time_emb_dim=32, hidden_dim=256)
        z_data = torch.randn(4, 256)
        z_noise, _ = cnf.forward_ode(z_data, t_start=1.0, t_end=0.0, method="euler", num_steps=50)
        # After integration to t=0, z should be different from original z_data
        # (unless the vector field is exactly zero, which is unlikely)
        diff = (z_noise - z_data).abs().mean()
        assert diff > 1e-4 or not torch.isclose(z_noise, z_data).all()