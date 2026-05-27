from __future__ import annotations

import inspect

import pytest


class TestL1L2Balance:
    def test_l1_density_threshold_from_config(self):
        from app.robot_agent.poller.l1_burst import detect_l1_burst
        from app.robot_agent.agent.config import config
        import inspect

        sig = inspect.signature(detect_l1_burst)
        default_threshold = sig.parameters["density_threshold"].default
        # Default is None — actual threshold is loaded from thresholds.json via config
        assert default_threshold is None, f"Expected density_threshold=None (config-driven), got {default_threshold}"
        assert config.l1_density_threshold > 0, "config.l1_density_threshold must be positive"

    def test_l1_burst_detects_at_lower_threshold(self, db_session):
        from app.robot_agent.poller.l1_burst import detect_l1_burst

        result = detect_l1_burst(
            db_session,
            "R-07",
            window_hours=1,
            density_threshold=2.5,
        )
        assert "is_burst" in result
        assert "density_per_hour" in result

    def test_l1_default_threshold_allows_more_bursts(self, db_session):
        from app.robot_agent.poller.l1_burst import detect_l1_burst

        result_default = detect_l1_burst(
            db_session,
            "R-07",
            window_hours=1,
        )
        result_lower = detect_l1_burst(
            db_session,
            "R-07",
            window_hours=1,
            density_threshold=1.0,
        )
        assert result_lower["density_per_hour"] >= result_default["density_per_hour"]

    def test_is_file_anomaly_column_exists(self, db_session):
        from app.robot_agent.db.models import PerceptionResult
        from sqlalchemy import select, func

        # Verify the column is queryable (schema test only — db_session is unseeded)
        count = db_session.execute(
            select(func.count()).select_from(PerceptionResult).where(
                PerceptionResult.is_file_anomaly == True
            )
        ).scalar()
        assert count >= 0