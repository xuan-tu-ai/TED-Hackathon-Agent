from __future__ import annotations

import inspect

import pytest


class TestL1L2Balance:
    def test_l1_density_threshold_is_2_5(self):
        from app.robot_agent.poller.l1_burst import detect_l1_burst

        sig = inspect.signature(detect_l1_burst)
        default_threshold = sig.parameters["density_threshold"].default
        assert default_threshold == 2.5, f"Expected density_threshold=2.5, got {default_threshold}"

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

    def test_is_file_anomaly_threshold_in_seed(self, db_session):
        from app.robot_agent.db.models import PerceptionResult

        from sqlalchemy import select, func

        total = db_session.execute(
            select(func.count()).select_from(PerceptionResult)
        ).scalar()

        file_anomaly_count = db_session.execute(
            select(func.count()).select_from(PerceptionResult).where(
                PerceptionResult.is_file_anomaly == True
            )
        ).scalar()

        ratio = file_anomaly_count / total if total > 0 else 0

        assert ratio > 0.25, f"is_file_anomaly ratio too low: {ratio:.2%} (expected >25%)"