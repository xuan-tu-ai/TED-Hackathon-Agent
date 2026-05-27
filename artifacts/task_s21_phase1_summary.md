# Task Summary: Phase 1 — SQL Seed Scripts for Databricks

**Sprint:** Sprint 21
**Tasks:** T12–T13

## Summary of Work
Created two Databricks SQL seed scripts under `data/sql/` for populating Delta Lake tables with welding robot data. `seed_databricks_pm_cards.sql` contains DDL + 45 robot-specific PM cards (≥10 per robot across R-00–R-03) and 5 shared fleet cards covering all 3 sensors. `seed_databricks_error_tables.sql` contains DDL + 2,880 perception result rows (30 days × 4 robots × 24 hours) with a single wire_feed_speed spike anomaly at R-00 day 5 hour 0, plus 20 maintenance event records (5 per robot).

## Files Created
- [data/sql/seed_databricks_pm_cards.sql](data/sql/seed_databricks_pm_cards.sql) — PM Cards DDL + 50 rows (45 robot-specific + 5 shared)
- [data/sql/seed_databricks_error_tables.sql](data/sql/seed_databricks_error_tables.sql) — PerceptionResults DDL + 2,880 rows, MaintenanceEvents DDL + 20 rows

## Testing
No tests — SQL scripts are run directly in Databricks SQL Editor by the user.

## Additional Notes
Run `seed_databricks_pm_cards.sql` first, then `seed_databricks_error_tables.sql`. Both scripts include verification SELECT queries at the end. The single anomaly is at R-00, day 5, hour 0 — wire_feed_speed spike with abnormal_score=0.88, severity_hint='critical'.
