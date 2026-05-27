"""Tiered memory package for robot_agent.

Exposes the four memory tiers: session-level turn logging (T1), context
compaction (T0), persistent project rules (T2), and semantic vector
look-ups (T3).
"""