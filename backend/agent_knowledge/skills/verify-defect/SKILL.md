---
name: verify-defect
description: Verify if signal anomaly has resulted in actual product surface defects
required_tools: [query_db]
---

# Verify Weld Defect Skill

## Purpose
Determine whether a signal anomaly has caused actual weld defects on the workpiece. This separates signal-level anomalies from actual quality issues.

## When to Use
- After diagnosis confirms a significant anomaly
- For families: SURGE, DROPOUT, INSTABILITY, DEVIATION
- When severity is HIGH or CRITICAL

## Important
**Signal anomalies do NOT automatically mean weld defects.** Weld defects can only be confirmed through visual or NDT inspection.

## Diagnostic Steps

### Step 1: Evaluate Signal Anomaly Severity
- Check if anomaly score exceeds threshold for weld quality impact
- Determine duration and phase of anomaly within the welding cycle

### Step 2: Query Recent Inspections
- Use `query_db` to check if any recent welds from this robot have been flagged by QC
- Note: Weld inspection data may come from a separate NDT or visual inspection system

### Step 3: Assess Impact Risk
Based on anomaly type:
- **SURGE**: May cause spatter contamination, burn-through on thin material
- **DROPOUT**: May cause cold weld, incomplete fusion, porosity
- **INSTABILITY**: May cause irregular bead geometry, variable penetration
- **DEVIATION**: May cause heat-affected zone (HAZ) cracking, distortion

### Step 4: Recommend Verification
- If risk is HIGH: Recommend immediate workpiece inspection (visual + NDT)
- If risk is LOW: Note for next scheduled QC check

## Completion
Return to the diagnose skill with verification status to include in the report.