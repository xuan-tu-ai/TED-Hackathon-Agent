---
name: verify-defect
description: Verify if signal anomaly has resulted in actual product surface defects
required_tools: [query_db]
---

# Verify Surface Defect Skill

## Purpose
Determine whether a signal anomaly has caused actual product defects on the painted surface. This separates signal-level anomalies from actual quality issues.

## When to Use
- After diagnosis confirms a significant anomaly
- For families: SURGE, DROPOUT, INSTABILITY
- When severity is HIGH or CRITICAL

## Important
**Signal anomalies do NOT automatically mean product defects.** Surface defects can only be confirmed through actual product inspection.

## Diagnostic Steps

### Step 1: Evaluate Signal Anomaly Severity
- Check if anomaly score exceeds threshold for product impact
- Determine duration and phase of anomaly within the cycle

### Step 2: Query Recent Products
- Use `query_db` to check if any recent products from this robot have been flagged
- Note: Product inspection data may come from a separate QC system

### Step 3: Assess Impact Risk
Based on anomaly type:
- **SURGE**: May cause orange peel, overspray
- **DROPOUT**: May cause incomplete coverage, holidays
- **INSTABILITY**: May cause texture defects, uneven coating

### Step 4: Recommend Verification
- If risk is HIGH: Recommend immediate product inspection
- If risk is LOW: Note for next scheduled QC check

## Completion
Return to the diagnose skill with verification status to include in the report.