---
name: check-fmea
description: Cross-reference the diagnosed anomaly against the FMEA (Failure Mode and Effects Analysis) knowledge base
required_tools: [query_pm_cards]
---

# Check FMEA Skill

## Purpose
Cross-reference the diagnosed anomaly and identified root cause against the FMEA knowledge base to validate the failure mode and identify known mitigation steps.

## When to Use
- After completing diagnosis and root cause analysis
- Before formulating final recommendations
- When severity is HIGH or CRITICAL

## Diagnostic Steps

### Step 1: Identify Failure Mode
Based on the diagnosed anomaly (family + subtype + root cause), determine the likely failure mode from the FMEA catalog.

### Step 2: Query FMEA Records
Use `query_pm_cards` to search for:
- "FMEA", "failure mode"
- Anomaly family + robot code
- Similar root cause keywords

### Step 3: Compare with FMEA Database
- Check if the failure mode is already documented
- Note the RPN (Risk Priority Number) if available
- Identify recommended mitigation actions

### Step 4: Update Risk Assessment
- If FMEA exists: Validate diagnosis against known pattern
- If no FMEA: Flag for FMEA review/update

## Completion
Return findings to the main diagnose skill for inclusion in the final report.

> HITL rules are enforced system-wide. See `agent_knowledge/rules/hitl-rules.md` for the full action severity table.
