---
name: correlate_history
description: Correlate current incident with historical PM Cards and maintenance events
required_tools: [query_pm_cards]
---

# Correlate History Skill

## Purpose
Search historical PM Cards and maintenance records to find similar past incidents and their resolutions.

## When to Use
- After loading a diagnose skill
- Before generating the final report
- To find evidence of similar root causes

## Diagnostic Steps

### Step 1: Identify Key Search Terms
Based on the incident:
- abnormal_family + sensor type
- Key symptoms from the diagnosis
- Robot code for robot-specific issues

### Step 2: Search PM Cards
Use `query_pm_cards` to search for:
- Similar sensor anomalies
- Same robot code
- Same failure mode keywords

### Step 3: Analyze Results
- Find PM Cards with matching symptoms
- Note the root causes and resolutions from past incidents
- Identify patterns (e.g., same issue recurs every X days)

### Step 4: Update Confidence
- High match with history = higher confidence in root cause
- No history = lower confidence, flag for review

## Completion
Return to the diagnose skill to incorporate findings into the report.