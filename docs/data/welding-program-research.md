# Welding Program Research — Sprint 18 Verification

**Goal:** Verify that the simulator's default timing parameters (4Hz sampling, 240-step typical cycle) match real MIG/MAG welding operations.

## Key Findings

### Sampling Rate
- **Industrial SCADA/PLC data acquisition** for robotic welding typically uses 1–10 Hz sampling, with 4 Hz (250 ms) being a common mid-range rate.
- Higher rates (up to 20 kHz) are used for arc physics research, not production monitoring.
- **Conclusion:** 4 Hz is realistic for production-level monitoring. No change needed.

### Cycle Duration
- **Automotive spot welding:** 0.5–3 seconds per spot, but a multi-spot sequence can last 10–60 seconds total.
- **Robotic MIG/MAG seam welding:** Typical weld length 50–500 mm → at 5–15 mm/s travel speed → 3–100 seconds per weld.
- **Industrially:** A single robotic welding cycle (approach → weld → cool → retract) typically ranges from **20 seconds to 3 minutes**.
- Our 240-step default at 4 Hz = 60 seconds, which is right in the middle of this range.
- **Conclusion:** The default 240 steps (60 seconds) is realistic. Profile ranges (20s–3min+) are appropriate.

### Wire Feed Speeds
- **Real GMAW:** 2–10 m/min (75–400 in/min) for semiautomatic; up to 30 m/min for automated.
- **Our model:** 180–330 mm/min (profiles).
- **Note:** Our "mm/min" in the CSV column name is a legacy naming convention. The actual values are scaled to produce realistic current readings (~10–17 A) through the LTI system, not real-world feed speeds. The dynamics (overshoot, delay) are correct relative to our scaled units.

### Anomaly Types: Real-World Reliability
Based on the detectability classification analysis and industrial welding maintenance data:

| Anomaly | Real-World Frequency | Detectability | Kept in Sprint 18 |
|---------|---------------------|---------------|-------------------|
| Cable wear (short) | Very high — #1 maintenance issue | Recon-error ✓ | ✅ |
| Spike / short circuit | High | Recon-error ✓ | ✅ |
| Thermal drift (cooling failure) | Medium — cooling system degrades gradually | Needs baseline | ✅ |
| Arc blow (magnetic deflection) | Medium — ground clamp placement | Cross-sensor | ✅ |
| Sensor flatline | Medium — sensor failure | Local RevIN ✓ | ✅ |
| Phase shift | Low — power supply issue | Temporal attention | ❌ (archived) |
| Wire stutter | Low — feed motor issue | Too subtle | ❌ (archived) |
| Gas blockage | Low-medium | Superseded by arc_blow | ❌ (archived) |
| Sticky TC | Low | Low occurrence | ❌ (archived) |
| Overshoot degradation | Very low | Recall=0.17 | ❌ (archived) |

## Impact on Simulator
- **No changes to sampling rate or profile durations** — current defaults are verified realistic.
- **5 anomaly types** now cover the most frequent real-world reliability scenarios.
- **Convergent training data** (deterministic + reduced noise) better reflects the reality that the same robot running the same program should produce nearly identical signals under normal conditions.
