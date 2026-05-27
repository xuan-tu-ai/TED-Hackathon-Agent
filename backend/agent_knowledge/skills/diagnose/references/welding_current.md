# Sensor Reference: welding_current

## Overview
The `welding_current` sensor measures the electrical current flowing through the welding circuit (typically 150–400 A for GMAW). Current stability directly affects arc stability, penetration depth, and spatter generation. Anomalies in this sensor may indicate power supply problems, arc instability, or ground path issues.

## Known Failure Modes

| Failure Mode | Description | Signal Pattern |
|---|---|---|
| Arc Instability | Fluctuating arc length due to wire feed / voltage mismatch | High-frequency oscillation ±20–50 A around setpoint |
| Power Supply Ripple | Rectifier filter capacitor degradation | 100/120 Hz ripple superimposed on DC current |
| Ground Fault | High-resistance ground path or missing ground connection | Intermittent dips or elevated baseline current |
| Torch Contamination | Spatter buildup inside gas nozzle / contact tip | Random spike bursts above rated current |
| Cable Inductance Mismatch | Welding cable length/ gauge mismatch with power source tuning | Ringing transients at arc ignition / extinction |
| Shunt / Hall Effect Sensor Drift | Current transducer calibration drift | Consistent offset (positive or negative) in reading |
| Primary Power Fluctuation | Line voltage sag or phase loss | Brownout dips: current drops 20–50 % for 1–5 cycles |

## Observable Signal Patterns

1. **High-frequency oscillation** — Current varies ±20–50 A at > 50 Hz above setpoint
2. **Spike bursts above rated current** — Brief (1–10 ms) peaks > 150 % of nominal current
3. **Brownout dips** — Full-wave dips to 50–80 % of nominal for 1–5 AC cycles
4. **100/120 Hz ripple** — Sinusoidal modulation at twice line frequency
5. **Ringing transients** — Damped oscillation at arc start/stop (10–50 kHz)
6. **Random intermittent dips** — Brief drops to near-zero, no periodic pattern

## Normal Operating Range
- GMAW steady-state: 150–350 A (depending on wire diameter and feed speed)
- Peak during short-circuit transfer: up to 500 A (< 5 ms duration)
- Ripple: < 5 % of setpoint at rated output

## Inspection Checklist

1. [ ] Verify ground clamp connection — clean, tight, low resistance
2. [ ] Inspect welding torch — check for spatter buildup in nozzle and on contact tip
3. [ ] Check primary power input voltage — measure L-L and L-N at source panel
4. [ ] Inspect welding cables for damage, cuts, or excessive length
5. [ ] Listen for power supply fan noise — overheating may cause derating
6. [ ] Verify current feedback sensor calibration per manufacturer procedure
7. [ ] Check capacitor bank health — visual (bulging/leaking) or ESR measurement
8. [ ] Inspect shunt / Hall Effect sensor mounting and wiring
9. [ ] Test arc voltage — compare with commanded voltage on weld schedule

## PM Card Keywords
- `welding current`, `welding_current`, `arc current`
- `arc instability`, `ripple`, `power supply`
- `ground fault`, `torch contamination`, `spatter`
- `cable inductance`, `shunt`, `hall effect`
- `brownout`, `line sag`, `phase loss`, `capacitor`

## Severity Guidelines

| Severity | Criteria |
|---|---|
| LOW | Single spike < 150 % of nominal, < 1 ms duration, isolated |
| MEDIUM | Recurring spikes > 3 per minute, or oscillation > ±30 A sustained |
| HIGH | Brownout dips to < 50 %, or ripple > 10 % of setpoint |
| CRITICAL | Sustained oscillation > ±50 A causing visible arc extinction, or ground fault detected |

## Recommended Actions

### Short-Term
- Clean torch nozzle and replace contact tip
- Tighten and clean ground clamp connection
- Reduce welding parameters (current / wire feed speed) temporarily
- Verify primary power quality (check supply voltage with scope)

### Long-Term
- Replace power supply filter capacitors at first sign of ripple
- Upgrade welding cable to correct gauge for cable length
- Install line reactor / power conditioner for primary power stability
- Recalibrate or replace current feedback transducer annually
- Schedule preventive maintenance on torch — clean gas diffuser, replace gas hose if cracked
