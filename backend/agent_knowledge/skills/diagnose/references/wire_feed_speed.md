# Sensor Reference: wire_feed_speed

## Overview
The `wire_feed_speed` sensor measures the rate at which welding wire is fed through the contact tip to the weld pool. Consistent wire feed is critical for stable arc characteristics and weld quality. Anomalies in this sensor often indicate mechanical issues in the wire feed system.

## Known Failure Modes

| Failure Mode | Description | Signal Pattern |
|---|---|---|
| Intermittent Contact | Loose or corroded connector causes sporadic signal dropout | Short-duration dips to near-zero, brief recoveries |
| Cable Wear | Shield damage from repeated flexing; intermittent short or open circuit | Dropout dips lasting 50–200 ms, irregular intervals |
| Wire Spool Jam | Spool binding or tangling causes reduced or stopped feed | Gradual decline to zero or step reduction |
| Drive Roller Slip | Worn or improperly tensioned rollers fail to grip wire | Irregular step changes in speed, missing feed pulses |
| Motor Driver Fault | Servo driver overheating or power supply fluctuation | High-frequency oscillation around setpoint |
| Tip Wear / Blockage | Contact tip orifice worn or clogged, increasing friction | Slowly declining feed rate trend, periodic stuttering |

## Observable Signal Patterns

1. **Dropout dips** — Brief (50–200 ms) drops to < 30% of nominal feed rate; irregular intervals
2. **Irregular step changes** — Sudden jumps or drops > 15% of setpoint, no monotonic trend
3. **Periodically missing feed pulses** — In wire-pulse systems, one or more pulses skipped per cycle
4. **High-frequency oscillation** — Signal oscillates ±10% around setpoint at > 5 Hz
5. **Slow monotonic decline** — Feed rate drifts downward > 5% per minute without recovery

## Inspection Checklist

1. [ ] Visual inspection of cable from controller to welding torch — check for cuts, kinks, crushed sections
2. [ ] Connector pin inspection at both ends — verify continuity with multimeter
3. [ ] Wire spool — check for tangling, correct tension, proper threading
4. [ ] Drive rollers — inspect for wear, correct groove size, tension adjustment
5. [ ] Wire liner — remove and inspect for debris, kinks, or wear
6. [ ] Contact tip — inspect for oval/blocked orifice, replace if worn
7. [ ] Motor driver status LED / error code — consult manufacturer manual

## PM Card Keywords
- `wire feed speed`, `wire_feed`, `feed rate`
- `cable wear`, `intermittent feed`, `dropout`
- `drive roller`, `spool jam`, `tip blockage`
- `motor driver`, `servo fault`, `feed oscillation`

## Severity Guidelines

| Severity | Criteria |
|---|---|
| LOW | Single dropout < 100 ms, no reoccurrence in 1 hour |
| MEDIUM | Repeated dropouts > 3 per 10 min, or step change > 15% |
| HIGH | Feed rate zero for > 1 s, or oscillation > 20% of setpoint |
| CRITICAL | Complete feed loss > 5 s, or visible arc instability / burnback |

## Recommended Actions

### Short-Term
- Loose/tangential connector → reseat connector at both ends
- Minor cable fray → tape and schedule replacement
- Drive roller slip → clean rollers, adjust tension

### Long-Term
- Replace worn cable assembly
- Replace drive rollers at rated service interval
- Install ferrite choke on motor driver power line if oscillation persists
- Calibrate wire feed servo potentiometer / encoder
