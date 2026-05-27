# Sensor Reference: wire_temperature

## Overview
The `wire_temperature` sensor monitors the thermal state of the welding wire, contact tip, or near-arc region. Temperature is a critical parameter because excess heat accelerates contact tip wear, increases spatter, and degrades weld mechanical properties. Anomalies often reflect cooling system problems or electrical contact resistance changes.

## Known Failure Modes

| Failure Mode | Description | Signal Pattern |
|---|---|---|
| Cooling System Degradation | Coolant flow reduction or pump inefficiency | Monotonic drift upward over minutes/hours |
| Thermal Runaway | Positive feedback: heat → resistance → more heat | Exponential rise; accelerates over time |
| Contact Resistance Increase | Oxidized/dirty contact surfaces generate heat | Step offset upward of 10–30 °C, stays elevated |
| Coolant Loop Blockage | Debris or air lock in cooling circuit | Rapid ramp followed by plateau above normal range |
| Ambient Temperature Shift | Sudden environmental change (door open, HVAC failure) | Slow drift, correlated across multiple robots |
| Sensor Drift / Calibration Loss | Thermocouple degradation or A/D converter offset | Slow baseline shift, may be negative or positive |

## Observable Signal Patterns

1. **Monotonic drift upward** — Temperature increases steadily > 2 °C per minute without plateaus
2. **Sudden step offset** — Abrupt jump of > 10 °C, sustained at new level
3. **Slow ramp beyond normal range** — Gradual rise > 5 °C over 30 min, exceeds operating envelope
4. **Exponential acceleration** — Temperature derivative increases, characteristic of thermal runaway
5. **Oscillation around setpoint** — Temperature swings ±5 °C with period 1–5 min (thermostat hunting)

## Normal Operating Range
- Steady-state: 80–120 °C (typical for GMAW at 200–300 A)
- Peak (during weld cycle): up to 180 °C
- Ambient compensation: ±5 °C variation acceptable

## Inspection Checklist

1. [ ] Check coolant reservoir level and condition (discoloration, debris)
2. [ ] Inspect coolant hoses for kinks, leaks, or blockage
3. [ ] Verify coolant pump operation — listen for cavitation noise
4. [ ] Check radiator / heat exchanger fins for dust or obstruction
5. [ ] Inspect contact tip — replace if orifice is oval or eroded
6. [ ] Verify thermocouple / RTD is seated properly in measurement well
7. [ ] Check A/D module input for reference voltage stability
8. [ ] Measure resistance of coolant temperature sensor with multimeter
9. [ ] Compare reading with handheld pyrometer at same location

## PM Card Keywords
- `wire temperature`, `wire_temperature`, `tip temp`
- `cooling system`, `coolant flow`, `overheating`
- `thermal runaway`, `contact resistance`
- `thermocouple`, `RTD`, `temperature drift`
- `heat exchanger`, `radiator`, `chiller`

## Severity Guidelines

| Severity | Criteria |
|---|---|
| LOW | Transient spike < 5 °C above normal, self-recovering within 60 s |
| MEDIUM | Sustained offset > 10 °C above normal for > 5 min, or slow drift > 2 °C/min |
| HIGH | Temperature exceeds 150 °C, or > 20 °C above normal sustained |
| CRITICAL | Exponential rise > 5 °C/min approaching 180+ °C; risk of tip melting / fire |

## Recommended Actions

### Short-Term
- Verify coolant pump is running and primed
- Clean radiator / heat exchanger fins
- Purge air from coolant loop
- Reduce duty cycle temporarily (lower wire feed speed or current)

### Long-Term
- Schedule coolant replacement per manufacturer interval
- Replace degraded coolant hoses
- Calibrate or replace temperature sensor if drift exceeds ±3 °C at known reference
- Install coolant flow sensor with alarm
- Review thermal management system design for peak current conditions
