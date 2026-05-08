from __future__ import annotations

from sqlalchemy.orm import Session

from ..db.models import PMCard


SENSORS = [
    "paint_pressure",
    "air_motor_rotation_speed",
    "high_pressure_voltage",
    "paint_flow_rate",
    "high_pressure_current",
    "electro_pneumatic_output",
]

BASE_LIBRARY = {
    "paint_pressure": {
        "cause": "Filter clogged by paint residue",
        "symptom": "Pressure drop appears in the middle of spray cycle",
        "action": "Replace inline filter and verify pressure stability",
    },
    "air_motor_rotation_speed": {
        "cause": "Bearing wear causes cyclic RPM loss",
        "symptom": "RPM oscillation appears every third cycle",
        "action": "Schedule bearing replacement and rebalance motor",
    },
    "high_pressure_voltage": {
        "cause": "Grounding quality degraded in spray zone",
        "symptom": "Voltage sag and unstable atomization",
        "action": "Clean grounding points and recheck continuity",
    },
    "paint_flow_rate": {
        "cause": "Nozzle partially clogged",
        "symptom": "Flow rate drop with intermittent spray pattern",
        "action": "Run purge sequence and clean nozzle",
    },
    "high_pressure_current": {
        "cause": "Electrostatic path has coating buildup",
        "symptom": "Current spikes near end of cycle",
        "action": "Clean electrostatic path and inspect insulation",
    },
    "electro_pneumatic_output": {
        "cause": "Solenoid drift introduces hysteresis",
        "symptom": "Actuator output lags command signal",
        "action": "Calibrate solenoid and verify response curve",
    },
}


def seed_pm_cards(session: Session) -> int:
    cards: list[PMCard] = []

    for idx in range(45):
        sensor = SENSORS[idx % len(SENSORS)]
        base = BASE_LIBRARY[sensor]
        cards.append(
            PMCard(
                card_id=f"PMC-{idx + 1:03d}",
                sensor=sensor,
                robot_code=None,
                cause=f"{base['cause']} (pattern {idx % 9 + 1})",
                symptom=f"{base['symptom']} (signal window {idx % 5 + 1})",
                action=f"{base['action']} (SOP-{120 + idx})",
            )
        )

    r07_specific = [
        (
            "PMC-R07-01",
            "paint_pressure",
            "R-07",
            "Inline filter at R-07 line accumulates fine particles quickly",
            "Pressure decays after initial stable window",
            "Replace filter mesh grade and shorten cleaning interval",
        ),
        (
            "PMC-R07-02",
            "paint_pressure",
            "R-07",
            "Pressure regulator drift after long idle periods",
            "Pressure overshoot then drop within same cycle",
            "Recalibrate regulator and enable warm-up purge",
        ),
        (
            "PMC-R07-03",
            "paint_pressure",
            "R-07",
            "Supply hose micro-leak near quick connector",
            "Sawtooth pressure profile during atomization",
            "Replace connector and perform leak test",
        ),
        (
            "PMC-R07-04",
            "paint_pressure",
            "R-07",
            "Viscosity drift due to delayed thinner mixing",
            "Pressure rises while flow declines",
            "Adjust thinner dosing and revalidate viscosity control",
        ),
        (
            "PMC-R07-05",
            "paint_pressure",
            "R-07",
            "Pump check valve wear at high duty cycles",
            "Pressure instability in repeated high-load batches",
            "Replace check valve and inspect pump seat",
        ),
    ]

    for card_id, sensor, robot_code, cause, symptom, action in r07_specific:
        cards.append(
            PMCard(
                card_id=card_id,
                sensor=sensor,
                robot_code=robot_code,
                cause=cause,
                symptom=symptom,
                action=action,
            )
        )

    session.add_all(cards)
    return len(cards)
