"""PM (Preventive Maintenance) Card definitions — prescriptive guidance lookup.

Each :class:`PMCard` links a cause-symptom-action triple to a sensor and robot
code, providing the diagnostic agent with expert-curated lookup knowledge for
paint-pressure anomalies on robot R-07 (and one shared-manifold card).

Functions:
    seed_pm_cards: Insert all cards into the database session.
"""

from sqlalchemy.orm import Session

from ..db.models import PMCard

CARDS = [
    PMCard(
        card_id="PMC-R07-001",
        sensor="paint_pressure",
        robot_code="R-07",
        cause="Paint residue buildup in inline filter restricts flow over time",
        symptom="Paint_pressure gradually drops 15-20% across consecutive cycles",
        action="Replace inline filter every 200 cycles or at first sign of pressure decay",
    ),
    PMCard(
        card_id="PMC-R07-002",
        sensor="paint_pressure",
        robot_code="R-07",
        cause="Pressure regulator diaphragm fatigue after extended operation",
        symptom="Pressure setpoint drifts upward by 5-8% then oscillates",
        action="Recalibrate regulator; replace diaphragm if drift exceeds 10%",
    ),
    PMCard(
        card_id="PMC-R07-003",
        sensor="paint_pressure",
        robot_code="R-07",
        cause="Supply hose micro-crack near quick-connect fitting",
        symptom="Sawtooth pressure profile during atomization phase",
        action="Inspect and replace quick-connect seal; pressure test the supply line",
    ),
    PMCard(
        card_id="PMC-R07-004",
        sensor="paint_pressure",
        robot_code="R-07",
        cause="Viscosity drift from delayed thinner mixing increases pump load",
        symptom="Paint_pressure rises while flow rate simultaneously declines",
        action="Adjust thinner dosing schedule; calibrate viscosity sensor",
    ),
    PMCard(
        card_id="PMC-R07-005",
        sensor="paint_pressure",
        robot_code="R-07",
        cause="Accumulating micro-particles in paint supply",
        symptom="Pressure instability worsens over multi-day production runs",
        action="Install secondary filtration and shorten pigment batch intervals",
    ),
    PMCard(
        card_id="PMC-R07-006",
        sensor="paint_pressure",
        robot_code="R-07",
        cause="Check valve wear at pump outlet allows backflow",
        symptom="Pressure drops sharply between pump strokes, increasing ripple",
        action="Replace check valve assembly; inspect pump seat and seals",
    ),
    PMCard(
        card_id="PMC-R07-007",
        sensor="paint_pressure",
        robot_code="R-07",
        cause="Temperature fluctuation in paint supply changes viscosity mid-shift",
        symptom="Pressure baseline shifts with ambient temperature changes",
        action="Stabilize paint temperature with circulation heater; verify thermostat",
    ),
    PMCard(
        card_id="PMC-R07-008",
        sensor="paint_pressure",
        robot_code=None,
        cause="Shared paint manifold has sedimentation buildup",
        symptom="Multiple robots show simultaneous pressure drop pattern",
        action="Flush main paint manifold; schedule quarterly manifold cleaning",
    ),
]


def seed_pm_cards(session: Session) -> int:
    session.add_all(CARDS)
    return len(CARDS)
