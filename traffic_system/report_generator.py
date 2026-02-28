from typing import Iterable

from traffic_system.types import ViolationRecord


def render_violation_report(records: Iterable[ViolationRecord]) -> str:
    rows = []
    for r in records:
        rows.append(
            " | ".join(
                [
                    r.timestamp.isoformat(timespec="seconds"),
                    f"Plate={r.detected_vehicle_number or 'UNREADABLE'}",
                    f"Violation={r.violation_type}",
                    f"Act={r.mv_act_section}",
                    f"Penalty=₹{r.penalty_amount}",
                ]
            )
        )
    return "\n".join(rows) if rows else "No violations detected."
