from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


Point2D = tuple[float, float]


class InspectionStatus(str, Enum):
    PENDING = "pending"
    DONE = "done"
    SKIPPED = "skipped"


@dataclass
class InspectionPoint:
    """
    A map-annotated task point for v3.0 inspection missions.

    priority is a dimensionless urgency/importance weight. risk_value is a
    task-local soft penalty, not a hard terrain or random-layer risk.
    """

    id: str
    xy: Point2D
    priority: float = 1.0
    service_time_s: float = 30.0
    risk_value: float = 0.0
    altitude_agl_m: float | None = None
    deadline_s: float | None = None
    status: InspectionStatus = InspectionStatus.PENDING

    @property
    def is_pending(self) -> bool:
        return self.status == InspectionStatus.PENDING


@dataclass
class ChargingStation:
    """A map-annotated charging/docking point."""

    id: str
    xy: Point2D
    charge_rate_j_per_s: float = 2500.0
    docking_time_s: float = 20.0
    target_soc: float = 0.95
    available: bool = True


@dataclass
class MissionMap:
    """
    V3.0 mission layer map.

    This is deliberately separate from terrain MapManager: it stores semantic
    task annotations over the physical map.
    """

    start_xy: Point2D
    inspection_points: list[InspectionPoint] = field(default_factory=list)
    charging_stations: list[ChargingStation] = field(default_factory=list)
    home_xy: Point2D | None = None
    name: str = "v30_mission"

    def pending_inspections(self) -> list[InspectionPoint]:
        return [point for point in self.inspection_points if point.is_pending]

    def available_chargers(self) -> list[ChargingStation]:
        return [station for station in self.charging_stations if station.available]

    def mark_done(self, point_id: str) -> None:
        point = self.get_inspection(point_id)
        point.status = InspectionStatus.DONE

    def mark_skipped(self, point_id: str) -> None:
        point = self.get_inspection(point_id)
        point.status = InspectionStatus.SKIPPED

    def get_inspection(self, point_id: str) -> InspectionPoint:
        for point in self.inspection_points:
            if point.id == point_id:
                return point
        raise KeyError(f"Unknown inspection point: {point_id}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["inspection_points"] = [
            {
                **asdict(point),
                "status": point.status.value,
            }
            for point in self.inspection_points
        ]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MissionMap":
        inspections = [
            InspectionPoint(
                id=str(item["id"]),
                xy=(float(item["xy"][0]), float(item["xy"][1])),
                priority=float(item.get("priority", 1.0)),
                service_time_s=float(item.get("service_time_s", 30.0)),
                risk_value=float(item.get("risk_value", 0.0)),
                altitude_agl_m=(
                    None if item.get("altitude_agl_m") is None else float(item.get("altitude_agl_m"))
                ),
                deadline_s=None if item.get("deadline_s") is None else float(item.get("deadline_s")),
                status=InspectionStatus(item.get("status", InspectionStatus.PENDING.value)),
            )
            for item in data.get("inspection_points", [])
        ]
        chargers = [
            ChargingStation(
                id=str(item["id"]),
                xy=(float(item["xy"][0]), float(item["xy"][1])),
                charge_rate_j_per_s=float(item.get("charge_rate_j_per_s", 2500.0)),
                docking_time_s=float(item.get("docking_time_s", 20.0)),
                target_soc=float(item.get("target_soc", 0.95)),
                available=bool(item.get("available", True)),
            )
            for item in data.get("charging_stations", [])
        ]
        home_xy = data.get("home_xy")
        return cls(
            name=str(data.get("name", "v30_mission")),
            start_xy=(float(data["start_xy"][0]), float(data["start_xy"][1])),
            home_xy=None if home_xy is None else (float(home_xy[0]), float(home_xy[1])),
            inspection_points=inspections,
            charging_stations=chargers,
        )

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "MissionMap":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

