from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class ElementState:
    element_id: str
    element_type: str
    role: str
    a: int = 1
    q: int = 1
    c: float = 1.0
    c_min: float = 1.0
    latency_s: int = 0
    isolated_until_s: int | None = None

    def u(self) -> int:
        return int(self.a == 1 and self.q == 1 and self.c >= self.c_min)


@dataclass(frozen=True)
class SafePath:
    path_id: str
    elements: tuple[str, ...]
    latency_s: int
    code_mapping: str


@dataclass
class SecurityTopology:
    elements: dict[str, ElementState] = field(default_factory=dict)
    paths: list[SafePath] = field(default_factory=list)
    active_path_id: str = "primary"
    l_k_max_s: int = 900
    chain_lost_since_s: int | None = None
    path_switch_history: list[dict] = field(default_factory=list)

    @classmethod
    def default(cls, l_k_max_s: int = 900) -> "SecurityTopology":
        elements = {
            "S_zone_primary": ElementState("S_zone_primary", "node", "S", c_min=1.0, latency_s=0),
            "S_zone_backup": ElementState("S_zone_backup", "node", "S", c_min=1.0, latency_s=30),
            "G_ingress": ElementState("G_ingress", "node", "G", c_min=1.0, latency_s=0),
            "G_validation": ElementState("G_validation", "node", "G", c_min=1.0, latency_s=60),
            "D_supervisory": ElementState("D_supervisory", "node", "D", c_min=1.0, latency_s=0),
            "D_fallback": ElementState("D_fallback", "node", "D", c_min=1.0, latency_s=60),
            "A_primary": ElementState("A_primary", "node", "A", c_min=1.0, latency_s=0),
            "A_guarded": ElementState("A_guarded", "node", "A", c_min=1.0, latency_s=60),
            "O_boptest": ElementState("O_boptest", "node", "O", c_min=1.0, latency_s=0),
            "e_SG_primary": ElementState("e_SG_primary", "edge", "S-G", c_min=1.0, latency_s=0),
            "e_SG_backup": ElementState("e_SG_backup", "edge", "S-G", c_min=1.0, latency_s=30),
            "e_GD_primary": ElementState("e_GD_primary", "edge", "G-D", c_min=1.0, latency_s=0),
            "e_GD_guarded": ElementState("e_GD_guarded", "edge", "G-D", c_min=1.0, latency_s=60),
            "e_DA_primary": ElementState("e_DA_primary", "edge", "D-A", c_min=1.0, latency_s=0),
            "e_DA_fallback": ElementState("e_DA_fallback", "edge", "D-A", c_min=1.0, latency_s=60),
            "e_AO_primary": ElementState("e_AO_primary", "edge", "A-O", c_min=1.0, latency_s=0),
            "e_AO_guarded": ElementState("e_AO_guarded", "edge", "A-O", c_min=1.0, latency_s=60),
        }
        paths = [
            SafePath(
                "primary",
                ("S_zone_primary", "e_SG_primary", "G_ingress", "e_GD_primary", "D_supervisory", "e_DA_primary", "A_primary", "e_AO_primary", "O_boptest"),
                0,
                "measurements -> ingress -> supervisory controller -> primary actuation -> /advance",
            ),
            SafePath(
                "redundant_measurement",
                ("S_zone_backup", "e_SG_backup", "G_ingress", "e_GD_primary", "D_supervisory", "e_DA_primary", "A_primary", "e_AO_primary", "O_boptest"),
                60,
                "backup estimator -> ingress -> supervisory controller -> primary actuation -> /advance",
            ),
            SafePath(
                "fallback_controller",
                ("S_zone_primary", "e_SG_primary", "G_validation", "e_GD_guarded", "D_fallback", "e_DA_fallback", "A_guarded", "e_AO_guarded", "O_boptest"),
                240,
                "validated measurements -> safe fallback controller -> guarded actuation -> /advance",
            ),
            SafePath(
                "reconfigured_guarded",
                ("S_zone_backup", "e_SG_backup", "G_validation", "e_GD_guarded", "D_fallback", "e_DA_fallback", "A_guarded", "e_AO_guarded", "O_boptest"),
                300,
                "backup estimator -> validation gateway -> safe fallback controller -> guarded actuation -> /advance",
            ),
        ]
        return cls(elements=elements, paths=paths, l_k_max_s=l_k_max_s)

    def path_by_id(self, path_id: str) -> SafePath:
        for path in self.paths:
            if path.path_id == path_id:
                return path
        raise KeyError(path_id)

    def phi_path(self, path: SafePath) -> int:
        if path.latency_s > self.l_k_max_s:
            return 0
        return int(all(self.elements[element_id].u() for element_id in path.elements))

    def valid_paths(self) -> list[SafePath]:
        return [path for path in self.paths if self.phi_path(path)]

    def phi_sys(self) -> int:
        return int(bool(self.valid_paths()))

    def number_of_safe_paths(self) -> int:
        return len(self.valid_paths())

    def active_path(self) -> SafePath | None:
        valid = {path.path_id: path for path in self.valid_paths()}
        if self.active_path_id in valid:
            return valid[self.active_path_id]
        return None

    def set_active_path(self, path_id: str, time_s: int, reason: str) -> None:
        if path_id != self.active_path_id:
            self.path_switch_history.append({"time_s": time_s, "from": self.active_path_id, "to": path_id, "reason": reason})
            self.active_path_id = path_id

    def degrade(self, element_ids: Iterable[str], time_s: int, reason: str) -> list[dict]:
        events = []
        for element_id in element_ids:
            if element_id in self.elements and self.elements[element_id].q != 0:
                self.elements[element_id].q = 0
                events.append({"time_s": time_s, "event_type": "element_untrusted", "component": element_id, "detail": reason})
        return events

    def isolate(self, element_id: str, time_s: int, reason: str) -> list[dict]:
        if element_id not in self.elements:
            return []
        element = self.elements[element_id]
        element.a = 0
        element.q = 0
        return [
            {"time_s": time_s, "event_type": "isolation_start", "component": element_id, "detail": reason},
            {"time_s": time_s, "event_type": "isolation_complete", "component": element_id, "detail": reason},
        ]

    def recover(self, element_id: str, time_s: int) -> list[dict]:
        if element_id not in self.elements:
            return []
        element = self.elements[element_id]
        element.a = 1
        element.q = 1
        element.c = max(element.c, element.c_min)
        return [
            {"time_s": time_s, "event_type": "recovery_verification", "component": element_id, "detail": "engineering_condition_passed"},
            {"time_s": time_s, "event_type": "recovery_complete", "component": element_id, "detail": "safe_rejoin_ready"},
            {"time_s": time_s, "event_type": "safe_rejoin", "component": element_id, "detail": "u_x=1"},
        ]



ATTACK_SURFACES = {
    "sensor_bias": {
        "surface": "S_zone_primary",
        "affected_component": "S_zone_primary",
        "security_property": "measurement_integrity",
        "description": "Bias one or more primary zone-temperature measurements before D_supervisory.",
    },
    "sensor_dropout": {
        "surface": "S_zone_primary",
        "affected_component": "S_zone_primary",
        "security_property": "measurement_availability",
        "description": "Remove primary zone-temperature measurements and break the 3-of-5 controller quorum.",
    },
    "command_injection": {
        "surface": "e_DA_primary",
        "affected_component": "e_DA_primary",
        "security_property": "command_integrity",
        "description": "Modify the D_supervisory -> A_primary command after control calculation and before execution.",
    },
    "combined": {
        "surface": "S_zone_primary+e_DA_primary",
        "affected_component": "S_zone_primary+e_DA_primary",
        "security_property": "cross_layer_integrity",
        "description": "Sensor bias plus command injection on the same time window; no dropout component.",
    },
}


def topology_rows() -> list[dict[str, object]]:
    topology = SecurityTopology.default()
    rows: list[dict[str, object]] = []
    for element in topology.elements.values():
        rows.append({
            "record_type": "element",
            "id": element.element_id,
            "role": element.role,
            "element_type": element.element_type,
            "latency_s": element.latency_s,
            "elements": "",
            "description": "",
        })
    for path in topology.paths:
        rows.append({
            "record_type": "path",
            "id": path.path_id,
            "role": "safe_path",
            "element_type": "path",
            "latency_s": path.latency_s,
            "elements": " -> ".join(path.elements),
            "description": path.code_mapping,
        })
    for attack, meta in ATTACK_SURFACES.items():
        rows.append({
            "record_type": "attack_surface",
            "id": attack,
            "role": str(meta["security_property"]),
            "element_type": "attack",
            "latency_s": 0,
            "elements": str(meta["surface"]),
            "description": str(meta["description"]),
        })
    mechanism_rows = [
        ("M_red", "redundancy", 300, "S_zone_backup -> primary actuator", "Switch to current independent backup measurements after primary sensing becomes untrusted."),
        ("M_cfg", "reconfiguration", 600, "G_validation -> D_fallback -> A_guarded", "Search a valid guarded path and use a dynamic fallback command derived from healthy primary or backup measurements."),
        ("M_iso", "isolation", 300, "S_zone_primary / e_DA_primary", "Quarantine an observably unhealthy sensing or command-transport element; an isolated command edge enters fail-static hold instead of relinquishing BOPTEST override control."),
        ("M_rec", "recovery", 4500, "isolated/untrusted component -> primary path", "Require 3600 s continuous health plus 900 s probation before safe rejoin and primary-path restoration."),
    ]
    for mid, role, latency, elements, description in mechanism_rows:
        rows.append({
            "record_type": "mechanism",
            "id": mid,
            "role": role,
            "element_type": "mechanism",
            "latency_s": latency,
            "elements": elements,
            "description": description,
        })
    return rows
