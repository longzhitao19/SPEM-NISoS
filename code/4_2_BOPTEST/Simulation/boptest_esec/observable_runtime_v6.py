"""Observation-only online protection runtime.

The runtime consumes controller-facing primary measurements, an optional
independent redundant measurement channel, actuator readbacks, local timers and
mechanism state.  It never receives ChallengeSpec, attack labels, attack start
or end times, injected offsets, future samples, or offline scoring variables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from statistics import median
from typing import Any

from .client import ZONE_TEMPERATURES, SUPPLY_SETPOINT, SUPPLY_SETPOINT_ACTIVATE
from .protocol import MECHANISM_EXECUTION_PROBABILITIES, MechanismConfig


HEAT_SETPOINT_K = 308.15
HOLD_SETPOINT_K = 292.15
COOL_SETPOINT_K = 285.15
LOW_SWITCH_K = 292.95
HIGH_SWITCH_K = 297.35


@dataclass(frozen=True)
class ObservableParameters:
    sensor_span_k: float = 3.5
    sensor_step_k: float = 4.6
    sensor_signature_k: float = 2.0
    sensor_min_k: float = 273.15
    sensor_max_k: float = 323.15
    min_controller_sensor_quorum: int = 3
    command_residual_k: float = 0.75
    # Passive integrity-monitor tolerance between the controller-facing primary
    # channel and the independent reference channel.  The simulated reference
    # calibration uncertainty is bounded by +/-0.03 K, so 0.12 K is a 4x
    # uncertainty guard band fixed from the measurement model rather than from
    # any attack magnitude.  M_red controls active substitution only; the
    # integrity monitor is common observation infrastructure for all configs.
    sensor_reference_residual_k: float = 0.12
    isolation_delay_s: int = 300
    redundancy_delay_s: int = 300
    reconfiguration_delay_s: int = 600
    recovery_min_delay_s: int = 600
    recovery_healthy_window_s: int = 3600
    recovery_probation_s: int = 900
    backup_max_age_s: int = 900
    containment_trusted_hold_max_s: int = 900
    sample_s: int = 300
    # Protection actions are not assumed to be cost-free.  Reconfiguration and
    # trusted primary rejoin use one fixed-step break-before-make/resync window.
    # During this interval the security boundary remains intact, but fresh
    # supervisory execution availability is temporarily interrupted.  This is
    # an engineering action state, not an arbitrary score penalty.
    reconfiguration_handover_s: int = 300
    recovery_resynchronization_s: int = 300
    handover_boundary: float = 1.00
    handover_availability: float = 0.00
    handover_service: float = 0.70
    # Per-trigger execution reliability. These are scenario parameters rather
    # than universal empirical constants. They preserve the reliability order
    # used by the theoretical case while allowing engineering feasibility to
    # veto an otherwise successful random execution.
    redundancy_success_probability: float = MECHANISM_EXECUTION_PROBABILITIES["M_red"]
    reconfiguration_success_probability: float = MECHANISM_EXECUTION_PROBABILITIES["M_cfg"]
    isolation_success_probability: float = MECHANISM_EXECUTION_PROBABILITIES["M_iso"]
    recovery_success_probability: float = MECHANISM_EXECUTION_PROBABILITIES["M_rec"]


@dataclass
class ObservableRuntime:
    mechanism: MechanismConfig
    parameters: ObservableParameters = field(default_factory=ObservableParameters)
    isolated: set[str] = field(default_factory=set)
    untrusted: set[str] = field(default_factory=set)
    isolation_due: dict[str, int] = field(default_factory=dict)
    isolated_at: dict[str, int] = field(default_factory=dict)
    untrusted_since: dict[str, int] = field(default_factory=dict)
    healthy_since: dict[str, int] = field(default_factory=dict)
    previous_observed: dict[str, float] = field(default_factory=dict)
    sensor_signature_reference: dict[str, float] = field(default_factory=dict)
    recovery_probation_until: dict[str, int] = field(default_factory=dict)
    degradation_mode: dict[str, str] = field(default_factory=dict)
    backup_memory: dict[str, float] = field(default_factory=dict)
    backup_updated_s: int | None = None
    backup_ok: bool = False
    backup_reference_ok: bool = False
    primary_reference_residual_k: float = 0.0
    primary_reference_disagreement: bool = False
    last_trusted_observation: dict[str, float] = field(default_factory=dict)
    last_trusted_command: float | None = None
    active_path: str = "primary"
    prepared_path: str = "primary"
    red_due: int | None = None
    cfg_due: int | None = None
    prepared_time: int | None = None
    protection_transient_until_s: int | None = None
    protection_transient_kind: str | None = None
    stochastic_seed: int = 0
    mechanism_attempt_count: dict[str, int] = field(default_factory=dict)
    isolation_attempted: set[str] = field(default_factory=set)
    recovery_attempted: set[str] = field(default_factory=set)
    red_attempted_episode: bool = False
    cfg_attempted_episode: bool = False
    last_mechanism_attempt: dict[str, dict[str, Any]] = field(default_factory=dict)

    def allowed_paths(self) -> set[str]:
        paths = {"primary"}
        if self.mechanism.m_red:
            paths.add("redundant_measurement")
        if self.mechanism.m_cfg:
            paths.add("fallback_controller")
        if self.mechanism.m_red and self.mechanism.m_cfg:
            paths.add("reconfigured_guarded")
        return paths

    @staticmethod
    def control_state(command: float | None) -> str:
        if command is None or not isfinite(float(command)):
            return "OFF"
        value = float(command)
        if abs(value - HEAT_SETPOINT_K) < 1e-6:
            return "HEAT"
        if abs(value - COOL_SETPOINT_K) < 1e-6:
            return "COOL"
        if abs(value - HOLD_SETPOINT_K) < 1e-6:
            return "HOLD"
        return "MODULATED"

    @staticmethod
    def controller(values: dict[str, float | None], min_valid: int = 3) -> float | None:
        valid = [float(x) for x in values.values() if x is not None and isfinite(float(x))]
        if len(valid) < int(min_valid):
            return None
        center = median(valid)
        if center > HIGH_SWITCH_K:
            return COOL_SETPOINT_K
        if center < LOW_SWITCH_K:
            return HEAT_SETPOINT_K
        return HOLD_SETPOINT_K

    def sensor_health(self, observed: dict[str, float | None]) -> bool:
        p = self.parameters
        if set(observed) != set(ZONE_TEMPERATURES):
            return False
        if any(v is None or not isfinite(float(v)) or not p.sensor_min_k <= float(v) <= p.sensor_max_k for v in observed.values()):
            return False
        values = [float(v) for v in observed.values()]
        if max(values) - min(values) > p.sensor_span_k:
            return False
        if self.previous_observed:
            delta = [abs(float(observed[k]) - self.previous_observed[k]) for k in observed]
            if max(delta) > p.sensor_step_k:
                return False
        if self.sensor_signature_reference:
            center = median(values)
            signature = {k: float(observed[k]) - center for k in observed}
            signature_error = max(abs(signature[k] - self.sensor_signature_reference[k]) for k in observed)
            if signature_error > p.sensor_signature_k:
                return False
        return True

    def backup_sensor_health(self, observed: dict[str, float | None] | None) -> bool:
        if observed is None or set(observed) != set(ZONE_TEMPERATURES):
            return False
        p = self.parameters
        valid = []
        for name in ZONE_TEMPERATURES:
            value = observed.get(name)
            if value is None or not isfinite(float(value)) or not p.sensor_min_k <= float(value) <= p.sensor_max_k:
                return False
            valid.append(float(value))
        # Independent redundant sensors may have a small calibration offset but
        # should still preserve a physically plausible inter-zone spread.
        return (max(valid) - min(valid)) <= (p.sensor_span_k + 0.75)

    def _update_primary_reference(self, observed: dict[str, float | None]) -> None:
        trusted = {k: float(v) for k, v in observed.items() if v is not None}
        self.previous_observed = trusted
        self.last_trusted_observation = dict(trusted)
        center = median(trusted.values())
        current_signature = {k: trusted[k] - center for k in trusted}
        if not self.sensor_signature_reference:
            self.sensor_signature_reference = current_signature
        else:
            alpha = 0.15
            self.sensor_signature_reference = {
                k: (1.0 - alpha) * self.sensor_signature_reference[k] + alpha * current_signature[k]
                for k in trusted
            }

    def _fallback_command(self, primary_observed: dict[str, float | None]) -> tuple[float, str]:
        """Return a safe dynamic fallback command and its observable source.

        Priority is current healthy primary measurements, then current healthy
        redundant measurements, then the last trusted command.  A fixed neutral
        setpoint is used only as the last resort.  This avoids the old behaviour
        where every fallback action forced 292.15 K even when benign control
        should have remained in HEAT or COOL mode.
        """
        if self.sensor_ok:
            cmd = self.controller(primary_observed, self.parameters.min_controller_sensor_quorum)
            if cmd is not None:
                return float(cmd), "healthy_primary_measurements"
        if self.backup_available:
            cmd = self.controller(self.backup_memory, self.parameters.min_controller_sensor_quorum)
            if cmd is not None:
                return float(cmd), "independent_backup_measurements"
        if self.last_trusted_command is not None:
            return float(self.last_trusted_command), "last_trusted_command"
        return HOLD_SETPOINT_K, "neutral_safe_default"

    def prepare(
        self,
        observed: dict[str, float | None],
        time_s: int,
        backup_observed: dict[str, float | None] | None = None,
    ) -> dict[str, float | None]:
        """Prepare endpoint commands before the untrusted command boundary."""
        if self.prepared_time is not None and time_s <= self.prepared_time:
            raise ValueError("observations must have strictly increasing timestamps")
        self.prepared_time = int(time_s)
        # Freeze the path that actually participates in this control sample.
        # ``finish`` may select another path for the *next* sample, but that new
        # path must not receive execution credit before it has produced a command.
        self.prepared_path = str(self.active_path)
        self.observed = dict(observed)
        intrinsic_sensor_ok = self.sensor_health(observed)

        # The independent channel has two deliberately separated roles.  Its
        # read-only comparison is part of the common state-awareness layer and
        # is therefore available in every mechanism configuration.  Active
        # substitution into the control path remains exclusive to M_red.
        backup_candidate = backup_observed
        if backup_candidate is None and intrinsic_sensor_ok:
            # Legacy/unit-test compatibility when no independent channel is
            # supplied.  This does not create extra information.
            backup_candidate = observed
        self.backup_reference_ok = bool(self.backup_sensor_health(backup_candidate))
        self.primary_reference_residual_k = 0.0
        self.primary_reference_disagreement = False
        if intrinsic_sensor_ok and self.backup_reference_ok and backup_candidate is not None:
            residuals = [
                abs(float(observed[name]) - float(backup_candidate[name]))
                for name in ZONE_TEMPERATURES
                if observed.get(name) is not None and backup_candidate.get(name) is not None
            ]
            if residuals:
                # Median residual makes the monitor robust to a single local
                # calibration outlier while remaining sensitive to the
                # majority-sensor bias topology used in this experiment.
                self.primary_reference_residual_k = float(median(residuals))
                self.primary_reference_disagreement = bool(
                    self.primary_reference_residual_k > self.parameters.sensor_reference_residual_k
                )

        self.sensor_ok = bool(intrinsic_sensor_ok and not self.primary_reference_disagreement)
        if self.sensor_ok:
            self._update_primary_reference(observed)

        self.backup_ok = bool(self.mechanism.m_red and self.backup_reference_ok)
        if self.backup_ok and backup_candidate is not None:
            self.backup_memory = {name: float(backup_candidate[name]) for name in ZONE_TEMPERATURES}
            self.backup_updated_s = int(time_s)
        self.backup_available = bool(
            self.mechanism.m_red
            and self.backup_memory
            and self.backup_updated_s is not None
            and int(time_s) - self.backup_updated_s <= self.parameters.backup_max_age_s
        )
        self.backup_used = bool(
            self.prepared_path in {"redundant_measurement", "reconfigured_guarded"}
            and self.backup_available
        )

        primary_values = self.backup_memory if self.backup_used else observed
        primary_cmd = self.controller(primary_values, self.parameters.min_controller_sensor_quorum)
        guarded_cmd, guarded_source = self._fallback_command(observed)
        self.fallback_source = guarded_source
        self.intended = {
            "primary": primary_cmd,
            "guarded": guarded_cmd if self.mechanism.m_cfg else None,
        }
        self.primary_control_state = self.control_state(primary_cmd)
        self.guarded_control_state = self.control_state(self.intended["guarded"])
        return dict(self.intended)

    def _event(
        self,
        events: list[dict[str, Any]],
        time_s: int,
        kind: str,
        component: str,
        detail: str = "",
        **extra: Any,
    ) -> None:
        row = {"time_s": int(time_s), "event_type": kind, "component": component, "detail": detail}
        row.update(extra)
        events.append(row)

    def _start_protection_transient(
        self,
        time_s: int,
        duration_s: int,
        kind: str,
        events: list[dict[str, Any]],
    ) -> None:
        """Start a bounded protection-action transient.

        The transient represents real handover/resynchronization work.  It is
        intentionally attached only to an action that is actually attempted or
        completed; merely enabling a mechanism does not lower Phi_exec.
        """
        duration = max(0, int(duration_s))
        if duration <= 0:
            return
        until = int(time_s) + duration
        self.protection_transient_until_s = max(
            int(self.protection_transient_until_s or time_s), until
        )
        self.protection_transient_kind = str(kind)
        self._event(
            events,
            time_s,
            "protection_transient_start",
            str(kind),
            f"until_s={self.protection_transient_until_s}",
            protection_transient_until_s=int(self.protection_transient_until_s),
        )

    def _protection_transient_active(self, time_s: int) -> bool:
        return bool(
            self.protection_transient_until_s is not None
            and int(time_s) < int(self.protection_transient_until_s)
        )

    def _mechanism_probability(self, mechanism: str) -> float:
        p = self.parameters
        return {
            "M_red": p.redundancy_success_probability,
            "M_cfg": p.reconfiguration_success_probability,
            "M_iso": p.isolation_success_probability,
            "M_rec": p.recovery_success_probability,
        }[mechanism]

    def _mechanism_attempt(
        self, mechanism: str, key: str, time_s: int, events: list[dict[str, Any]]
    ) -> bool:
        """Draw one reproducible Bernoulli execution outcome.

        The draw is keyed by challenge-level ``stochastic_seed``, mechanism,
        component key and attempt index.  Therefore the same mechanism receives
        the same random draw in 1000 and 1111 for a matched challenge (common
        random numbers), while different mechanisms retain independent streams.
        """
        counter_key = f"{mechanism}|{key}"
        attempt = self.mechanism_attempt_count.get(counter_key, 0) + 1
        self.mechanism_attempt_count[counter_key] = attempt
        payload = f"{int(self.stochastic_seed)}|{mechanism}|{key}|{attempt}".encode("utf-8")
        raw = int.from_bytes(sha256(payload).digest()[:8], "big")
        draw = raw / float(2**64)
        probability = float(self._mechanism_probability(mechanism))
        success = bool(draw < probability)
        record = {
            "mechanism": mechanism,
            "key": key,
            "attempt": int(attempt),
            "probability": probability,
            "draw": float(draw),
            "success": int(success),
        }
        self.last_mechanism_attempt[mechanism] = record
        self._event(
            events, time_s, "mechanism_execution_attempt", mechanism, key,
            mechanism_attempt=int(attempt),
            mechanism_success_probability=probability,
            mechanism_random_draw=float(draw),
            mechanism_execution_success=int(success),
        )
        return success

    def _mark_health(self, node: str, ok: bool, time_s: int, events: list[dict[str, Any]], mode: str | None = None) -> None:
        if ok:
            self.healthy_since.setdefault(node, int(time_s))
            return
        self.healthy_since.pop(node, None)
        if mode in {"availability", "integrity"}:
            self.degradation_mode.setdefault(node, mode)
        if node in self.recovery_probation_until:
            self.recovery_probation_until.pop(node, None)
            self._event(events, time_s, "recovery_probation_failed", node, "observable_anomaly_during_probation")
        self._event(events, time_s, "anomaly_detected", node, "observable_measurement_or_readback_check")
        if node not in self.untrusted:
            self.untrusted.add(node)
            self.untrusted_since[node] = int(time_s)
            self._event(events, time_s, "element_untrusted", node, "observable_health_check_failed")
        if self.mechanism.m_iso and node not in self.isolated and node not in self.isolation_due and node not in self.isolation_attempted:
            self.isolation_due[node] = int(time_s) + self.parameters.isolation_delay_s
            self._event(events, time_s, "mechanism_trigger", "M_iso", node)
            self._event(events, time_s, "isolation_start", node, f"complete_due_s={self.isolation_due[node]}")

    def _complete_isolation(
        self, health: dict[str, bool], time_s: int, events: list[dict[str, Any]]
    ) -> None:
        for node, due in list(self.isolation_due.items()):
            if time_s < due:
                continue
            # Isolation acts on a currently observable fault.  If the element has
            # already returned to an observably healthy state before the actuation
            # delay expires, cancel this pending quarantine instead of isolating a
            # healthy component.  Trust is still not restored here; M_rec owns
            # trusted rejoin semantics.
            if bool(health.get(node, True)):
                del self.isolation_due[node]
                self._event(
                    events, time_s, "isolation_cancelled", node,
                    "observable_health_restored_before_completion",
                )
                continue
            self.isolation_attempted.add(node)
            del self.isolation_due[node]
            if self._mechanism_attempt("M_iso", node, time_s, events):
                self.isolated.add(node)
                self.isolated_at[node] = int(time_s)
                self._event(events, time_s, "isolation_complete", node, "quarantine_active")
            else:
                self._event(events, time_s, "isolation_failed", node, "probabilistic_execution_failure")

    def _complete_recovery(self, health: dict[str, bool], time_s: int, events: list[dict[str, Any]]) -> list[str]:
        if not self.mechanism.m_rec:
            return []
        recovered: list[str] = []
        p = self.parameters
        for node in sorted(self.untrusted):
            if node in self.recovery_attempted:
                continue
            fault_age = int(time_s) - self.untrusted_since.get(node, int(time_s))
            healthy_start = self.healthy_since.get(node)
            healthy_window = 0 if healthy_start is None else int(time_s) - healthy_start
            if fault_age < p.recovery_min_delay_s or not health.get(node, False):
                continue
            if healthy_window < p.recovery_healthy_window_s:
                self._event(events, time_s, "recovery_verification_pending", node, f"observed_healthy_window_s={healthy_window}")
                continue
            probation_until = self.recovery_probation_until.get(node)
            if probation_until is None:
                probation_until = int(time_s) + p.recovery_probation_s
                self.recovery_probation_until[node] = probation_until
                self._event(events, time_s, "recovery_verification", node, f"observed_healthy_window_s={healthy_window}")
                self._event(events, time_s, "recovery_probation_start", node, f"probation_until_s={probation_until}")
                continue
            if time_s < probation_until:
                self._event(events, time_s, "recovery_probation_pending", node, f"probation_until_s={probation_until}")
                continue
            self._event(events, time_s, "mechanism_trigger", "M_rec", node)
            self.recovery_attempted.add(node)
            if not self._mechanism_attempt("M_rec", node, time_s, events):
                self._event(events, time_s, "recovery_failed", node, "probabilistic_execution_failure")
                continue
            self.untrusted.discard(node)
            self.isolated.discard(node)
            self.isolated_at.pop(node, None)
            self.untrusted_since.pop(node, None)
            self.isolation_due.pop(node, None)
            self.recovery_probation_until.pop(node, None)
            self.degradation_mode.pop(node, None)
            self.isolation_attempted.discard(node)
            self.recovery_attempted.discard(node)
            self._event(events, time_s, "recovery_complete", node, "healthy_window_and_probation_passed")
            self._event(events, time_s, "safe_rejoin", node, "trust_restored")
            recovered.append(node)
        return recovered

    def _good(self, node: str, health: dict[str, bool]) -> bool:
        return bool(health.get(node, True) and node not in self.untrusted and node not in self.isolated)

    def _valid_paths(self, health: dict[str, bool]) -> dict[str, bool]:
        valid = {
            "primary": self._good("S_zone_primary", health) and self._good("e_DA_primary", health),
            "redundant_measurement": self.backup_available and self._good("e_DA_primary", health),
            "fallback_controller": self._good("S_zone_primary", health) and self._good("e_DA_guarded", health),
            "reconfigured_guarded": self.backup_available and self._good("e_DA_guarded", health),
        }
        allowed = self.allowed_paths()
        return {name: state for name, state in valid.items() if name in allowed}

    def _schedule_path_mechanisms(self, valid: dict[str, bool], time_s: int, events: list[dict[str, Any]]) -> None:
        if valid.get(self.active_path, False):
            return
        if self.mechanism.m_red and self.red_due is None and not self.red_attempted_episode:
            self.red_due = int(time_s) + self.parameters.redundancy_delay_s
            self._event(events, time_s, "mechanism_trigger", "M_red", f"complete_due_s={self.red_due}")
            self._event(events, time_s, "redundancy_start", "M_red", "independent_backup_measurement_path_selection")
        if self.mechanism.m_cfg and self.cfg_due is None and not self.cfg_attempted_episode:
            self.cfg_due = int(time_s) + self.parameters.reconfiguration_delay_s
            self._event(events, time_s, "mechanism_trigger", "M_cfg", f"complete_due_s={self.cfg_due}")
            self._event(events, time_s, "reconfiguration_start", "M_cfg", "safe_subgraph_search")

    def _complete_path_mechanisms(self, valid: dict[str, bool], time_s: int, events: list[dict[str, Any]]) -> None:
        # A path mechanism is scheduled only while the active path is invalid.
        # If another mechanism restores an executable path before a pending action
        # reaches its completion delay, cancel the stale action instead of forcing
        # a second, unnecessary path switch.  This prevents artificial synergy in
        # multi-mechanism coalitions caused purely by queued scheduling.
        if valid.get(self.active_path, False):
            if self.red_due is not None:
                self._event(events, time_s, "redundancy_cancelled", "M_red", "active_path_already_observably_valid")
                self.red_due = None
            if self.cfg_due is not None:
                self._event(events, time_s, "reconfiguration_cancelled", "M_cfg", "active_path_already_observably_valid")
                self.cfg_due = None
            return

        if self.red_due is not None and time_s >= self.red_due:
            self.red_attempted_episode = True
            feasible = bool(valid.get("redundant_measurement", False))
            if feasible and self._mechanism_attempt("M_red", "redundant_measurement", time_s, events):
                self.active_path = "redundant_measurement"
                self._event(events, time_s, "redundancy_complete", "M_red", self.active_path)
                self._event(events, time_s, "backup_resource_in_use", "S_zone_backup", self.active_path)
            elif feasible:
                self._event(events, time_s, "redundancy_failed", "M_red", "probabilistic_execution_failure")
            else:
                self._event(events, time_s, "redundancy_failed", "M_red", "backup_path_not_observably_valid")
            self.red_due = None
        if self.cfg_due is not None and time_s >= self.cfg_due:
            self.cfg_attempted_episode = True
            preference = ("fallback_controller", "reconfigured_guarded", "primary", "redundant_measurement")
            candidates = [path for path in preference if valid.get(path, False)]
            # Use a mechanism-level common random number rather than the
            # selected path name as the stochastic key.  The feasible path can
            # differ across coalitions, but the same M_cfg reliability draw must
            # remain paired for a matched challenge so Shapley/factorial effects
            # are not contaminated by configuration-dependent random streams.
            if candidates:
                # A real path-switch attempt has a short handover cost whether
                # the final Bernoulli execution succeeds or fails.  This allows
                # protection-induced harm to be observable without inventing an
                # arbitrary fixed score penalty.
                self._start_protection_transient(
                    time_s,
                    self.parameters.reconfiguration_handover_s,
                    "reconfiguration_handover",
                    events,
                )
                if self._mechanism_attempt("M_cfg", "path_switch", time_s, events):
                    self.active_path = candidates[0]
                    self._event(events, time_s, "reconfiguration_complete", "M_cfg", self.active_path)
                    if self.active_path in {"fallback_controller", "reconfigured_guarded"}:
                        self._event(events, time_s, "fallback_activation", "D_fallback", f"{self.active_path};source={self.fallback_source}")
                    if self.active_path == "reconfigured_guarded":
                        self._event(events, time_s, "backup_resource_in_use", "S_zone_backup", self.active_path)
                else:
                    self._event(events, time_s, "reconfiguration_failed", "M_cfg", "probabilistic_execution_failure")
            else:
                self._event(events, time_s, "reconfiguration_failed", "M_cfg", "no_observably_valid_path")
            self.cfg_due = None


    @staticmethod
    def _path_latency_s(path_id: str) -> int:
        return {
            "primary": 0,
            "redundant_measurement": 60,
            "fallback_controller": 240,
            "reconfigured_guarded": 300,
        }.get(path_id, 10**9)

    def _execution_state(
        self,
        *,
        sensor: str,
        actuator: str,
        health: dict[str, bool],
        command_value: float | None,
        execution_path: str,
        containment_active: bool = False,
        protection_transient_active: bool = False,
    ) -> dict[str, Any]:
        """Map observable engineering state to one unified theoretical state.

        The BOPTEST instance does not split disturbances into F/C classes.  Any
        selected element that is currently unhealthy is represented by one
        active degraded state; an element that has become healthy but has not
        completed trusted recovery is represented by one unrecovered state.
        Once the selected protection path contains no unresolved element, the
        execution state returns to the normal/restored state.

        Frozen state values:
          normal/restored:       (b,a,s)=(1.00,1.00,1.00)
          disturbed active:      (b,a,s)=(0.35,0.20,0.70)
          disturbed unrecovered: (b,a,s)=(0.50,0.40,0.70)
          isolated containment:   (b,a,s)=(1.00,1.00,0.70)

        Successful isolation with fail-static command hold is a safe contained
        state: the compromised chain is cut and actuator authority is preserved,
        but fresh closed-loop service has not been restored.  It therefore earns
        partial execution quality rather than remaining indistinguishable from an
        actively compromised chain.

        Phi_exec=min(b,a,s) when the selected execution mode latency is admissible.
        """
        selected: list[str] = []
        if sensor == "S_zone_primary":
            selected.append(sensor)
        if actuator in {"e_DA_primary", "e_DA_guarded", "A_primary", "A_guarded"}:
            selected.append(actuator)

        active_degradation = False
        unrecovered_degradation = False
        for node in selected:
            if node not in self.untrusted and node not in self.isolated:
                continue
            currently_healthy = bool(health.get(node, True))
            if currently_healthy:
                unrecovered_degradation = True
            else:
                active_degradation = True

        # No executable command is treated as an active degraded state.
        if command_value is None or not isfinite(float(command_value)):
            active_degradation = True

        if protection_transient_active:
            # The path-switch/rejoin action preserves the security boundary but
            # temporarily interrupts fresh supervisory availability.  Keep the
            # same theoretical min(b,a,s) construction rather than subtracting
            # an ad-hoc penalty from Phi_exec.
            b = float(self.parameters.handover_boundary)
            a = float(self.parameters.handover_availability)
            s = float(self.parameters.handover_service)
            state = f"protection_transient:{self.protection_transient_kind or 'handover'}"
            latency_s = int(self.parameters.sample_s)
        elif containment_active and command_value is not None and isfinite(float(command_value)):
            # M_iso has cut the compromised primary chain and the actuator is
            # explicitly kept under the experimental controller by a trusted
            # fail-static command.  Security boundary and command availability are
            # preserved, while service freshness remains degraded because the
            # controller is holding rather than closing the loop on that path.
            b, a, s = 1.00, 1.00, 0.70
            state = "isolated_containment"
            latency_s = int(self.parameters.isolation_delay_s)
        elif active_degradation:
            b, a, s = 0.35, 0.20, 0.70
            state = "disturbed_active"
            latency_s = self._path_latency_s(execution_path)
        elif unrecovered_degradation:
            b, a, s = 0.50, 0.40, 0.70
            state = "disturbed_unrecovered"
            latency_s = self._path_latency_s(execution_path)
        else:
            b, a, s = 1.00, 1.00, 1.00
            state = "normal_or_restored"
            latency_s = self._path_latency_s(execution_path)

        latency_limit_s = 900
        latency_ok = latency_s <= latency_limit_s
        phi_exec = min(b, a, s) if latency_ok else 0.0
        if not latency_ok:
            state = "disturbed_active"

        return {
            "b_exec": float(b),
            "a_exec": float(a),
            "s_exec": float(s),
            "L_exec_s": int(latency_s),
            "L_exec_norm": float(latency_s / latency_limit_s) if latency_s < 10**9 else float("inf"),
            "L_exec_limit_s": int(latency_limit_s),
            "phi_exec": float(phi_exec),
            "execution_state": state,
            "protection_transient_active": bool(protection_transient_active),
            "protection_transient_kind": self.protection_transient_kind if protection_transient_active else None,
            "protection_transient_until_s": self.protection_transient_until_s,
        }

    def finish(self, received: dict[str, float | None], time_s: int) -> dict[str, Any]:
        """Consume endpoint readbacks and return the selected plant command."""
        if time_s != self.prepared_time:
            raise ValueError("prepare must be called before finish for the same timestamp")
        p = self.parameters
        events: list[dict[str, Any]] = []
        health: dict[str, bool] = {"S_zone_primary": bool(self.sensor_ok)}
        # Readback mismatch identifies the D->A transport channel as untrusted.
        # The attack is injected on e_DA_primary; conflating that edge with the
        # physical actuator caused isolation to disable the whole plant input.
        for key, edge in (("primary", "e_DA_primary"), ("guarded", "e_DA_guarded")):
            want = self.intended[key]
            got = received.get(key)
            if want is None:
                health[edge] = edge not in self.untrusted and edge not in self.isolated
            else:
                health[edge] = bool(
                    got is not None
                    and isfinite(float(got))
                    and abs(float(got) - float(want)) < p.command_residual_k
                    and COOL_SETPOINT_K <= float(got) <= HEAT_SETPOINT_K
                )
        if not self.mechanism.m_cfg:
            health.pop("e_DA_guarded")

        sensor_mode = "availability" if self.controller(self.observed, p.min_controller_sensor_quorum) is None else "integrity"
        if self.primary_reference_disagreement:
            self._event(
                events,
                time_s,
                "sensor_integrity_crosscheck_failed",
                "S_zone_primary",
                f"median_primary_reference_residual_k={self.primary_reference_residual_k:.4f}",
            )
        self._mark_health("S_zone_primary", health.get("S_zone_primary", True), time_s, events, mode=sensor_mode)
        for edge in ("e_DA_primary", "e_DA_guarded"):
            if edge not in health:
                continue
            key = "primary" if edge == "e_DA_primary" else "guarded"
            got = received.get(key)
            edge_mode = "availability" if got is None else "integrity"
            self._mark_health(edge, health[edge], time_s, events, mode=edge_mode)
        self._complete_isolation(health, time_s, events)
        recovered = self._complete_recovery(health, time_s, events)

        valid = self._valid_paths(health)
        self._schedule_path_mechanisms(valid, time_s, events)
        self._complete_path_mechanisms(valid, time_s, events)

        if recovered:
            valid = self._valid_paths(health)
            if valid.get("primary", False) and self.active_path != "primary":
                self._start_protection_transient(
                    time_s,
                    self.parameters.recovery_resynchronization_s,
                    "recovery_resynchronization",
                    events,
                )
                self.active_path = "primary"
                self._event(events, time_s, "primary_path_rejoin", "M_rec", "primary_observably_valid")
        if valid.get("primary", False) and not self.untrusted:
            self.red_attempted_episode = False
            self.cfg_attempted_episode = False

        # Update the last trusted normal command only after both the primary
        # measurement and command channel pass their observable checks.
        if (
            self._good("S_zone_primary", health)
            and self._good("e_DA_primary", health)
            and self.intended.get("primary") is not None
        ):
            self.last_trusted_command = float(self.intended["primary"])

        # Use the path that actually generated commands in ``prepare``.  Any
        # path selected above is a next-sample decision and is exposed separately
        # as ``next_path``.
        execution_path = self.prepared_path
        guarded = execution_path in {"fallback_controller", "reconfigured_guarded"}
        endpoint = "guarded" if guarded else "primary"
        transport = "e_DA_guarded" if guarded else "e_DA_primary"
        sensor = "S_zone_backup" if execution_path in {"redundant_measurement", "reconfigured_guarded"} else "S_zone_primary"

        if sensor == "S_zone_backup" and not self.backup_used:
            command_value = None
        else:
            command_value = received.get(endpoint)

        blocked = transport in self.isolated or (sensor == "S_zone_primary" and sensor in self.isolated)
        containment_active = False
        containment_source = "none"
        baseline_fail_static_active = False
        baseline_fail_static_source = "none"
        if blocked:
            # Isolation is implemented at the compromised channel boundary.  A
            # recently trusted command may be held only for a bounded stale-control
            # interval.  Holding it indefinitely can itself become unsafe when the
            # plant operating point changes after quarantine.  After the fixed
            # 900 s stale-command budget expires, containment transitions to the
            # neutral degraded setpoint while retaining BOPTEST external authority.
            blocked_components: list[str] = []
            if transport in self.isolated:
                blocked_components.append(transport)
            if sensor == "S_zone_primary" and sensor in self.isolated:
                blocked_components.append(sensor)
            isolation_times = [self.isolated_at[x] for x in blocked_components if x in self.isolated_at]
            containment_age_s = 0 if not isolation_times else int(time_s) - min(isolation_times)
            trusted_hold_admissible = bool(
                self.last_trusted_command is not None
                and isfinite(float(self.last_trusted_command))
                and containment_age_s <= p.containment_trusted_hold_max_s
            )
            if trusted_hold_admissible:
                command_value = float(self.last_trusted_command)
                containment_source = "bounded_last_trusted_command_hold"
            else:
                command_value = HOLD_SETPOINT_K
                containment_source = "neutral_safe_hold_after_stale_budget"
            containment_active = True
        elif command_value is None or not isfinite(float(command_value)):
            # Unprotected degraded-controller contract.  If the selected path
            # cannot form a fresh command, keep external BOPTEST authority but
            # fall back to one fixed neutral command.  Reusing the last trusted
            # command here would itself be a memory-based protection action and
            # was found to mask sensor-dropout attacks in the 0000 baseline.
            # The fixed default therefore preserves the experiment interface
            # without silently adding a fifth protection mechanism.
            command_value = HOLD_SETPOINT_K
            baseline_fail_static_source = "fixed_neutral_degraded_default"
            baseline_fail_static_active = True
            self._event(
                events,
                time_s,
                "baseline_fail_static_activation",
                "A_primary",
                baseline_fail_static_source,
            )

        phi = int(valid.get(execution_path, False) or containment_active)
        execution_metric = self._execution_state(
            sensor=sensor,
            actuator=transport,
            health=health,
            command_value=command_value,
            execution_path=execution_path,
            containment_active=containment_active,
            protection_transient_active=self._protection_transient_active(time_s),
        )
        command = {
            SUPPLY_SETPOINT: HOLD_SETPOINT_K if command_value is None else float(command_value),
            SUPPLY_SETPOINT_ACTIVATE: float(command_value is not None),
        }
        return {
            "command": command,
            "selected_endpoint": endpoint,
            "selected_sensor": sensor,
            "active_path": execution_path,
            "next_path": self.active_path,
            "phi_observed": phi,
            **execution_metric,
            "valid_paths": valid,
            "events": events,
            "backup_used": bool(self.backup_used),
            "backup_available": bool(self.backup_available),
            "backup_sensor_health_observed": bool(self.backup_ok),
            "isolated": sorted(self.isolated),
            "untrusted": sorted(self.untrusted),
            "health": health,
            "primary_reference_residual_k": float(self.primary_reference_residual_k),
            "primary_reference_disagreement": bool(self.primary_reference_disagreement),
            "backup_reference_health_observed": bool(self.backup_reference_ok),
            "containment_active": bool(containment_active),
            "containment_source": containment_source,
            "baseline_fail_static_active": bool(baseline_fail_static_active),
            "baseline_fail_static_source": baseline_fail_static_source,
            "intended": dict(self.intended),
            "primary_control_state": self.control_state(self.intended.get("primary")),
            "guarded_control_state": self.control_state(self.intended.get("guarded")),
            "final_control_state": self.control_state(command_value),
            "fallback_command_source": self.fallback_source,
            "mechanism_state": {
                "isolation_due": dict(self.isolation_due),
                "untrusted_since": dict(self.untrusted_since),
                "healthy_since": dict(self.healthy_since),
                "recovery_probation_until": dict(self.recovery_probation_until),
                "red_due": self.red_due,
                "cfg_due": self.cfg_due,
                "backup_updated_s": self.backup_updated_s,
                "degradation_mode": dict(self.degradation_mode),
                "mechanism_attempt_count": dict(self.mechanism_attempt_count),
                "last_mechanism_attempt": dict(self.last_mechanism_attempt),
                "prepared_path": self.prepared_path,
                "protection_transient_until_s": self.protection_transient_until_s,
                "protection_transient_kind": self.protection_transient_kind,
            },
        }
