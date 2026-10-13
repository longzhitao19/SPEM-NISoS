"""Offline attack injection utilities for the BOPTEST experiment.

This module is deliberately separated from the online protection runtime.
It may read ChallengeSpec ground truth because it generates experimental
perturbations and scoring annotations. The defense runtime must never import
this module or ChallengeSpec.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .client import ZONE_TEMPERATURES
from .protocol import ChallengeSpec

COMMAND_CONTRACT_MIN_K = 285.15
COMMAND_CONTRACT_MAX_K = 308.15
COMMAND_UNSAFE_OFFSET_K = 2.0
CONTROL_LOW_SWITCH_K = 292.95
CONTROL_HIGH_SWITCH_K = 297.35
CONTROL_HEAT_K = 308.15
CONTROL_HOLD_K = 292.15
CONTROL_COOL_K = 285.15


def _controller_command(values: dict[str, float | None], min_valid: int = 3) -> float | None:
    valid = sorted(float(v) for v in values.values() if v is not None and np.isfinite(float(v)))
    if len(valid) < min_valid:
        return None
    center = float(np.median(valid))
    if center > CONTROL_HIGH_SWITCH_K:
        return CONTROL_COOL_K
    if center < CONTROL_LOW_SWITCH_K:
        return CONTROL_HEAT_K
    return CONTROL_HOLD_K


def adversarial_bias_sign(baseline_median_k: float | None, preferred_sign: int = 1) -> int:
    """Deterministic adversarial direction for sensing attacks.

    The pre-registered perturbation contract requires every delivered sensing
    attack to be capable of changing the downstream controller state.  A random
    sign can push the measured median *deeper* into the current control region,
    which made even the largest bias physically ineffective (v0.8.9 N=1
    screening: sensor_bias extreme with a -4.95 K bias left the controller in
    HEAT because the plant was already below the low switch point).  The
    attacker model therefore selects the direction that moves the un-attacked
    controller decision toward its nearest switch boundary.

    The rule uses only the measurements the attacker is replacing, so it
    depends on attack-channel information and not on any defense-internal
    state.  Magnitude remains frozen at ``normal_operating_range_k * intensity``
    and is never escalated: only the direction is chosen adversarially.
    """
    if baseline_median_k is None or not np.isfinite(float(baseline_median_k)):
        return int(preferred_sign)
    center = float(baseline_median_k)
    if center < CONTROL_LOW_SWITCH_K:
        # Controller is in HEAT; push the measurement up across the low switch.
        return 1
    if center > CONTROL_HIGH_SWITCH_K:
        # Controller is in COOL; push the measurement down across the high switch.
        return -1
    # Inside the hold band: cross the nearest switch boundary.
    distance_low = center - CONTROL_LOW_SWITCH_K
    distance_high = CONTROL_HIGH_SWITCH_K - center
    if abs(distance_low - distance_high) <= 1e-9:
        return int(preferred_sign)
    return -1 if distance_low < distance_high else 1


def adversarial_sensor_targets(
    observed: dict[str, float | None], sign: int, count: int
) -> list[str]:
    """Select which sensors a sensing attack compromises.

    The controller is a median voter over five zone measurements, so a bias on
    an arbitrary subset can leave the median almost unchanged: biasing the three
    *coldest* zones shifts the median by far less than the injected magnitude.
    The attacker therefore compromises the sensors on the side of the
    measurement distribution it is pushing the median toward, which moves the
    median by the full injected magnitude.

    Only the *realized* target identities are selected here.  The number of
    compromised sensors remains the pre-registered scope
    (``affected_sensor_count``), and the selection is a deterministic function
    of the measurements the attacker is already reading, not of defense state.
    """
    candidates = [name for name, value in observed.items() if value is not None and np.isfinite(float(value))]
    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda name: float(observed[name]), reverse=bool(sign > 0))
    return sorted(ranked[: max(0, min(int(count), len(ranked)))])


def _baseline_median_k(values: dict[str, float | None], min_valid: int = 3) -> float | None:
    valid = sorted(float(v) for v in values.values() if v is not None and np.isfinite(float(v)))
    if len(valid) < int(min_valid):
        return None
    return float(np.median(valid))


def _command_state(value: float | None) -> str:
    if value is None or not np.isfinite(float(value)):
        return "OFF"
    value = float(value)
    if abs(value - CONTROL_HEAT_K) < 1e-6:
        return "HEAT"
    if abs(value - CONTROL_HOLD_K) < 1e-6:
        return "HOLD"
    if abs(value - CONTROL_COOL_K) < 1e-6:
        return "COOL"
    return "MODULATED"


def attack_is_scheduled(challenge: ChallengeSpec, time_s: int) -> bool:
    """Whether the pre-registered attack attempt is inside its time window."""
    duration = max(0, int(challenge.attack_duration_s))
    return (
        challenge.attack_type != "benign"
        and duration > 0
        and int(challenge.attack_start_s) <= int(time_s) < int(challenge.attack_start_s) + duration
    )


def attack_is_active(challenge: ChallengeSpec, time_s: int) -> bool:
    """Whether a scheduled attack was delivered and is currently active.

    Controlled main/pre-experiments use attack_delivery_probability=1.0.  A
    dedicated robustness study may pre-register a lower delivery probability;
    the Bernoulli realization is sampled once when the challenge is created and
    is shared by all mechanism configurations for paired comparability.
    """
    return bool(attack_is_scheduled(challenge, time_s) and getattr(challenge, "attack_delivered", True))


def attack_fraction(challenge: ChallengeSpec, time_s: int, sample_s: int = 300) -> float:
    if not attack_is_active(challenge, time_s):
        return 0.0
    ramp_s = max(0, int(getattr(challenge, "attack_ramp_s", 0) or 0))
    if ramp_s <= 0:
        return 1.0
    elapsed = max(0, int(time_s) - int(challenge.attack_start_s))
    return float(min(1.0, max(0.0, (elapsed + sample_s) / ramp_s)))


def dropout_is_active(challenge: ChallengeSpec, time_s: int) -> bool:
    if not attack_is_active(challenge, time_s):
        return False
    pattern = getattr(challenge, "dropout_pattern", "none")
    if pattern != "intermittent":
        return True
    period_s = max(300, int(getattr(challenge, "dropout_period_s", 900) or 900))
    duty = min(1.0, max(0.0, float(getattr(challenge, "dropout_duty_cycle", 0.5) or 0.5)))
    elapsed = int(time_s) - int(challenge.attack_start_s)
    return (elapsed % period_s) < period_s * duty


def affected_sensors(challenge: ChallengeSpec) -> list[str]:
    raw = getattr(challenge, "affected_sensor_identity_json", "[]")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else list(raw)
    except Exception:
        parsed = []
    sensors = [str(name) for name in parsed if str(name) in ZONE_TEMPERATURES]
    if sensors:
        return sensors
    count = max(0, min(int(getattr(challenge, "affected_sensor_count", 0) or 0), len(ZONE_TEMPERATURES)))
    return list(ZONE_TEMPERATURES[:count])


def inject_sensor_measurements(
    true_values: dict[str, float],
    challenge: ChallengeSpec,
    time_s: int,
    sample_s: int = 300,
    frozen_bias_sign: int | None = None,
    frozen_sensor_targets: list[str] | None = None,
) -> tuple[dict[str, float | None], dict[str, Any]]:
    """Corrupt the controller-facing measurements for one simulation step.

    ``frozen_bias_sign`` and ``frozen_sensor_targets`` freeze the adversarial
    direction and the compromised sensor identities chosen at attack onset for
    the whole episode.  Freezing is required for paired comparability: both are
    decided once from the pre-attack operating point, so all 16 mechanism
    configurations of one challenge receive the identical injected
    perturbation.
    """
    observed: dict[str, float | None] = {name: float(true_values[name]) for name in ZONE_TEMPERATURES}
    active = attack_is_active(challenge, time_s)
    fraction = attack_fraction(challenge, time_s, sample_s)
    sensors = affected_sensors(challenge)
    baseline_command = _controller_command(observed)
    baseline_median_k = _baseline_median_k(observed)
    diag: dict[str, Any] = {
        "attack_attempted": attack_is_scheduled(challenge, time_s),
        "attack_delivered": bool(getattr(challenge, "attack_delivered", True)),
        "attack_active": active,
        "attack_fraction": fraction,
        "sensor_attack_changed": False,
        "sensor_bias_applied_k": 0.0,
        "dropout_count": 0,
        "affected_sensor_count": len(sensors),
        "baseline_controller_command_k": baseline_command,
        "attacked_controller_command_k": baseline_command,
        "sensor_control_state_changed": False,
        "baseline_control_state": _command_state(baseline_command),
        "attacked_control_state": _command_state(baseline_command),
        "baseline_median_k": baseline_median_k,
    }
    if active and challenge.attack_type in {"sensor_bias", "combined"}:
        nominal = abs(float(getattr(challenge, "bias_magnitude_k", 0.0) or 0.0))
        if nominal <= 1e-12:
            nominal = float(getattr(challenge, "normal_operating_range_k", 5.5) or 5.5) * float(challenge.intensity)
        nominal *= fraction
        preferred_sign = int(getattr(challenge, "bias_sign", 1) or 1)
        # Adversarial direction.  When the caller has already frozen the sign at
        # attack onset, reuse it; otherwise derive it from this step's un-biased
        # operating point.  Magnitude is never escalated here.
        active_sign = (
            int(frozen_bias_sign)
            if frozen_bias_sign
            else adversarial_bias_sign(baseline_median_k, preferred_sign)
        )
        # Adversarial targeting: compromise the sensors that determine the
        # median-voter output in the chosen direction.  The compromised count
        # stays at the pre-registered scope.
        if frozen_sensor_targets:
            sensors = [str(name) for name in frozen_sensor_targets if str(name) in observed]
        else:
            scope = int(getattr(challenge, "affected_sensor_count", 0) or 0) or len(sensors)
            sensors = adversarial_sensor_targets(observed, active_sign, scope)
        force_change = bool(getattr(challenge, "force_control_state_change", False))
        max_bias = max(nominal, float(getattr(challenge, "max_sensor_bias_k", 8.0) or 8.0) * max(fraction, 1e-9))

        def trial_with_bias(bias: float) -> tuple[dict[str, float | None], float | None]:
            trial = dict(observed)
            for sensor in sensors:
                if trial[sensor] is not None and np.isfinite(float(trial[sensor])):
                    trial[sensor] = float(trial[sensor]) + bias
            return trial, _controller_command(trial)

        chosen_bias = active_sign * nominal
        candidate, attacked_command = trial_with_bias(chosen_bias)
        if force_change and baseline_command == attacked_command and sensors and fraction > 0:
            # Deterministic state-forcing rule: search the smallest magnitude
            # (preferred sign first) that changes the controller state.  This is
            # part of the perturbation generator and is never visible to the
            # defense runtime.
            margin = max(0.1, float(getattr(challenge, "sensor_state_change_margin_k", 0.25) or 0.25))
            magnitudes = np.arange(max(0.25, nominal), max_bias + 0.001, 0.25)
            found = None
            for magnitude in magnitudes:
                for sign in (preferred_sign, -preferred_sign):
                    trial, cmd = trial_with_bias(float(sign) * float(magnitude + margin))
                    if cmd != baseline_command:
                        found = (float(sign) * float(magnitude + margin), trial, cmd)
                        break
                if found is not None:
                    break
            if found is not None:
                chosen_bias, candidate, attacked_command = found
        observed = candidate
        diag["sensor_attack_changed"] = bool(sensors and abs(chosen_bias) > 1e-9)
        diag["sensor_bias_applied_k"] = float(chosen_bias)
        diag["sensor_bias_sign"] = int(1 if chosen_bias >= 0 else -1)
        diag["sensor_bias_preferred_sign"] = int(preferred_sign)
        diag["sensor_target_identity_json"] = json.dumps(sorted(sensors))
        diag["attacked_controller_command_k"] = attacked_command
        diag["sensor_control_state_changed"] = bool(attacked_command != baseline_command)
        diag["attacked_control_state"] = _command_state(attacked_command)

    if active and challenge.attack_type == "sensor_dropout" and dropout_is_active(challenge, time_s):
        dropout_sensors = list(ZONE_TEMPERATURES) if bool(getattr(challenge, "communication_disruption", False)) else sensors
        for sensor in dropout_sensors:
            observed[sensor] = None
        attacked_command = _controller_command(observed)
        diag["sensor_attack_changed"] = bool(dropout_sensors)
        diag["dropout_count"] = len(dropout_sensors)
        diag["attacked_controller_command_k"] = attacked_command
        diag["sensor_control_state_changed"] = bool(attacked_command != baseline_command)
        diag["attacked_control_state"] = _command_state(attacked_command)
    return observed, diag

def inject_command_readbacks(
    intended: dict[str, float | None],
    challenge: ChallengeSpec,
    time_s: int,
    sample_s: int = 300,
    baseline_command: float | None = None,
) -> tuple[dict[str, float | None], dict[str, Any]]:
    """Corrupt the D->A transport readbacks for one simulation step.

    ``baseline_command`` is the controller decision computed from the
    *un-attacked* measurements of the same step.  For pure command injection it
    equals ``intended`` and changes nothing.  For a combined attack it is the
    reference the command component is displaced away from, so both components
    push the plant in the same direction instead of cancelling each other.
    """
    received = dict(intended)
    active = attack_is_active(challenge, time_s)
    fraction = attack_fraction(challenge, time_s, sample_s)
    baseline_primary = intended.get("primary")
    diag: dict[str, Any] = {
        "attack_attempted": attack_is_scheduled(challenge, time_s),
        "attack_delivered": bool(getattr(challenge, "attack_delivered", True)),
        "command_attack_changed": False,
        "command_attack_offset_k": 0.0,
        "command_attack_abs_offset_k": 0.0,
        "command_boundary_relevant": False,
        "command_contract_violation": False,
        "compromised_endpoints": [],
        "baseline_command_control_state": _command_state(baseline_primary),
        "attacked_command_control_state": _command_state(baseline_primary),
        "command_control_state_changed": False,
        "command_attack_saturated": False,
        "command_attack_reference_command_k": (
            float(baseline_command) if baseline_command is not None and np.isfinite(float(baseline_command)) else None
        ),
    }
    if not (active and challenge.attack_type in {"command_injection", "combined"}):
        return received, diag
    mode = str(getattr(challenge, "command_attack_mode", "bounded_opposite_extreme") or "bounded_opposite_extreme")
    configured_offset = float(getattr(challenge, "command_offset_k", 0.0) or 0.0)
    strength = float(getattr(challenge, "command_strength", 0.0) or 0.0)
    if strength <= 0.0:
        strength = min(0.95, max(0.10, float(challenge.intensity)))
    strength = min(0.95, max(0.0, strength)) * fraction
    endpoints = ["primary"]
    if int(getattr(challenge, "affected_actuator_count", 0) or 0) > 1:
        endpoints.append("guarded")
    changed_endpoints: list[str] = []
    contract_violation = False
    contract_saturated = False
    realized_offsets: list[float] = []
    for endpoint in endpoints:
        base = received.get(endpoint)
        if base is None:
            continue
        base = float(base)
        if mode == "bounded_opposite_extreme":
            midpoint = 0.5 * (COMMAND_CONTRACT_MIN_K + COMMAND_CONTRACT_MAX_K)
            # Reference the *un-attacked* controller decision when it is
            # available.  Using the sensor-biased ``base`` as the reference made
            # the two components of a combined attack oppose each other: the
            # sensing component moved the decision away from benign and the
            # command component then pulled it back toward the opposite extreme
            # of the already-deceived command (v0.8.9 N=1 screening: combined
            # moderate/strong unprotected deviations were ~1e-6 K, i.e. the
            # attack cancelled itself).
            reference = base
            if baseline_command is not None and np.isfinite(float(baseline_command)):
                reference = float(baseline_command)
            target = COMMAND_CONTRACT_MIN_K if reference >= midpoint else COMMAND_CONTRACT_MAX_K
            attacked = base + strength * (target - base)
            attacked = min(COMMAND_CONTRACT_MAX_K, max(COMMAND_CONTRACT_MIN_K, attacked))
            if abs(float(target) - float(base)) <= 1e-9:
                # The sensing component has already driven the command to the
                # opposite extreme.  The command component is still delivered
                # but cannot add further displacement inside the input contract.
                contract_saturated = True
        elif mode == "bounded_fixed_target":
            configured_target = float(getattr(challenge, "command_fixed_target_k", 0.0) or 0.0)
            if configured_target <= 0.0:
                configured_target = COMMAND_CONTRACT_MIN_K if int(getattr(challenge, "command_direction", -1) or -1) < 0 else COMMAND_CONTRACT_MAX_K
            target = min(COMMAND_CONTRACT_MAX_K, max(COMMAND_CONTRACT_MIN_K, configured_target))
            attacked = base + strength * (target - base)
            attacked = min(COMMAND_CONTRACT_MAX_K, max(COMMAND_CONTRACT_MIN_K, attacked))
        else:
            direction = int(getattr(challenge, "command_direction", 1) or 1)
            if abs(configured_offset) <= 1e-12:
                configured_offset = direction * (0.4 + 8.8 * float(challenge.intensity))
            attacked = base + configured_offset * fraction
        received[endpoint] = float(attacked)
        realized_offsets.append(float(attacked - base))
        if abs(attacked - base) > 1e-9:
            changed_endpoints.append(endpoint)
        contract_violation = contract_violation or not (COMMAND_CONTRACT_MIN_K <= attacked <= COMMAND_CONTRACT_MAX_K)
    offset = max(realized_offsets, key=abs) if realized_offsets else 0.0
    attacked_primary = received.get("primary")
    diag.update(
        {
            "command_attack_changed": bool(changed_endpoints and abs(offset) > 1e-9),
            "command_attack_saturated": bool(contract_saturated),
            "command_attack_offset_k": float(offset),
            "command_attack_abs_offset_k": float(abs(offset)),
            "command_boundary_relevant": bool(abs(offset) >= COMMAND_UNSAFE_OFFSET_K or contract_violation),
            "command_contract_violation": bool(contract_violation),
            "compromised_endpoints": changed_endpoints,
            "attacked_command_control_state": _command_state(attacked_primary),
            "command_control_state_changed": bool(_command_state(attacked_primary) != _command_state(baseline_primary)),
        }
    )
    return received, diag

