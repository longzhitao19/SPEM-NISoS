from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import VERSION
from .client import BOPTESTClient, BOPTESTError, SUPPLY_SETPOINT, SUPPLY_SETPOINT_ACTIVATE, ZONE_TEMPERATURES
from .fault_injection import (
    COMMAND_CONTRACT_MAX_K,
    COMMAND_CONTRACT_MIN_K,
    COMMAND_UNSAFE_OFFSET_K,
    attack_fraction,
    attack_is_active,
    attack_is_scheduled,
    inject_command_readbacks,
    inject_sensor_measurements,
)
from .observable_runtime_v6 import ObservableParameters, ObservableRuntime
from .topology import topology_rows
from .io_utils import write_frame
from .protocol import (
    TrialSpec,
    build_trials,
    build_trials_from_challenges,
    build_v2_challenges,
    build_preexperiment_challenges,
    PREEXPERIMENT_CONFIGS,
    MECHANISM_EXECUTION_PROBABILITIES,
    PREEXPERIMENT_POST_ATTACK_S,
    PREEXPERIMENT_TRIAL_DURATION_S,
    V2_FIXED_SCENARIO,
    V2_POST_ATTACK_OBSERVATION_S,
    V2_TRIAL_DURATION_S,
    V2_EVALUATION_WINDOW_S,
    canonical_hash,
    stable_seed,
    write_custom_manifests,
    write_manifests,
)

# Primary online security criterion. A transient in-contract command deviation is
# tolerated for at most two 5 min samples so delayed protection can act.
T_SAFE_MAX_S = 1800
T_UNTRUSTED_COMMAND_MAX_S = 600
# Unified execution threshold. The active/unrecovered states are at or below
# 0.40 and a fully executable chain is 1.00. This threshold is frozen before
# plant-side validation and never uses BOPTEST temperature outcomes.
PHI_EXEC_MIN = 0.60

# Physical criteria are reported as an independent external criterion and never
# enter the online protection runtime.
PHYSICAL_SAFE_MIN_K = 293.15
PHYSICAL_SAFE_MAX_K = 297.15
PHYSICAL_MAX_CONTINUOUS_VIOLATION_S = 1800
PHYSICAL_RECOVERY_REQUIRED_FRACTION = 0.95


def _independent_backup_observation(true_values: dict[str, float], challenge_id: str) -> dict[str, float]:
    """Simulate S_zone_backup as an independent redundant sensor channel.

    The backup channel measures the same plant state with a small fixed
    calibration offset (<=0.03 K per zone).  Its read-only residual is available
    to the common state-awareness layer in every configuration, while active
    substitution into control is enabled only by M_red.  It contains no attack
    labels, injected offsets, future information or scoring results.  The fixed
    offset is challenge-deterministic so all paired mechanism configurations
    receive exactly the same reference measurements.
    """
    out: dict[str, float] = {}
    for name in ZONE_TEMPERATURES:
        u = stable_seed(f"backup|{challenge_id}|{name}") / float(0xFFFFFFFF)
        calibration = (u - 0.5) * 0.06
        out[name] = float(true_values[name]) + calibration
    return out


def _control_state(command: float | None, activate: float = 1.0) -> str:
    if activate <= 0.0 or command is None or not np.isfinite(float(command)):
        return "OFF"
    return ObservableRuntime.control_state(float(command))


def prepare_trial(client: BOPTESTClient, spec: TrialSpec, step_s: int) -> tuple[str, dict[str, Any]]:
    test_id = client.select()
    try:
        version = client.version()
        inputs = client.inputs(test_id)
        measurements = client.measurements(test_id)
        missing = {SUPPLY_SETPOINT, SUPPLY_SETPOINT_ACTIVATE}.difference(inputs) | set(ZONE_TEMPERATURES).difference(measurements)
        if missing:
            raise BOPTESTError(f"test case lacks registered I/O: {sorted(missing)}")
        if int(getattr(spec.challenge, "boptest_start_time_s", 0) or 0) > 0:
            initialization_meta = client.initialize(
                test_id,
                int(spec.challenge.boptest_start_time_s),
                int(spec.challenge.boptest_warmup_period_s),
            )
            scenario_meta = {
                "scenario_label": spec.challenge.scenario,
                "initialization_mode": "direct_target_day_window",
                "start_time_s": int(spec.challenge.boptest_start_time_s),
                "warmup_period_s": int(spec.challenge.boptest_warmup_period_s),
                "initialize_payload": initialization_meta,
            }
        else:
            scenario_meta = client.scenario(test_id, spec.challenge.scenario, spec.challenge.seed)
        step_meta = client.step(test_id, step_s)
        # Log available supply-air measurements so command attacks can be verified
        # against plant-side realized temperatures rather than only the written
        # overwrite signal. This discovery uses only the public measurements API.
        diagnostic_measurements = sorted(
            name for name in measurements
            if ("TSup" in name or "TSupAir" in name) and name not in ZONE_TEMPERATURES
        )[:12]
        return test_id, {
            "boptest_version": version,
            "inputs": inputs,
            "measurements": measurements,
            "diagnostic_measurements": diagnostic_measurements,
            "scenario": scenario_meta,
            "step": step_meta,
        }
    except Exception:
        # Important for parallel runs: never leak a provisioned test instance if
        # setup fails before run_trial enters its normal try/finally lifecycle.
        client.stop(test_id)
        raise


def _max_consecutive_true(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in mask.astype(bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _number_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _zone_columns(prefix: str, values: dict[str, Any]) -> dict[str, float]:
    """Flatten five-zone measurements into explicit numeric time-series columns."""
    return {f"{prefix}{name}": _number_or_nan(values.get(name)) for name in ZONE_TEMPERATURES}


def _physical_step_state(values: dict[str, Any]) -> dict[str, Any]:
    temperatures = np.asarray([_number_or_nan(values.get(name)) for name in ZONE_TEMPERATURES], dtype=float)
    finite = np.isfinite(temperatures)
    if not np.all(finite):
        return {
            "all_zone_temperatures_available": False,
            "any_zone_outside_safe_envelope": None,
            "zone_outside_safe_envelope_count": None,
            "mean_abs_temperature_deviation_k": float("nan"),
            "max_abs_temperature_deviation_k": float("nan"),
        }
    violation = (temperatures < PHYSICAL_SAFE_MIN_K) | (temperatures > PHYSICAL_SAFE_MAX_K)
    deviation = np.abs(temperatures - 295.15)
    return {
        "all_zone_temperatures_available": True,
        "any_zone_outside_safe_envelope": bool(np.any(violation)),
        "zone_outside_safe_envelope_count": int(np.sum(violation)),
        "mean_abs_temperature_deviation_k": float(np.mean(deviation)),
        "max_abs_temperature_deviation_k": float(np.max(deviation)),
    }


def _event_timing_summary(event_frame: pd.DataFrame) -> pd.DataFrame:
    """Return one row per trial with first occurrence times for key experiment events."""
    if event_frame.empty:
        return pd.DataFrame()
    event_names = [
        "attack_attempt_start_truth",
        "attack_start_truth",
        "attack_attempt_end_truth",
        "attack_end_truth",
        "anomaly_detected",
        "element_untrusted",
        "isolation_start",
        "isolation_complete",
        "redundancy_start",
        "redundancy_complete",
        "reconfiguration_start",
        "reconfiguration_complete",
        "recovery_verification",
        "recovery_probation_start",
        "recovery_probation_failed",
        "recovery_complete",
        "safe_rejoin",
        "primary_path_rejoin",
        "boundary_breach",
    ]
    subset = event_frame[event_frame["event_type"].isin(event_names)].copy()
    if subset.empty:
        return pd.DataFrame()
    timing = subset.groupby(["trial_id", "challenge_id", "mechanism_config", "event_type"], as_index=False)["time_s"].min()
    wide = timing.pivot(
        index=["trial_id", "challenge_id", "mechanism_config"],
        columns="event_type",
        values="time_s",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={name: f"first_{name}_s" for name in event_names if name in wide.columns})
    start = "first_attack_start_truth_s"
    end = "first_attack_end_truth_s"
    if start in wide.columns and "first_anomaly_detected_s" in wide.columns:
        wide["detection_delay_from_attack_start_s"] = wide["first_anomaly_detected_s"] - wide[start]
    if start in wide.columns and "first_isolation_complete_s" in wide.columns:
        wide["isolation_complete_delay_from_attack_start_s"] = wide["first_isolation_complete_s"] - wide[start]
    if start in wide.columns and "first_reconfiguration_complete_s" in wide.columns:
        wide["reconfiguration_complete_delay_from_attack_start_s"] = wide["first_reconfiguration_complete_s"] - wide[start]
    if end in wide.columns and "first_recovery_complete_s" in wide.columns:
        wide["recovery_complete_delay_from_attack_end_s"] = wide["first_recovery_complete_s"] - wide[end]
    if end in wide.columns and "first_primary_path_rejoin_s" in wide.columns:
        wide["primary_rejoin_delay_from_attack_end_s"] = wide["first_primary_path_rejoin_s"] - wide[end]
    return wide


def evaluate_physical_from_steps(steps: list[dict[str, Any]], kpi: Any, trial_id: str, step_s: int) -> dict[str, Any]:
    """Evaluate plant behavior only over the online experiment window.

    This intentionally avoids ``/results(..., 0, huge)`` because that request
    also returned the BOPTEST warm-up trajectory and made a 4 h experiment look
    like several days of physical violation.  Absolute 20--24 C envelope fields
    are retained only as diagnostics; baseline-relative trajectory effects are
    computed later against the benign reference.
    """
    if not steps:
        raise BOPTESTError("online experiment produced no synchronized step rows")
    temp_cols = [f"plant_post_{name}" for name in ZONE_TEMPERATURES]
    temperatures = np.asarray([[row.get(col, np.nan) for col in temp_cols] for row in steps], dtype=float)
    if temperatures.size == 0 or not np.isfinite(temperatures).all():
        raise BOPTESTError("online experiment window contains missing zone temperatures")
    deviation = np.abs(temperatures - 295.15)
    violation = (temperatures < PHYSICAL_SAFE_MIN_K) | (temperatures > PHYSICAL_SAFE_MAX_K)
    any_zone_violation = np.any(violation, axis=1)
    n = temperatures.shape[0]
    late_start = int(0.75 * n)
    late_violation = violation[late_start:, :]
    max_consecutive_violation_s = _max_consecutive_true(any_zone_violation) * int(step_s)
    envelope_fraction = float(1.0 - np.mean(violation))
    late_envelope_fraction = float(1.0 - np.mean(late_violation)) if late_violation.size else envelope_fraction
    physical_constraint_met = int(max_consecutive_violation_s <= PHYSICAL_MAX_CONTINUOUS_VIOLATION_S)
    physical_recovery_met = int(late_envelope_fraction >= PHYSICAL_RECOVERY_REQUIRED_FRACTION)
    return {
        "trial_id": trial_id,
        "online_window_samples": int(n),
        "online_window_duration_s": int(n * step_s),
        "mean_zone_temperature_k": float(np.mean(temperatures)),
        "mean_abs_temperature_deviation_k": float(np.mean(deviation)),
        "max_abs_temperature_deviation_k": float(np.max(deviation)),
        "zone_time_outside_safe_envelope_fraction": float(np.mean(violation)),
        "comfort_violation_fraction": float(np.mean(violation)),
        "time_outside_safe_envelope_fraction": float(np.mean(violation)),
        "any_zone_outside_safe_envelope_fraction": float(np.mean(any_zone_violation)),
        "safe_envelope_fraction": envelope_fraction,
        "max_consecutive_any_zone_violation_s": int(max_consecutive_violation_s),
        "physical_recovery_envelope_fraction_last_quarter": late_envelope_fraction,
        "physical_constraint_met_absolute_diagnostic": physical_constraint_met,
        "physical_recovery_met_absolute_diagnostic": physical_recovery_met,
        "Y_physical_absolute_diagnostic": int(bool(physical_constraint_met and physical_recovery_met)),
        # No formal binary physical success is assigned here.  The fixed
        # 20--24 C envelope can be inappropriate for an unoccupied BOPTEST
        # period.  Formal physical validation is baseline-relative and is
        # computed after all paired trials are available.
        "Y_physical": float("nan"),
        "safe_envelope_min_k": PHYSICAL_SAFE_MIN_K,
        "safe_envelope_max_k": PHYSICAL_SAFE_MAX_K,
        "physical_max_continuous_violation_s": PHYSICAL_MAX_CONTINUOUS_VIOLATION_S,
        "physical_recovery_required_fraction": PHYSICAL_RECOVERY_REQUIRED_FRACTION,
        "physical_window_scope": "online_event_window_only",
        "boptest_kpi_json": json.dumps(kpi, sort_keys=True, ensure_ascii=False),
        "sealed_after_security_outcome": True,
    }


def _selected_command_is_compromised(
    decision: dict[str, Any], received: dict[str, float | None], command_diag: dict[str, Any]
) -> tuple[bool, bool]:
    # A successful isolation hold is sourced from trusted local state, not from
    # the compromised transport readback.  Do not classify it as attack ingress
    # even if the numeric held value happens to equal the malicious readback.
    if bool(decision.get("containment_active", False)):
        return False, False
    endpoint = str(decision["selected_endpoint"])
    compromised = endpoint in set(command_diag.get("compromised_endpoints", []))
    if not compromised or not bool(command_diag.get("command_boundary_relevant", False)):
        return False, False
    if float(decision["command"].get(SUPPLY_SETPOINT_ACTIVATE, 0.0)) <= 0.0:
        return False, False
    selected_readback = received.get(endpoint)
    if selected_readback is None:
        return False, False
    final_value = float(decision["command"][SUPPLY_SETPOINT])
    entered = abs(final_value - float(selected_readback)) < 1e-9
    hard = entered and not (COMMAND_CONTRACT_MIN_K <= final_value <= COMMAND_CONTRACT_MAX_K)
    return bool(entered), bool(hard)


def _sensor_corruption_reached_control(
    decision: dict[str, Any], true_values: dict[str, float], sensor_diag: dict[str, Any]
) -> tuple[bool, float]:
    if not bool(sensor_diag.get("sensor_attack_changed", False)):
        return False, 0.0
    # Fail-static and isolation holds deliberately avoid consuming the current
    # corrupted sensor sample.  Stale-control consequences are accounted for by
    # Phi_exec/continuity and the independent plant diagnostics, not as direct
    # sensor-corruption ingress.
    if bool(decision.get("containment_active", False)) or bool(decision.get("baseline_fail_static_active", False)):
        return False, 0.0
    if decision.get("selected_sensor") != "S_zone_primary" or decision.get("selected_endpoint") != "primary":
        return False, 0.0
    if float(decision["command"].get(SUPPLY_SETPOINT_ACTIVATE, 0.0)) <= 0.0:
        return False, 0.0
    oracle = ObservableRuntime.controller({name: true_values[name] for name in ZONE_TEMPERATURES})
    if oracle is None:
        return False, 0.0
    final_value = float(decision["command"][SUPPLY_SETPOINT])
    deviation = abs(final_value - float(oracle))
    return bool(deviation >= COMMAND_UNSAFE_OFFSET_K), float(deviation)


def run_trial(
    client: BOPTESTClient,
    spec: TrialSpec,
    step_s: int = 300,
    trial_duration_s: int = V2_TRIAL_DURATION_S,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if step_s <= 0:
        raise ValueError("step_s must be positive")
    if step_s > 300:
        raise ValueError("step_s must be <= 300 s so 300/600 s mechanism delays are temporally observable")

    test_id, metadata = prepare_trial(client, spec, step_s)
    runtime = ObservableRuntime(spec.mechanism, parameters=ObservableParameters(sample_s=step_s), stochastic_seed=int(spec.challenge.seed))
    events: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    initialize_payload = metadata.get("scenario", {}).get("initialize_payload") if isinstance(metadata.get("scenario"), dict) else None
    if isinstance(initialize_payload, dict) and all(name in initialize_payload for name in ZONE_TEMPERATURES):
        response = dict(initialize_payload)
    else:
        response = {name: math.nan for name in ZONE_TEMPERATURES}

    b_unsafe = 0
    b_continuity = 0
    boundary_time_s: int | None = None
    unsafe_exposure_s = 0
    max_unsafe_exposure_s = 0
    chain_lost_since_s: int | None = None
    max_chain_interruption_s = 0
    phi_exec_values: list[float] = []
    phi_exec_evaluation_area_s = 0.0
    phi_exec_evaluation_duration_s = 0
    phi_exec_attack_area_s = 0.0
    phi_exec_attack_duration_s = 0
    phi_exec_post_area_s = 0.0
    phi_exec_post_duration_s = 0
    time_below_phi_exec_min_s = 0
    attack_start_logged = False
    attack_end_logged = False
    # Formal V2 challenges use one common attack-relative evaluation horizon.
    # Randomized attack onset therefore changes the physical clock time but does
    # not change the amount of recovery time entering Q_exec.  Pre-experiment
    # and legacy runs retain their full remaining trial window.
    if getattr(spec.challenge, "challenge_family", "") == "v2_event_window" and spec.challenge.attack_type != "benign":
        phi_exec_evaluation_target_s = int(V2_EVALUATION_WINDOW_S)
    else:
        phi_exec_evaluation_target_s = max(0, int(trial_duration_s) - int(spec.challenge.attack_start_s))
    phi_exec_evaluation_end_s = min(
        int(trial_duration_s),
        int(spec.challenge.attack_start_s) + int(phi_exec_evaluation_target_s),
    )
    saw_sensor_effect = spec.challenge.attack_type in {"benign", "command_injection"}
    saw_command_effect = spec.challenge.attack_type in {"benign", "sensor_bias", "sensor_dropout"}
    # The adversarial bias direction is decided once, from the pre-attack
    # operating point, and then held fixed for the whole attack episode.  This
    # keeps the injected perturbation identical across the 16 mechanism
    # configurations of one challenge, which is required for paired comparisons.
    frozen_bias_sign: int | None = None
    frozen_sensor_targets: list[str] | None = None
    saw_target_control_state_change = spec.challenge.attack_type == "benign"
    saw_unprotected_final_control_state_change = spec.challenge.attack_type == "benign"

    try:
        for time_s in range(0, trial_duration_s, step_s):
            true_values = {
                name: float(response.get(name, math.nan))
                for name in ZONE_TEMPERATURES
                if np.isfinite(response.get(name, math.nan))
            }
            # The attack windows begin no earlier than 06:00. This initial value
            # is therefore only a warm-start placeholder before any attack.
            if len(true_values) != len(ZONE_TEMPERATURES):
                true_values = {name: 295.15 for name in ZONE_TEMPERATURES}

            attempted = attack_is_scheduled(spec.challenge, time_s)
            active = attack_is_active(spec.challenge, time_s)
            if attempted and not attack_start_logged:
                attack_start_logged = True
                events.append({
                    "time_s": time_s, "event_type": "attack_attempt_start_truth",
                    "component": spec.challenge.attack_target, "detail": spec.challenge.attack_type,
                    "event_source": "experiment_truth",
                })
                if bool(getattr(spec.challenge, "attack_delivered", True)):
                    events.append({
                        "time_s": time_s, "event_type": "attack_start_truth",
                        "component": spec.challenge.attack_target, "detail": spec.challenge.attack_type,
                        "event_source": "experiment_truth",
                    })
                else:
                    events.append({
                        "time_s": time_s, "event_type": "attack_delivery_failed_truth",
                        "component": spec.challenge.attack_target, "detail": spec.challenge.attack_type,
                        "event_source": "experiment_truth",
                    })
            if attack_start_logged and not attempted and not attack_end_logged and time_s >= spec.challenge.attack_start_s:
                attack_end_logged = True
                events.append({
                    "time_s": time_s, "event_type": "attack_attempt_end_truth",
                    "component": spec.challenge.attack_target, "detail": spec.challenge.attack_type,
                    "event_source": "experiment_truth",
                })
                if bool(getattr(spec.challenge, "attack_delivered", True)):
                    events.append({
                        "time_s": time_s, "event_type": "attack_end_truth",
                        "component": spec.challenge.attack_target, "detail": spec.challenge.attack_type,
                        "event_source": "experiment_truth",
                    })

            observed, sensor_diag = inject_sensor_measurements(
                true_values,
                spec.challenge,
                time_s,
                step_s,
                frozen_bias_sign=frozen_bias_sign,
                frozen_sensor_targets=frozen_sensor_targets,
            )
            if frozen_bias_sign is None and sensor_diag.get("sensor_bias_sign") is not None:
                frozen_bias_sign = int(sensor_diag["sensor_bias_sign"])
                targets_json = str(sensor_diag.get("sensor_target_identity_json") or "[]")
                try:
                    frozen_sensor_targets = [str(name) for name in json.loads(targets_json)]
                except (TypeError, ValueError):
                    frozen_sensor_targets = None
                events.append({
                    "time_s": time_s,
                    "event_type": "sensor_bias_direction_frozen_truth",
                    "component": "S_zone_primary",
                    "detail": (
                        f"sign={frozen_bias_sign};baseline_median_k="
                        f"{sensor_diag.get('baseline_median_k')};preferred_sign="
                        f"{sensor_diag.get('sensor_bias_preferred_sign')};targets="
                        f"{sensor_diag.get('sensor_target_identity_json')}"
                    ),
                    "event_source": "experiment_truth",
                })
            if sensor_diag["sensor_attack_changed"]:
                saw_sensor_effect = True
                events.append(
                    {
                        "time_s": time_s,
                        "event_type": "sensor_attack_entered_controller_measurement",
                        "component": "S_zone_primary",
                        "detail": spec.challenge.attack_type,
                        "event_source": "experiment_truth",
                        "bias_k": sensor_diag["sensor_bias_applied_k"],
                        "bias_sign": sensor_diag.get("sensor_bias_sign"),
                        "targets": sensor_diag.get("sensor_target_identity_json"),
                        "dropout_count": sensor_diag["dropout_count"],
                    }
                )

            backup_observed = _independent_backup_observation(true_values, spec.challenge.challenge_id)
            if bool(sensor_diag.get("sensor_control_state_changed", False)):
                saw_target_control_state_change = True
                events.append({
                    "time_s": time_s,
                    "event_type": "attack_changed_controller_state_truth",
                    "component": "D_supervisory",
                    "detail": f"{sensor_diag.get('baseline_control_state')}->{sensor_diag.get('attacked_control_state')}",
                    "event_source": "experiment_truth",
                })

            intended = runtime.prepare(observed, time_s, backup_observed=backup_observed)
            received, command_diag = inject_command_readbacks(
                intended,
                spec.challenge,
                time_s,
                step_s,
                baseline_command=sensor_diag.get("baseline_controller_command_k"),
            )
            # A combined attack whose sensing component has already driven the
            # command to the opposite contract extreme produces zero incremental
            # displacement.  The component is still delivered, so it counts as
            # entering the transport rather than as an undelivered attack.
            if command_diag.get("command_attack_saturated", False):
                saw_command_effect = True
            if command_diag["command_attack_changed"]:
                saw_command_effect = True
                events.append(
                    {
                        "time_s": time_s,
                        "event_type": "command_injection_entered_transport",
                        "component": "e_DA_primary",
                        "detail": spec.challenge.attack_type,
                        "event_source": "experiment_truth",
                        "offset_k": command_diag["command_attack_offset_k"],
                        "boundary_relevant": command_diag["command_boundary_relevant"],
                        "compromised_endpoints": json.dumps(command_diag["compromised_endpoints"]),
                    }
                )
            if bool(command_diag.get("command_control_state_changed", False)):
                saw_target_control_state_change = True
                events.append({
                    "time_s": time_s,
                    "event_type": "attack_changed_controller_state_truth",
                    "component": "e_DA_primary",
                    "detail": f"{command_diag.get('baseline_command_control_state')}->{command_diag.get('attacked_command_control_state')}",
                    "event_source": "experiment_truth",
                })

            decision = runtime.finish(received, time_s)
            for event in decision["events"]:
                event.setdefault("event_source", "online_defense")
            events.extend(decision["events"])

            oracle_command = ObservableRuntime.controller(true_values)
            oracle_state = _control_state(oracle_command, 1.0 if oracle_command is not None else 0.0)
            final_state = _control_state(
                decision["command"].get(SUPPLY_SETPOINT),
                float(decision["command"].get(SUPPLY_SETPOINT_ACTIVATE, 0.0)),
            )
            if active and spec.mechanism.config_id == "0000" and final_state != oracle_state:
                saw_unprotected_final_control_state_change = True

            command_compromised, hard_command_violation = _selected_command_is_compromised(decision, received, command_diag)
            sensor_compromised, sensor_command_deviation_k = _sensor_corruption_reached_control(decision, true_values, sensor_diag)
            unsafe_this_step = bool(command_compromised or sensor_compromised)
            if unsafe_this_step:
                unsafe_exposure_s += step_s
                max_unsafe_exposure_s = max(max_unsafe_exposure_s, unsafe_exposure_s)
            else:
                unsafe_exposure_s = 0

            if not b_unsafe and (hard_command_violation or unsafe_exposure_s > T_UNTRUSTED_COMMAND_MAX_S):
                b_unsafe = 1
                boundary_time_s = time_s
                events.append(
                    {
                        "time_s": time_s,
                        "event_type": "boundary_breach",
                        "component": "B_unsafe",
                        "detail": "hard_contract_violation" if hard_command_violation else "untrusted_command_exposure_exceeded_budget",
                        "event_source": "offline_scoring",
                    }
                )

            phi_sys = int(decision["phi_observed"])
            phi_exec = float(decision["phi_exec"])
            phi_exec_values.append(phi_exec)
            attack_end_eval_s = int(spec.challenge.attack_start_s + max(0, spec.challenge.attack_duration_s))
            if int(spec.challenge.attack_start_s) <= time_s < phi_exec_evaluation_end_s:
                phi_exec_evaluation_area_s += phi_exec * step_s
                phi_exec_evaluation_duration_s += step_s
                if time_s < attack_end_eval_s:
                    phi_exec_attack_area_s += phi_exec * step_s
                    phi_exec_attack_duration_s += step_s
                else:
                    phi_exec_post_area_s += phi_exec * step_s
                    phi_exec_post_duration_s += step_s
            below_exec_threshold = bool(phi_exec < PHI_EXEC_MIN)
            if below_exec_threshold:
                time_below_phi_exec_min_s += step_s
                if chain_lost_since_s is None:
                    chain_lost_since_s = time_s
                    events.append(
                        {
                            "time_s": time_s,
                            "event_type": "execution_degraded",
                            "component": "Phi_exec",
                            "detail": f"phi_exec={phi_exec:.3f}<phi_min={PHI_EXEC_MIN:.3f};state={decision.get('execution_state')}",
                            "event_source": "offline_scoring",
                        }
                    )
                current_interruption = time_s + step_s - chain_lost_since_s
                max_chain_interruption_s = max(max_chain_interruption_s, current_interruption)
                if not b_continuity and current_interruption > T_SAFE_MAX_S:
                    b_continuity = 1
                    boundary_time_s = time_s if boundary_time_s is None else boundary_time_s
                    events.append(
                        {
                            "time_s": time_s,
                            "event_type": "boundary_breach",
                            "component": "B_continuity",
                            "detail": "Phi_exec_below_threshold_exceeded_T_safe_max",
                            "event_source": "offline_scoring",
                        }
                    )
            elif chain_lost_since_s is not None:
                interruption = time_s - chain_lost_since_s
                max_chain_interruption_s = max(max_chain_interruption_s, interruption)
                events.append(
                    {
                        "time_s": time_s,
                        "event_type": "execution_restored",
                        "component": "Phi_exec",
                        "detail": f"interruption_s={interruption};phi_exec={phi_exec:.3f}",
                        "event_source": "offline_scoring",
                    }
                )
                chain_lost_since_s = None

            response = client.advance(test_id, decision["command"])
            next_true_values = {name: _number_or_nan(response.get(name)) for name in ZONE_TEMPERATURES}
            pre_physical = _physical_step_state(true_values)
            post_physical = _physical_step_state(next_true_values)
            mechanism_state = dict(decision.get("mechanism_state", {}))
            current_chain_interruption_s = 0 if chain_lost_since_s is None else int(time_s + step_s - chain_lost_since_s)
            attack_end_s = int(spec.challenge.attack_start_s + max(0, spec.challenge.attack_duration_s))
            row = {
                "trial_id": spec.trial_id,
                "challenge_id": spec.challenge.challenge_id,
                "mechanism_config": spec.mechanism.config_id,
                "split": spec.challenge.split,
                "scenario": spec.challenge.scenario,
                "attack_type_truth": spec.challenge.attack_type,
                "disturbance_class_truth": getattr(spec.challenge, "disturbance_class", "B"),
                "intensity_label_truth": spec.challenge.intensity_label,
                "intensity_truth": float(spec.challenge.intensity),
                "attack_target_truth": spec.challenge.attack_target,
                "attack_start_s_truth": int(spec.challenge.attack_start_s),
                "attack_end_s_truth": attack_end_s,
                "attack_duration_s_truth": int(spec.challenge.attack_duration_s),
                "phi_exec_evaluation_end_s_truth": int(phi_exec_evaluation_end_s),
                "phi_exec_evaluation_window_target_s_truth": int(phi_exec_evaluation_target_s),
                "boptest_start_time_s_truth": int(getattr(spec.challenge, "boptest_start_time_s", 0) or 0),
                "boptest_warmup_period_s_truth": int(getattr(spec.challenge, "boptest_warmup_period_s", 0) or 0),
                "time_s": int(time_s),
                "next_time_s": int(time_s + step_s),
                "boptest_absolute_time_s": int(getattr(spec.challenge, "boptest_start_time_s", 0) or 0) + int(time_s),
                "boptest_next_absolute_time_s": int(getattr(spec.challenge, "boptest_start_time_s", 0) or 0) + int(time_s + step_s),
                "attack_attempted_truth": bool(attempted),
                "attack_active_truth": bool(active),
                "attack_delivery_probability_truth": float(getattr(spec.challenge, "attack_delivery_probability", 1.0)),
                "attack_delivery_draw_truth": float(getattr(spec.challenge, "attack_delivery_draw", 0.0)),
                "attack_delivered_truth": bool(getattr(spec.challenge, "attack_delivered", True)),
                "attack_fraction_truth": float(attack_fraction(spec.challenge, time_s, step_s)),
                "time_from_attack_start_s_truth": int(time_s - spec.challenge.attack_start_s),
                "final_command_k": float(decision["command"][SUPPLY_SETPOINT]),
                "final_command_activate": float(decision["command"][SUPPLY_SETPOINT_ACTIVATE]),
                "oracle_benign_command_k_truth": _number_or_nan(oracle_command),
                "oracle_benign_control_state_truth": oracle_state,
                "final_control_state": final_state,
                "primary_control_state": decision.get("primary_control_state"),
                "guarded_control_state": decision.get("guarded_control_state"),
                "fallback_command_source": decision.get("fallback_command_source"),
                "intended_primary_command_k": _number_or_nan(intended.get("primary")),
                "intended_guarded_command_k": _number_or_nan(intended.get("guarded")),
                "received_primary_command_k": _number_or_nan(received.get("primary")),
                "received_guarded_command_k": _number_or_nan(received.get("guarded")),
                "phi_sys": phi_sys,
                "b_exec": float(decision["b_exec"]),
                "a_exec": float(decision["a_exec"]),
                "s_exec": float(decision["s_exec"]),
                "L_exec_s": int(decision["L_exec_s"]),
                "L_exec_norm": float(decision["L_exec_norm"]),
                "L_exec_limit_s": int(decision["L_exec_limit_s"]),
                "phi_exec": phi_exec,
                "phi_exec_min": PHI_EXEC_MIN,
                "phi_exec_safe": int(not below_exec_threshold),
                "execution_state": decision["execution_state"],
                "active_safe_path": decision["active_path"],
                "next_safe_path": decision.get("next_path", decision["active_path"]),
                "selected_sensor": decision["selected_sensor"],
                "selected_endpoint": decision["selected_endpoint"],
                "protection_transient_active": bool(decision.get("protection_transient_active", False)),
                "protection_transient_kind": decision.get("protection_transient_kind"),
                "protection_transient_until_s": decision.get("protection_transient_until_s"),
                "backup_available": bool(decision["backup_available"]),
                "backup_used": bool(decision["backup_used"]),
                "backup_sensor_health_observed": bool(decision.get("backup_sensor_health_observed", False)),
                "backup_reference_health_observed": bool(decision.get("backup_reference_health_observed", False)),
                "sensor_health_observed": bool(decision.get("health", {}).get("S_zone_primary", False)),
                "primary_reference_residual_k": float(decision.get("primary_reference_residual_k", 0.0)),
                "primary_reference_disagreement": bool(decision.get("primary_reference_disagreement", False)),
                # Historical column names are retained for file compatibility,
                # but they now report command-transport integrity because the
                # injected command attack is on e_DA, not on the actuator body.
                "primary_actuator_health_observed": bool(decision.get("health", {}).get("e_DA_primary", False)),
                "guarded_actuator_health_observed": decision.get("health", {}).get("e_DA_guarded", None),
                "primary_command_channel_health_observed": bool(decision.get("health", {}).get("e_DA_primary", False)),
                "guarded_command_channel_health_observed": decision.get("health", {}).get("e_DA_guarded", None),
                "isolation_containment_active": bool(decision.get("containment_active", False)),
                "isolation_containment_source": decision.get("containment_source", "none"),
                "baseline_fail_static_active": bool(decision.get("baseline_fail_static_active", False)),
                "baseline_fail_static_source": decision.get("baseline_fail_static_source", "none"),
                "isolated_json": json.dumps(decision["isolated"]),
                "untrusted_json": json.dumps(decision["untrusted"]),
                "isolation_pending_json": json.dumps(mechanism_state.get("isolation_due", {}), sort_keys=True),
                "recovery_pending_json": json.dumps(mechanism_state.get("untrusted_since", {}), sort_keys=True),
                "healthy_since_json": json.dumps(mechanism_state.get("healthy_since", {}), sort_keys=True),
                "recovery_probation_until_json": json.dumps(mechanism_state.get("recovery_probation_until", {}), sort_keys=True),
                "redundancy_due_s": mechanism_state.get("red_due"),
                "reconfiguration_due_s": mechanism_state.get("cfg_due"),
                "backup_updated_s": mechanism_state.get("backup_updated_s"),
                "mechanism_attempt_count_json": json.dumps(mechanism_state.get("mechanism_attempt_count", {}), sort_keys=True),
                "last_mechanism_attempt_json": json.dumps(mechanism_state.get("last_mechanism_attempt", {}), sort_keys=True),
                "sensor_attack_changed_truth": bool(sensor_diag["sensor_attack_changed"]),
                "sensor_bias_applied_k_truth": float(sensor_diag["sensor_bias_applied_k"]),
                "sensor_bias_sign_truth": sensor_diag.get("sensor_bias_sign"),
                "sensor_target_identity_json_truth": str(sensor_diag.get("sensor_target_identity_json") or ""),
                "baseline_median_k_truth": _number_or_nan(sensor_diag.get("baseline_median_k")),
                "dropout_count_truth": int(sensor_diag["dropout_count"]),
                "sensor_control_state_changed_truth": bool(sensor_diag.get("sensor_control_state_changed", False)),
                "sensor_baseline_control_state_truth": sensor_diag.get("baseline_control_state"),
                "sensor_attacked_control_state_truth": sensor_diag.get("attacked_control_state"),
                "command_attack_changed_truth": bool(command_diag["command_attack_changed"]),
                "command_control_state_changed_truth": bool(command_diag.get("command_control_state_changed", False)),
                "command_baseline_control_state_truth": command_diag.get("baseline_command_control_state"),
                "command_attacked_control_state_truth": command_diag.get("attacked_command_control_state"),
                "attack_changed_target_control_state_truth": bool(
                    sensor_diag.get("sensor_control_state_changed", False)
                    or command_diag.get("command_control_state_changed", False)
                ),
                "command_attack_offset_k_truth": float(command_diag["command_attack_offset_k"]),
                "command_attack_saturated_truth": bool(command_diag.get("command_attack_saturated", False)),
                "command_attack_reference_command_k_truth": _number_or_nan(command_diag.get("command_attack_reference_command_k")),
                "command_boundary_relevant_truth": bool(command_diag["command_boundary_relevant"]),
                "unsafe_this_step_truth": bool(unsafe_this_step),
                "unsafe_command_exposure_s": int(unsafe_exposure_s),
                "sensor_command_deviation_k_truth": float(sensor_command_deviation_k),
                "B_unsafe_so_far": int(b_unsafe),
                "B_continuity_so_far": int(b_continuity),
                "current_chain_interruption_s": current_chain_interruption_s,
                "pre_any_zone_outside_safe_envelope": pre_physical["any_zone_outside_safe_envelope"],
                "pre_zone_outside_safe_envelope_count": pre_physical["zone_outside_safe_envelope_count"],
                "pre_mean_abs_temperature_deviation_k": pre_physical["mean_abs_temperature_deviation_k"],
                "post_any_zone_outside_safe_envelope": post_physical["any_zone_outside_safe_envelope"],
                "post_zone_outside_safe_envelope_count": post_physical["zone_outside_safe_envelope_count"],
                "post_mean_abs_temperature_deviation_k": post_physical["mean_abs_temperature_deviation_k"],
                "post_max_abs_temperature_deviation_k": post_physical["max_abs_temperature_deviation_k"],
            }
            row.update(_zone_columns("true_pre_", true_values))
            row.update(_zone_columns("observed_", observed))
            row.update(_zone_columns("backup_observed_", backup_observed))
            row.update(_zone_columns("plant_post_", next_true_values))
            for name in metadata.get("diagnostic_measurements", []):
                row[f"plant_diag_{name}"] = _number_or_nan(response.get(name))
                row[f"plant_diag_post_{name}"] = _number_or_nan(response.get(name))
            steps.append(row)

        if bool(getattr(spec.challenge, "attack_delivered", True)):
            if spec.challenge.attack_type in {"sensor_bias", "sensor_dropout", "combined"} and not saw_sensor_effect:
                raise BOPTESTError("delivered sensor attack never altered controller-facing measurements")
            if spec.challenge.attack_type in {"command_injection", "combined"} and not saw_command_effect:
                raise BOPTESTError("delivered command injection never altered command transport")
        kpi = client.kpi(test_id)
    finally:
        client.stop(test_id)

    if chain_lost_since_s is not None:
        max_chain_interruption_s = max(max_chain_interruption_s, trial_duration_s - chain_lost_since_s)

    b = int(bool(b_unsafe or b_continuity))
    outcome = {
        "trial_id": spec.trial_id,
        "challenge_id": spec.challenge.challenge_id,
        "mechanism_config": spec.mechanism.config_id,
        "disturbance_class": getattr(spec.challenge, "disturbance_class", "B"),
        "Y": 1 - b,
        "Y_security": 1 - b,
        "B": b,
        "B_unsafe": b_unsafe,
        "B_continuity": b_continuity,
        "boundary_time_s": boundary_time_s,
        "maximum_chain_interruption_s": int(max_chain_interruption_s),
        "minimum_phi_exec": float(min(phi_exec_values)) if phi_exec_values else float("nan"),
        "mean_phi_exec": float(np.mean(phi_exec_values)) if phi_exec_values else float("nan"),
        "phi_exec_evaluation_window_s": int(phi_exec_evaluation_duration_s),
        "phi_exec_evaluation_window_target_s": int(phi_exec_evaluation_target_s),
        "phi_exec_evaluation_end_s": int(phi_exec_evaluation_end_s),
        "phi_exec_evaluation_auc_s": float(phi_exec_evaluation_area_s),
        "phi_exec_evaluation_mean": float(phi_exec_evaluation_area_s / phi_exec_evaluation_duration_s) if phi_exec_evaluation_duration_s else float("nan"),
        "phi_exec_evaluation_deficit_auc_s": float(phi_exec_evaluation_duration_s - phi_exec_evaluation_area_s) if phi_exec_evaluation_duration_s else float("nan"),
        "phi_exec_attack_mean": float(phi_exec_attack_area_s / phi_exec_attack_duration_s) if phi_exec_attack_duration_s else float("nan"),
        "phi_exec_post_attack_mean": float(phi_exec_post_area_s / phi_exec_post_duration_s) if phi_exec_post_duration_s else float("nan"),
        "time_below_phi_exec_min_s": int(time_below_phi_exec_min_s),
        "phi_exec_min": PHI_EXEC_MIN,
        "maximum_untrusted_command_exposure_s": int(max_unsafe_exposure_s),
        "T_safe_max_s": T_SAFE_MAX_S,
        "T_untrusted_command_max_s": T_UNTRUSTED_COMMAND_MAX_S,
        "attack_target_control_state_changed_truth": int(bool(saw_target_control_state_change)),
        "unprotected_final_control_state_changed_truth": int(bool(saw_unprotected_final_control_state_change)) if spec.mechanism.config_id == "0000" else None,
        "path_switch_history_json": json.dumps(
            [event for event in events if event.get("event_type") in {"redundancy_complete", "reconfiguration_complete", "primary_path_rejoin"}],
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    physical = evaluate_physical_from_steps(steps, kpi, spec.trial_id, step_s)
    outcome["Y_physical"] = physical["Y_physical"]
    outcome["Y_joint_diagnostic"] = float("nan")

    for event in events:
        event.update(
            {
                "trial_id": spec.trial_id,
                "challenge_id": spec.challenge.challenge_id,
                "mechanism_config": spec.mechanism.config_id,
            }
        )
    metadata.update(
        {
            "trial_id": spec.trial_id,
            "challenge": asdict(spec.challenge),
            "mechanism": asdict(spec.mechanism),
            "outcome": outcome,
            "step_rows": steps,
        }
    )
    return events, outcome, physical, metadata


def _build_design(
    output: Path,
    split: str,
    pilot_replicates: int,
    formal_replicates: int,
    design: str,
    design_seed: int,
    attack_delivery_probability: float = 1.0,
    write_files: bool = True,
) -> list[TrialSpec]:
    if design == "legacy":
        if write_files:
            write_manifests(output, split, pilot_replicates, formal_replicates)
        return build_trials(split, pilot_replicates, formal_replicates)

    if design == "preexperiment":
        challenges = build_preexperiment_challenges(seed=design_seed, scenario=V2_FIXED_SCENARIO, replicates=pilot_replicates)
        if write_files:
            write_custom_manifests(output, challenges, config_ids=PREEXPERIMENT_CONFIGS)
        return build_trials_from_challenges(challenges, config_ids=PREEXPERIMENT_CONFIGS)

    replicates = pilot_replicates if split == "development" else formal_replicates
    scenarios = (V2_FIXED_SCENARIO,)
    challenges = build_v2_challenges(
        split=split,
        seed=design_seed,
        per_attack_per_intensity=replicates,
        # A single benign challenge per physical scenario is sufficient because
        # the no-attack trajectory is deterministic and should not invoke any
        # stochastic mechanism execution.  It is still run under every mechanism
        # configuration to detect false triggers or configuration side effects.
        benign_count=1,
        scenarios=scenarios,
        attack_delivery_probability=attack_delivery_probability,
    )
    if write_files:
        write_custom_manifests(output, challenges)
    return build_trials_from_challenges(challenges)


def _run_one_trial_with_retries(
    base_url: str,
    spec: TrialSpec,
    step_s: int,
    trial_duration_s: int,
    retries: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any], int]:
    """Execute one complete trial using its own BOPTEST client/session.

    Parallelism is intentionally applied at the trial level.  A trial remains
    strictly sequential in time, while different trials use independent
    BOPTEST test instances.  Transient service/provisioning failures are
    retried with a short bounded backoff.
    """
    last_exc: Exception | None = None
    attempts = max(1, int(retries) + 1)
    for attempt in range(1, attempts + 1):
        client = BOPTESTClient(base_url)
        try:
            result = run_trial(client, spec, step_s=step_s, trial_duration_s=trial_duration_s)
            return (*result, attempt)
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            time.sleep(min(8.0, 1.5 * (2 ** (attempt - 1))))
    assert last_exc is not None
    raise last_exc


def _checkpoint_json_default(value: Any) -> Any:
    """Convert common NumPy/Pandas values into JSON-safe checkpoint values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


class TrialCheckpointStore:
    """Durably persist one complete trial at a time.

    The checkpoint is append-only JSONL.  The in-memory index stores only
    lightweight completion metadata and the byte offset of each payload, so a
    5000+ trial formal campaign does not retain time-series dictionaries in RAM.
    """

    SCHEMA_VERSION = 2

    def __init__(self, output: Path, planned: int) -> None:
        self.output = output
        self.planned = int(planned)
        self.path = output / "trial_checkpoints.txt"
        self.manifest_path = output / "checkpoint_manifest.txt"
        self.records = self._load_records()

    def _load_records(self) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        if not self.path.exists():
            return records
        try:
            with self.path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # A process can be interrupted during the final line.
                        continue
                    trial_id = str(payload.get("trial_id", ""))
                    if trial_id and payload.get("status") == "complete":
                        records[trial_id] = {
                            "trial_id": trial_id,
                            "attempts": int(payload.get("attempts", 1) or 1),
                            "status": "complete",
                            "offset": int(offset),
                        }
        except OSError:
            return records
        return records

    def _write_manifest(self, last_trial_id: str | None = None, status: str = "checkpointing") -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "status": status,
            "planned": self.planned,
            "completed": len(self.records),
            "remaining": max(0, self.planned - len(self.records)),
            "failures": [],
            "last_trial_id": last_trial_id,
            "checkpoint_file": self.path.name,
            "updated_at_epoch_s": time.time(),
        }
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.manifest_path)

    def append(
        self,
        trial_id: str,
        events: list[dict[str, Any]],
        outcome: dict[str, Any],
        physical: dict[str, Any],
        step_rows: list[dict[str, Any]],
        attempts: int,
    ) -> bool:
        trial_id = str(trial_id)
        if trial_id in self.records:
            return False
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "status": "complete",
            "trial_id": trial_id,
            "attempts": int(attempts),
            "events": events,
            "outcome": outcome,
            "physical": physical,
            "step_rows": step_rows,
        }
        encoded = (json.dumps(
            payload, ensure_ascii=False, allow_nan=True,
            default=_checkpoint_json_default, separators=(",", ":"),
        ) + "\n").encode("utf-8")
        offset = self.path.stat().st_size if self.path.exists() else 0
        with self.path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self.records[trial_id] = {
            "trial_id": trial_id,
            "attempts": int(attempts),
            "status": "complete",
            "offset": int(offset),
        }
        self._write_manifest(last_trial_id=trial_id)
        return True

    def iter_payloads(self, trial_ids: set[str] | None = None):
        """Yield complete payloads in deterministic trial-id order using byte offsets."""
        if not self.path.exists():
            return
        wanted = set(self.records) if trial_ids is None else (set(trial_ids) & set(self.records))
        with self.path.open("rb") as handle:
            for trial_id in sorted(wanted):
                record = self.records.get(trial_id) or {}
                offset = record.get("offset")
                if offset is None:
                    continue
                handle.seek(int(offset))
                line = handle.readline()
                try:
                    payload = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if str(payload.get("trial_id", "")) == trial_id and payload.get("status") == "complete":
                    yield payload

    def payloads(self, trial_ids: set[str] | None = None) -> list[dict[str, Any]]:
        return list(self.iter_payloads(trial_ids))

    def finalize(self, status: str) -> None:
        self._write_manifest(last_trial_id=None, status=status)

def merge_checkpoint_outputs(
    output: Path,
    checkpoint_store: TrialCheckpointStore,
    trial_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge durable per-trial checkpoints into the legacy aggregate tables."""
    all_events: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for payload in checkpoint_store.iter_payloads(trial_ids):
        all_events.extend(payload.get("events") or [])
        outcomes.append(payload.get("outcome") or {})
        physical_rows.append(payload.get("physical") or {})
        step_rows.extend(payload.get("step_rows") or [])

    # Parallel completion order is nondeterministic.  Sort before persistence
    # and hashing so repeated runs keep stable table ordering and hashes.
    all_events.sort(key=lambda row: (str(row.get("trial_id", "")), int(row.get("time_s", -1)), str(row.get("event_type", ""))))
    outcomes.sort(key=lambda row: str(row.get("trial_id", "")))
    physical_rows.sort(key=lambda row: str(row.get("trial_id", "")))
    step_rows.sort(key=lambda row: (str(row.get("trial_id", "")), int(row.get("time_s", -1))))

    if all_events:
        event_frame = pd.DataFrame(all_events)
        write_frame(event_frame, output / "raw_security_events")
        event_timing = _event_timing_summary(event_frame)
        if not event_timing.empty:
            write_frame(event_timing, output / "trial_event_timing")
        mechanism_frame = event_frame[
            event_frame["event_type"].isin(
                [
                    "mechanism_trigger",
                    "mechanism_execution_attempt",
                    "element_untrusted",
                    "isolation_start",
                    "isolation_complete",
                    "isolation_failed",
                    "redundancy_start",
                    "redundancy_complete",
                    "redundancy_failed",
                    "reconfiguration_start",
                    "reconfiguration_complete",
                    "reconfiguration_failed",
                    "fallback_activation",
                    "recovery_verification_pending",
                    "recovery_verification",
                    "recovery_probation_start",
                    "recovery_probation_pending",
                    "recovery_probation_failed",
                    "recovery_complete",
                    "recovery_failed",
                    "safe_rejoin",
                    "primary_path_rejoin",
                ]
            )
        ]
        write_frame(mechanism_frame, output / "mechanism_events")
    if outcomes:
        write_frame(pd.DataFrame(outcomes), output / "paired_outcomes")
    if physical_rows:
        write_frame(pd.DataFrame(physical_rows), output / "physical_criteria")
    if step_rows:
        step_frame = pd.DataFrame(step_rows)
        # Backward-compatible file name plus the explicit unified time-series
        # table used for paper figures and dynamic physical validation.
        write_frame(step_frame, output / "chain_state_timeseries")
        write_frame(step_frame, output / "physical_security_timeseries")
    return all_events, outcomes, physical_rows, step_rows


def materialize_checkpoint_outputs_streaming(
    output: Path,
    checkpoint_store: TrialCheckpointStore,
    trial_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    """Materialize formal outputs while streaming the wide step table to CSV.

    Full trial time-series never coexist in memory.  Payloads are read in
    deterministic trial-id order and each trial's rows are appended immediately.
    """
    all_events: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    step_csv = output / "physical_security_timeseries.csv"
    chain_csv = output / "chain_state_timeseries.csv"
    for stale in [step_csv, chain_csv, output / "physical_security_timeseries.parquet", output / "chain_state_timeseries.parquet"]:
        if stale.exists():
            stale.unlink()
    header = True
    step_count = 0
    for payload in checkpoint_store.iter_payloads(trial_ids):
        all_events.extend(payload.get("events") or [])
        outcomes.append(payload.get("outcome") or {})
        physical_rows.append(payload.get("physical") or {})
        rows = payload.get("step_rows") or []
        if rows:
            chunk = pd.DataFrame(rows)
            if {"trial_id", "time_s"}.issubset(chunk.columns):
                chunk = chunk.sort_values(["trial_id", "time_s"], kind="stable")
            chunk.to_csv(step_csv, mode="w" if header else "a", header=header, index=False)
            header = False
            step_count += len(chunk)

    all_events.sort(key=lambda row: (str(row.get("trial_id", "")), int(row.get("time_s", -1)), str(row.get("event_type", ""))))
    outcomes.sort(key=lambda row: str(row.get("trial_id", "")))
    physical_rows.sort(key=lambda row: str(row.get("trial_id", "")))
    if all_events:
        event_frame = pd.DataFrame(all_events)
        write_frame(event_frame, output / "raw_security_events")
        event_timing = _event_timing_summary(event_frame)
        if not event_timing.empty:
            write_frame(event_timing, output / "trial_event_timing")
        mechanism_frame = event_frame[event_frame["event_type"].isin([
            "mechanism_trigger", "mechanism_execution_attempt", "element_untrusted",
            "isolation_start", "isolation_complete", "isolation_failed",
            "redundancy_start", "redundancy_complete", "redundancy_failed",
            "redundancy_cancelled",
            "reconfiguration_start", "reconfiguration_complete", "reconfiguration_failed",
            "reconfiguration_cancelled", "fallback_activation",
            "recovery_verification_pending", "recovery_verification",
            "recovery_probation_start", "recovery_probation_pending",
            "recovery_probation_failed", "recovery_complete", "recovery_failed",
            "safe_rejoin", "primary_path_rejoin",
        ])]
        write_frame(mechanism_frame, output / "mechanism_events")
    if outcomes:
        write_frame(pd.DataFrame(outcomes), output / "paired_outcomes")
    if physical_rows:
        write_frame(pd.DataFrame(physical_rows), output / "physical_criteria")
    if step_count:
        shutil.copyfile(step_csv, chain_csv)
    return all_events, outcomes, physical_rows, step_count


def run_protocol(
    base_url: str,
    output: Path,
    split: str,
    limit: int | None,
    pilot_replicates: int,
    formal_replicates: int,
    step_s: int,
    trial_duration_s: int = V2_TRIAL_DURATION_S,
    challenge_id: str | None = None,
    shards: int = 1,
    shard_index: int = 0,
    design: str = "v2",
    design_seed: int | None = None,
    attack_delivery_probability: float = 1.0,
    workers: int = 1,
    retries: int = 1,
    resume: bool = False,
) -> int:
    if output.exists():
        if not output.is_dir():
            raise RuntimeError("output must be a directory")
        if any(output.iterdir()) and not resume:
            raise RuntimeError("output must be empty")
    else:
        if resume:
            raise RuntimeError("cannot resume because output directory does not exist")
        output.mkdir(parents=True, exist_ok=False)
    if step_s > 300:
        raise RuntimeError("step_s must be <= 300 s for the current mechanism timing configuration")
    if workers < 1:
        raise RuntimeError("workers must be at least 1")
    if retries < 0:
        raise RuntimeError("retries must be >= 0")
    if design_seed is None:
        design_seed = stable_seed(f"boptest_esec|{design}|{split}|2026-09")

    trials = _build_design(
        output,
        split,
        pilot_replicates,
        formal_replicates,
        design,
        design_seed,
        attack_delivery_probability,
        write_files=not resume,
    )
    topology_path = output / "topology_manifest.csv"
    if not topology_path.exists():
        pd.DataFrame(topology_rows()).to_csv(topology_path, index=False)
    if design in {"v2", "preexperiment"}:
        required_post = V2_POST_ATTACK_OBSERVATION_S if design == "v2" else PREEXPERIMENT_POST_ATTACK_S
        too_short = [
            trial.challenge.challenge_id
            for trial in trials
            if trial.challenge.attack_type != "benign"
            and trial.challenge.attack_start_s + trial.challenge.attack_duration_s + required_post > trial_duration_s
        ]
        if too_short:
            raise RuntimeError(
                "trial_duration_s is too short for the V2 event window; each attack requires at least "
                f"{required_post} s post-attack observation. First invalid challenge: {too_short[0]}"
            )
    if challenge_id is not None:
        trials = [trial for trial in trials if trial.challenge.challenge_id == challenge_id]
        if not trials:
            raise RuntimeError(f"unknown challenge_id for split {split}: {challenge_id}")
    if shards < 1:
        raise RuntimeError("shards must be at least 1")
    if not 0 <= shard_index < shards:
        raise RuntimeError("shard_index must satisfy 0 <= shard_index < shards")
    if shards > 1:
        challenge_order = sorted({trial.challenge.challenge_id for trial in trials})
        selected = {cid for idx, cid in enumerate(challenge_order) if idx % shards == shard_index}
        trials = [trial for trial in trials if trial.challenge.challenge_id in selected]
    if limit is not None:
        trials = trials[:limit]

    planned_trial_ids = {trial.trial_id for trial in trials}

    # Freeze a complete resume contract before any trial is executed.  Trial IDs
    # intentionally omit random seeds and attack timing, so checking IDs alone is
    # insufficient: a run resumed with another design seed could otherwise reuse
    # incompatible checkpoints.
    design_contract = {
        "version": VERSION,
        "design": design,
        "split": split,
        "design_seed": int(design_seed),
        "pilot_replicates": int(pilot_replicates),
        "formal_replicates": int(formal_replicates),
        "step_s": int(step_s),
        "trial_duration_s": int(trial_duration_s),
        "attack_delivery_probability": float(1.0 if design == "preexperiment" else attack_delivery_probability),
        "shards": int(shards),
        "shard_index": int(shard_index),
        "challenge_id_filter": challenge_id,
        "limit": limit,
        "trials": [
            {
                "trial_id": trial.trial_id,
                "challenge": asdict(trial.challenge),
                "mechanism": asdict(trial.mechanism),
            }
            for trial in trials
        ],
    }
    design_fingerprint = canonical_hash(design_contract)
    fingerprint_path = output / "design_fingerprint.txt"
    if resume:
        if not fingerprint_path.exists():
            raise RuntimeError("cannot resume safely: design_fingerprint.txt is missing")
        previous = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if str(previous.get("fingerprint")) != design_fingerprint:
            raise RuntimeError(
                "resume design mismatch: code version, seed, trial parameters or filtered trial set differ from the original run"
            )
    else:
        fingerprint_path.write_text(
            json.dumps({"fingerprint": design_fingerprint, "contract": design_contract}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    checkpoint_store = TrialCheckpointStore(output, planned=len(trials))

    all_events: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    attempts_by_trial: dict[str, int] = {}

    # Restore only checkpoints belonging to this exact filtered design.  This
    # supports --resume together with challenge filtering, limits and shards.
    restored_trial_ids: set[str] = set()
    if resume:
        for trial_id, record in checkpoint_store.records.items():
            if trial_id not in planned_trial_ids:
                continue
            restored_trial_ids.add(trial_id)
            attempts_by_trial[trial_id] = int(record.get("attempts", 1) or 1)
    pending_trials = [trial for trial in trials if trial.trial_id not in restored_trial_ids]

    max_workers = min(int(workers), max(1, len(pending_trials)))
    completed_count = len(restored_trial_ids)
    if max_workers == 1:
        for spec in pending_trials:
            try:
                events, outcome, physical, metadata, attempts = _run_one_trial_with_retries(
                    base_url, spec, step_s, trial_duration_s, retries
                )
                checkpoint_store.append(
                    spec.trial_id,
                    events,
                    outcome,
                    physical,
                    metadata["step_rows"],
                    attempts,
                )
                attempts_by_trial[spec.trial_id] = attempts
                completed_count += 1
                print(f"completed {spec.trial_id} ({completed_count}/{len(trials)}, attempt {attempts})", flush=True)
            except Exception as exc:
                attempts_by_trial[spec.trial_id] = int(retries) + 1
                failures.append({"trial_id": spec.trial_id, "error": repr(exc), "attempts": int(retries) + 1})
                print(f"failed {spec.trial_id}: {exc}", file=sys.stderr, flush=True)
    else:
        print(f"starting {len(pending_trials)} trials with {max_workers} parallel workers ({completed_count}/{len(trials)} already checkpointed)", flush=True)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="boptest-trial") as pool:
            future_map = {
                pool.submit(
                    _run_one_trial_with_retries,
                    base_url,
                    spec,
                    step_s,
                    trial_duration_s,
                    retries,
                ): spec
                for spec in pending_trials
            }
            for future in as_completed(future_map):
                spec = future_map[future]
                try:
                    events, outcome, physical, metadata, attempts = future.result()
                    checkpoint_store.append(
                        spec.trial_id,
                        events,
                        outcome,
                        physical,
                        metadata["step_rows"],
                        attempts,
                    )
                    attempts_by_trial[spec.trial_id] = attempts
                    completed_count += 1
                    print(
                        f"completed {spec.trial_id} ({completed_count}/{len(trials)}, attempt {attempts})",
                        flush=True,
                    )
                except Exception as exc:
                    attempts_by_trial[spec.trial_id] = int(retries) + 1
                    failures.append({"trial_id": spec.trial_id, "error": repr(exc), "attempts": int(retries) + 1})
                    print(f"failed {spec.trial_id}: {exc}", file=sys.stderr, flush=True)

    # Re-read the durable per-trial records and merge them into the original
    # aggregate output contract.  This makes the final tables independent of
    # the in-memory state of the current process.
    all_events, outcomes, physical_rows, step_row_count = materialize_checkpoint_outputs_streaming(
        output,
        checkpoint_store,
        trial_ids=planned_trial_ids,
    )
    failures.sort(key=lambda row: str(row.get("trial_id", "")))

    status = "complete_raw_boptest_esec_execution" if not failures and len(outcomes) == len(trials) else "invalid_incomplete"

    manifest = {
        "status": status,
        "version": VERSION,
        "design": design,
        "design_seed": design_seed,
        "design_fingerprint": design_fingerprint,
        "planned": len(trials),
        "completed": len(outcomes),
        "failures": failures,
        "shards": shards,
        "shard_index": shard_index,
        "parallel_workers_requested": int(workers),
        "parallel_workers_effective": int(max_workers),
        "trial_retry_count": int(retries),
        "trial_attempts": attempts_by_trial,
        "parallelism_scope": "independent complete trials; each trial keeps sequential 300 s plant-control dynamics and its own BOPTEST testid",
        "step_s": step_s,
        "trial_duration_s": trial_duration_s,
        "attack_delivery_probability": 1.0 if design == "preexperiment" else float(attack_delivery_probability),
        "attacked_replicates_per_cell": (pilot_replicates if split == "development" else formal_replicates),
        "benign_challenges_per_physical_scenario": 1 if design in {"v2", "preexperiment"} else None,
        "fixed_physical_scenario": V2_FIXED_SCENARIO,
        "event_window": {
            "pre_attack_start_range_s": [3600, 5400],
            "attack_duration_range_s": [3600, 5400],
            "minimum_post_attack_observation_s": V2_POST_ATTACK_OBSERVATION_S,
        } if design == "v2" else ({
            "pre_attack_s": 2 * 3600,
            "attack_duration_s": 90 * 60,
            "post_attack_observation_s": PREEXPERIMENT_POST_ATTACK_S,
            "trial_duration_s": PREEXPERIMENT_TRIAL_DURATION_S,
        } if design == "preexperiment" else None),
        "online_runtime": "ObservableRuntime",
        "information_boundary": "defense sees attacked measurements, command readbacks, local state and timers only",
        "primary_outcome": "continuous E_sec from paired Q_exec, where Q_exec is the time-average of Phi_exec from disturbance onset through recovery observation; binary Y_security is auxiliary",
        "mechanism_success_probabilities": dict(MECHANISM_EXECUTION_PROBABILITIES),
        "mechanism_randomization": "challenge-seeded mechanism-specific common random numbers across matched configurations",
        "protection_action_transients": {
            "reconfiguration_handover_s": 300,
            "recovery_resynchronization_s": 300,
            "phi_exec_mapping": "min(b,a,s) with (b,a,s)=(1.0,0.0,0.70) during an active handover/resynchronization interval",
            "path_credit_rule": "a path selected in finish receives execution credit only after the next prepare call actually uses it",
        },
        "physical_outcome_role": "independent post-run external criterion; Y_joint_diagnostic is reported but is not the primary measurand",
        "timeseries_output": "physical_security_timeseries contains synchronized plant truth, controller-facing measurements, attack annotations, mechanism state, commands and physical envelope status",
        "checkpoint": {
            "file": checkpoint_store.path.name,
            "wide_timeseries_rows_streamed": int(step_row_count),
            "completed": len(checkpoint_store.records),
            "resume_supported": True,
        },
        "T_safe_max_s": T_SAFE_MAX_S,
        "T_untrusted_command_max_s": T_UNTRUSTED_COMMAND_MAX_S,
        "phi_exec_min": PHI_EXEC_MIN,
        "event_hash": canonical_hash(all_events) if all_events else None,
        "outcome_hash": canonical_hash(outcomes) if outcomes else None,
        "python": sys.version,
        "platform": platform.platform(),
    }
    (output / "run_manifest.txt").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint_store.finalize(status)
    return 0 if status.startswith("complete") else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("development", "formal"), default="development")
    parser.add_argument("--design", choices=("v2", "preexperiment", "legacy"), default="v2")
    parser.add_argument("--design-seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--challenge-id", default=None)
    parser.add_argument("--pilot-replicates", type=int, default=5, help="N per development (scenario, attack, intensity) cell")
    parser.add_argument("--formal-replicates", type=int, default=20, help="N per formal (scenario, attack, intensity) cell; use 30 for a larger final run")
    parser.add_argument("--step-s", type=int, default=300)
    parser.add_argument("--trial-duration-s", type=int, default=None, help="v2 default 21600 s (6 h); preexperiment default 18000 s (5 h)")
    parser.add_argument("--attack-delivery-probability", type=float, default=1.0, help="Optional stochastic attack delivery for dedicated robustness studies; main/pre-experiment default is 1.0")
    parser.add_argument("--workers", type=int, default=1, help="Number of complete trials to execute concurrently. The dedicated preexperiment entry defaults to 12.")
    parser.add_argument("--retries", type=int, default=1, help="Retries per failed trial for transient BOPTEST service/provisioning errors")
    parser.add_argument("--resume", action="store_true", help="Resume from trial_checkpoints.txt in an existing output directory")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    trial_duration_s = args.trial_duration_s
    if trial_duration_s is None:
        trial_duration_s = PREEXPERIMENT_TRIAL_DURATION_S if args.design == "preexperiment" else V2_TRIAL_DURATION_S
    return run_protocol(
        args.base_url,
        args.output,
        args.split,
        args.limit,
        args.pilot_replicates,
        args.formal_replicates,
        args.step_s,
        trial_duration_s,
        args.challenge_id,
        args.shards,
        args.shard_index,
        args.design,
        args.design_seed,
        args.attack_delivery_probability,
        args.workers,
        args.retries,
        args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
