from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

from .client import ZONE_TEMPERATURES


INTENSITIES = {"Low": 0.25, "Medium": 0.50, "High": 0.75}
MECHANISM_BITS = ("m_red", "m_cfg", "m_iso", "m_rec")
MECHANISM_EXECUTION_PROBABILITIES = {"M_red": 0.90, "M_cfg": 0.80, "M_iso": 0.90, "M_rec": 0.65}
MECHANISM_CONFIGS = tuple("".join(bits) for bits in product("01", repeat=4))
ATTACKS_DEVELOPMENT = ("benign", "sensor_bias", "sensor_dropout", "command_injection", "combined")
ATTACKS_FORMAL = ("benign", "sensor_bias", "sensor_dropout", "command_injection", "combined")
# Frozen formal attack-intensity levels.  The numeric intensity is the
# pre-registered severity coordinate used by every attack family.  Labels are
# kept only for readability; values are exact and are not sampled from ranges.
# User-specified order: [0.90, 0.65, 0.40, 0.15].
V2_INTENSITY_VALUES = {
    "weak": 0.15,
    "moderate": 0.40,
    "strong": 0.65,
    "extreme": 0.90,
}
# Compatibility alias for older callers/tests that inspect this name.  Each
# range is degenerate by design so formal intensity remains exactly frozen.
V2_INTENSITY_RANGES = {label: (value, value) for label, value in V2_INTENSITY_VALUES.items()}
# V2 uses a compact event-centred simulation window.  Time values below are
# elapsed seconds from the beginning of the selected BOPTEST time period.
# The latest attack ends at 3 h and leaves at least 3 h for plant recovery in
# the default 6 h trial.
V2_ATTACK_START_RANGE_S = (3600, 5400)       # 60--90 min pre-attack baseline
V2_ATTACK_DURATION_RANGE_S = (3600, 5400)    # 60--90 min attack
V2_POST_ATTACK_OBSERVATION_S = 3 * 3600      # >= 3 h recovery observation
V2_TRIAL_DURATION_S = 6 * 3600
# Common formal evaluation horizon measured from each attack onset.  Because
# attack onset is randomized between 60 and 90 min, 4.5 h is the longest
# horizon available to every formal challenge inside the fixed 6 h trial.
# Freezing this horizon prevents early-onset challenges from receiving extra
# post-attack recovery weight in Q_exec.
V2_EVALUATION_WINDOW_S = V2_TRIAL_DURATION_S - V2_ATTACK_START_RANGE_S[1]
V2_NORMAL_OPERATING_RANGE_K = 5.5

# multizone_office_simple_air official time-period definitions.  Each named
# scenario is a two-week interval centred on the target day with one-week
# baseline warm-up.  V2 initializes directly at the target day and a
# load-relevant clock time, retaining the one-week warm-up while evaluating only
# the 6 h event window.
V2_SCENARIO_CENTER_DAY = {
    "typical_heat_day": 18,  # official period Day 11--25
    "peak_heat_day": 26,     # official period Day 19--33
    "peak_cool_day": 199,    # official period Day 192--206
}
V2_SCENARIO_WINDOW_START_CLOCK_S = {
    "typical_heat_day": 5 * 3600,
    "peak_heat_day": 5 * 3600,
    "peak_cool_day": 11 * 3600,
}
V2_BOPTEST_WARMUP_S = 7 * 24 * 3600
V2_FIXED_SCENARIO = "peak_heat_day"
PREEXPERIMENT_INTENSITY = V2_INTENSITY_VALUES["weak"]
# The quick diagnostic window is aligned with the observed peak-heat control
# transition: two hours of baseline, 90 min of attack and 90 min of recovery.
PREEXPERIMENT_ATTACK_START_S = 2 * 3600
PREEXPERIMENT_ATTACK_DURATION_S = 90 * 60
PREEXPERIMENT_POST_ATTACK_S = 90 * 60
PREEXPERIMENT_TRIAL_DURATION_S = 5 * 3600
PREEXPERIMENT_CONFIGS = ("0000", "1000", "0100", "0010", "0001", "1111")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def disturbance_class_for_attack(attack_type: str) -> str:
    """Map all attacked engineering scenarios to one unified disturbed class.

    BOPTEST is used here to demonstrate the measurement chain rather than to
    reproduce the theoretical F/C taxonomy.  Benign controls are class B and
    every injected attack is class D (disturbed).
    """
    if attack_type == "benign":
        return "B"
    if attack_type in {"sensor_bias", "sensor_dropout", "command_injection", "combined"}:
        return "D"
    raise ValueError(f"unsupported attack_type for disturbed-state mapping: {attack_type}")


@dataclass(frozen=True)
class MechanismConfig:
    config_id: str
    m_red: bool
    m_cfg: bool
    m_iso: bool
    m_rec: bool

    @classmethod
    def from_id(cls, config_id: str) -> "MechanismConfig":
        if config_id not in MECHANISM_CONFIGS:
            raise ValueError(f"unknown mechanism configuration {config_id}")
        return cls(config_id, *(bit == "1" for bit in config_id))

    def enabled_names(self) -> list[str]:
        return [name for name, enabled in zip(MECHANISM_BITS, (self.m_red, self.m_cfg, self.m_iso, self.m_rec)) if enabled]


@dataclass(frozen=True)
class ChallengeSpec:
    challenge_id: str
    split: str
    scenario: str
    attack_type: str
    intensity_label: str
    intensity: float
    attack_target: str
    attack_start_s: int
    attack_duration_s: int
    seed: int
    disturbance_class: str = "B"
    boptest_start_time_s: int = 0
    boptest_warmup_period_s: int = 0
    timing_label: str = "fixed_0900"
    duration_label: str = "fixed_3h"
    challenge_family: str = "legacy"
    expected_difficulty: str = "legacy"
    relative_bias: float = 0.0
    normal_operating_range_k: float = V2_NORMAL_OPERATING_RANGE_K
    bias_magnitude_k: float = 0.0
    bias_sign: int = 1
    affected_sensor_count: int = 0
    affected_sensor_identity_json: str = "[]"
    dropout_pattern: str = "none"
    dropout_period_s: int = 0
    dropout_duty_cycle: float = 0.0
    command_offset_k: float = 0.0
    command_direction: int = 1
    command_target: str = "none"
    affected_actuator_count: int = 0
    affected_actuator_identity_json: str = "[]"
    attack_ramp_s: int = 0
    communication_disruption: bool = False
    recovery_sensitive: bool = False
    # Attack delivery is experimental ground truth.  Main/pre-experiment runs
    # default to 1.0 so every scheduled attack is actually injected.  A lower
    # probability can be enabled only for a dedicated stochastic-delivery study.
    attack_delivery_probability: float = 1.0
    attack_delivery_draw: float = 0.0
    attack_delivered: bool = True
    # Contract-preserving command attacks move the command toward the opposite
    # valid control extreme instead of exceeding the BOPTEST input contract.
    command_attack_mode: str = "bounded_opposite_extreme"
    command_strength: float = 0.0
    command_fixed_target_k: float = 0.0
    # Pre-registered perturbation requirement.  When enabled, the attack
    # generator deterministically adjusts only its own injected magnitude/sign
    # so that the targeted controller state changes at least once.  The defense
    # runtime never receives this flag or any attack truth.
    force_control_state_change: bool = False
    sensor_state_change_margin_k: float = 0.25
    max_sensor_bias_k: float = 8.0

    def external_key(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrialSpec:
    challenge: ChallengeSpec
    mechanism: MechanismConfig

    @property
    def trial_id(self) -> str:
        return f"{self.challenge.challenge_id}_{self.mechanism.config_id}"


def build_challenges(split: str, pilot_replicates: int = 2, formal_replicates: int = 4) -> list[ChallengeSpec]:
    if split == "development":
        scenarios = ("typical_heat_day",)
        attacks = ATTACKS_DEVELOPMENT
        repeats = pilot_replicates
    elif split == "formal":
        scenarios = ("peak_heat_day", "peak_cool_day")
        attacks = ATTACKS_FORMAL
        repeats = formal_replicates
    else:
        raise ValueError(split)
    output: list[ChallengeSpec] = []
    for scenario, attack_type, intensity_label, replicate in product(scenarios, attacks, INTENSITIES, range(1, repeats + 1)):
        seed_material = f"{split}|{scenario}|{attack_type}|{intensity_label}|{replicate}"
        seed = stable_seed(seed_material)
        target = {
            "sensor_bias": "S_zone_primary",
            "sensor_dropout": "S_zone_primary",
            "command_injection": "A_primary",
            "combined": "S_zone_primary+A_primary",
            "benign": "none",
        }[attack_type]
        challenge_id = f"{split}_{scenario}_{attack_type}_{intensity_label}_r{replicate:03d}"
        intensity = INTENSITIES[intensity_label]
        affected_count = 0
        affected_json = "[]"
        bias_magnitude_k = 0.0
        command_offset_k = 0.0
        command_direction = 1
        affected_actuator_count = 0
        if attack_type in {"sensor_bias", "sensor_dropout", "combined"}:
            affected_count = max(1, min(len(ZONE_TEMPERATURES), int(round(1 + 4 * intensity))))
            affected_json = json.dumps(_v2_sensor_identities(seed, affected_count))
        if attack_type in {"sensor_bias", "combined"}:
            bias_magnitude_k = round(V2_NORMAL_OPERATING_RANGE_K * intensity, 3)
            command_direction = -1 if seed % 2 else 1
        if attack_type in {"command_injection", "combined"}:
            command_direction = -1 if seed % 2 else 1
            command_offset_k = round(command_direction * (0.4 + 8.8 * intensity), 3)
            affected_actuator_count = 1
        output.append(
            ChallengeSpec(
                challenge_id=challenge_id,
                split=split,
                scenario=scenario,
                attack_type=attack_type,
                intensity_label=intensity_label,
                intensity=intensity,
                attack_target=target,
                attack_start_s=9 * 3600,
                attack_duration_s=3 * 3600,
                seed=seed,
                disturbance_class=disturbance_class_for_attack(attack_type),
                bias_magnitude_k=bias_magnitude_k,
                bias_sign=1,
                affected_sensor_count=affected_count,
                affected_sensor_identity_json=affected_json,
                dropout_pattern="single_long" if attack_type == "sensor_dropout" else "none",
                dropout_duty_cycle=1.0 if attack_type == "sensor_dropout" else 0.0,
                command_offset_k=command_offset_k,
                command_direction=command_direction,
                command_target="hvac_oveAhu_TSupSet_u" if affected_actuator_count else "none",
                affected_actuator_count=affected_actuator_count,
                affected_actuator_identity_json=json.dumps(["primary"] if affected_actuator_count else []),
            )
        )
    return output


def _round_to_step(value: int, step_s: int = 300) -> int:
    return int(round(value / step_s) * step_s)


def _v2_sensor_identities(seed: int, count: int) -> list[str]:
    import random

    from .client import ZONE_TEMPERATURES

    rng = random.Random(seed)
    count = max(0, min(count, len(ZONE_TEMPERATURES)))
    return sorted(rng.sample(list(ZONE_TEMPERATURES), count))


def _v2_intensity_value(seed: int, intensity_label: str) -> float:
    # ``seed`` is intentionally unused here: intensity is a frozen experimental
    # factor, not a random nuisance variable.  Keeping the argument preserves
    # backward compatibility with existing callers.
    _ = seed
    return float(V2_INTENSITY_VALUES[intensity_label])


def _v2_boptest_start_time_s(scenario: str) -> int:
    if scenario not in V2_SCENARIO_CENTER_DAY or scenario not in V2_SCENARIO_WINDOW_START_CLOCK_S:
        raise ValueError(f"unsupported V2 BOPTEST scenario {scenario}")
    return int(V2_SCENARIO_CENTER_DAY[scenario] * 24 * 3600 + V2_SCENARIO_WINDOW_START_CLOCK_S[scenario])


def make_v2_challenge(
    split: str,
    scenario: str,
    attack_type: str,
    intensity_label: str,
    replicate: int,
    seed: int,
    normal_operating_range_k: float = V2_NORMAL_OPERATING_RANGE_K,
    attack_delivery_probability: float = 1.0,
) -> ChallengeSpec:
    """Create one pre-registered V2 challenge inside a compact 6 h event window.

    Attack topology is fixed by attack type.  Intensity changes magnitude/scope
    within that topology, rather than silently changing to another attack type.
    """
    import random

    rng = random.Random(seed)
    attack_delivery_probability = min(1.0, max(0.0, float(attack_delivery_probability)))
    delivery_draw = random.Random(seed ^ 0x5A17B9D3).random()
    attack_delivered = bool(delivery_draw < attack_delivery_probability)
    start_s = _round_to_step(rng.randint(*V2_ATTACK_START_RANGE_S))
    boptest_start_time_s = _v2_boptest_start_time_s(scenario)

    if attack_type == "benign":
        challenge_id = f"v2_{split}_{scenario}_{attack_type}_{replicate:03d}"
        return ChallengeSpec(
            challenge_id=challenge_id,
            split=split,
            scenario=scenario,
            attack_type=attack_type,
            intensity_label="benign",
            intensity=0.0,
            attack_target="none",
            attack_start_s=start_s,
            attack_duration_s=0,
            seed=seed,
            disturbance_class="B",
            boptest_start_time_s=boptest_start_time_s,
            boptest_warmup_period_s=V2_BOPTEST_WARMUP_S,
            timing_label="relative_60_90min_reference",
            duration_label="none",
            challenge_family="v2_event_window",
            expected_difficulty="benign",
            normal_operating_range_k=normal_operating_range_k,
            affected_sensor_identity_json="[]",
            affected_actuator_identity_json="[]",
            attack_delivery_probability=1.0,
            attack_delivery_draw=0.0,
            attack_delivered=False,
            command_strength=0.0,
        )

    if attack_type not in {"sensor_bias", "sensor_dropout", "command_injection", "combined"}:
        raise ValueError(f"unsupported V2 attack_type {attack_type}")
    if intensity_label not in V2_INTENSITY_RANGES:
        raise ValueError(f"unsupported V2 intensity_label {intensity_label}")

    intensity = _v2_intensity_value(seed, intensity_label)
    # Attack duration is deliberately independent of intensity.  Otherwise a
    # weak level would mix lower amplitude with shorter exposure, making it
    # impossible to attribute a null control response to intensity itself.
    duration_s = _round_to_step(rng.randint(*V2_ATTACK_DURATION_RANGE_S))
    sign = rng.choice([-1, 1])

    # Sensor scope increases with intensity, but the attacked topology remains
    # the primary sensing surface.  Combined is sensor-bias + command injection;
    # it intentionally does not include sensor dropout.
    # Effective-attack design: sensing attacks must be capable of changing the
    # downstream controller state.  Bias/combined therefore affect a majority
    # of the five-zone sensing surface, while dropout always breaks the 3-of-5
    # controller quorum.  Intensity still controls magnitude, timing and
    # dropout pattern/severity.
    bias_count_by_intensity = {"weak": 3, "moderate": 3, "strong": 4, "extreme": 5}
    dropout_count_by_intensity = {"weak": 3, "moderate": 3, "strong": 4, "extreme": 5}
    if attack_type in {"sensor_bias", "combined"}:
        affected_sensor_count = bias_count_by_intensity[intensity_label]
    elif attack_type == "sensor_dropout":
        affected_sensor_count = dropout_count_by_intensity[intensity_label]
    else:
        affected_sensor_count = 0
    affected_sensors = _v2_sensor_identities(seed, affected_sensor_count)

    relative_bias = intensity if attack_type in {"sensor_bias", "combined"} else 0.0
    bias_magnitude_k = round(normal_operating_range_k * relative_bias, 3)

    dropout_pattern = "none"
    dropout_period_s = 0
    dropout_duty_cycle = 0.0
    if attack_type == "sensor_dropout":
        # Dropout severity is graded by duty cycle while the attacked sensing
        # surface remains the same topology.  Values are quantized to the
        # 300 s simulation step, so the nominal lambda=0.15 weak level uses a
        # 0.20 duty cycle as the nearest practical low-severity probe.
        dropout_pattern = "intermittent"
        dropout_period_s = 3000
        dropout_duty_cycle = {
            "weak": 0.20,
            "moderate": 0.40,
            "strong": 0.70,
            "extreme": 0.90,
        }[intensity_label]

    command_direction = rng.choice([-1, 1])
    command_offset_k = 0.0
    command_strength = 0.0
    affected_actuator_count = 0
    affected_actuators: list[str] = []
    attack_ramp_s = 0
    if attack_type in {"command_injection", "combined"}:
        scale = 1.0
        command_strength = round(min(0.95, max(0.10, scale * intensity)), 4)
        # Kept only as a human-readable nominal magnitude.  Runtime injection
        # uses command_strength and remains inside the command contract.
        command_offset_k = round(command_direction * scale * (0.4 + 8.8 * intensity), 3)
        # Keep the attacked node fixed at A_primary for all intensities.  The
        # guarded endpoint stays a defensive alternative, which preserves the
        # interpretation of reconfiguration.
        affected_actuator_count = 1
        affected_actuators = ["primary"]
        # Keep command-attack temporal shape fixed across intensity levels so
        # lambda changes command severity rather than ramp duration.
        attack_ramp_s = 600

    # Network-wide communication loss is a different disturbance topology and
    # is therefore not silently introduced by increasing dropout intensity.
    communication_disruption = False
    # Deprecated V2 metadata flag retained in ChallengeSpec for file compatibility.
    # The current observable runtime does not consume it, so formal challenges
    # keep it frozen rather than injecting a meaningless random nuisance factor.
    recovery_sensitive = False

    expected_difficulty = {
        "weak": "easy",
        "moderate": "moderate",
        "strong": "hard",
        "extreme": "extreme",
    }[intensity_label]
    target = {
        "sensor_bias": "S_zone_primary",
        "sensor_dropout": "S_zone_primary",
        "command_injection": "e_DA_primary->A_primary",
        "combined": "S_zone_primary+e_DA_primary->A_primary",
    }[attack_type]
    challenge_id = f"v2_{split}_{scenario}_{attack_type}_{intensity_label}_{replicate:03d}"
    return ChallengeSpec(
        challenge_id=challenge_id,
        split=split,
        scenario=scenario,
        attack_type=attack_type,
        intensity_label=intensity_label,
        intensity=intensity,
        attack_target=target,
        attack_start_s=start_s,
        attack_duration_s=duration_s,
        seed=seed,
        disturbance_class=disturbance_class_for_attack(attack_type),
        boptest_start_time_s=boptest_start_time_s,
        boptest_warmup_period_s=V2_BOPTEST_WARMUP_S,
        timing_label="relative_60_90min",
        duration_label="event_60_90min",
        challenge_family="v2_event_window",
        expected_difficulty=expected_difficulty,
        relative_bias=relative_bias,
        normal_operating_range_k=normal_operating_range_k,
        bias_magnitude_k=bias_magnitude_k,
        bias_sign=sign,
        affected_sensor_count=affected_sensor_count,
        affected_sensor_identity_json=json.dumps(affected_sensors),
        dropout_pattern=dropout_pattern,
        dropout_period_s=dropout_period_s,
        dropout_duty_cycle=dropout_duty_cycle,
        command_offset_k=command_offset_k,
        command_direction=command_direction,
        command_target="hvac_oveAhu_TSupSet_u" if affected_actuator_count else "none",
        affected_actuator_count=affected_actuator_count,
        affected_actuator_identity_json=json.dumps(affected_actuators),
        attack_ramp_s=attack_ramp_s,
        communication_disruption=communication_disruption,
        recovery_sensitive=recovery_sensitive,
        attack_delivery_probability=attack_delivery_probability,
        attack_delivery_draw=delivery_draw,
        attack_delivered=attack_delivered,
        command_attack_mode="bounded_opposite_extreme",
        command_strength=command_strength,
        # Formal attacks use frozen parameters and are never adjusted online to
        # force a controller-state change. The pre-experiment keeps that visual
        # diagnostic option separately.
        force_control_state_change=False,
        sensor_state_change_margin_k=0.25,
        max_sensor_bias_k=8.0,
    )


def build_v2_challenges(
    split: str,
    seed: int,
    per_attack_per_intensity: int = 5,
    benign_count: int | None = None,
    scenarios: tuple[str, ...] = (V2_FIXED_SCENARIO,),
    normal_operating_range_k: float = V2_NORMAL_OPERATING_RANGE_K,
    attack_delivery_probability: float = 1.0,
) -> list[ChallengeSpec]:
    """Build a balanced V2 design.

    ``per_attack_per_intensity`` is N for every
    (physical scenario, attack type, intensity) cell.  Benign references have no
    intensity dimension and default to one challenge per physical scenario.
    That single benign challenge is still executed under every mechanism
    configuration so false triggers/configuration side effects remain observable.
    """
    if per_attack_per_intensity < 1:
        raise ValueError("per_attack_per_intensity must be >= 1")
    if benign_count is None:
        benign_count = 1
    if benign_count < 1:
        raise ValueError("benign_count must be >= 1")

    challenges: list[ChallengeSpec] = []
    attack_types = ("sensor_bias", "sensor_dropout", "command_injection", "combined")
    for scenario in scenarios:
        for local_rep in range(1, benign_count + 1):
            challenge_seed = stable_seed(f"v2|{split}|{scenario}|benign|{local_rep}|{seed}")
            challenges.append(
                make_v2_challenge(
                    split, scenario, "benign", "benign", local_rep, challenge_seed, normal_operating_range_k, 1.0
                )
            )
        for attack_type, intensity_label in product(attack_types, V2_INTENSITY_VALUES):
            for local_rep in range(1, per_attack_per_intensity + 1):
                challenge_seed = stable_seed(
                    f"v2|{split}|{scenario}|{attack_type}|{intensity_label}|{local_rep}|{seed}"
                )
                challenges.append(
                    make_v2_challenge(
                        split,
                        scenario,
                        attack_type,
                        intensity_label,
                        local_rep,
                        challenge_seed,
                        normal_operating_range_k,
                        attack_delivery_probability,
                    )
                )
    return challenges



def make_preexperiment_challenge(
    attack_type: str,
    seed: int,
    scenario: str = V2_FIXED_SCENARIO,
    replicate: int = 1,
    intensity: float = PREEXPERIMENT_INTENSITY,
    normal_operating_range_k: float = V2_NORMAL_OPERATING_RANGE_K,
) -> ChallengeSpec:
    """Create a lowest-intensity pre-experiment challenge.

    The pre-experiment is now a boundary diagnostic for the weakest formal
    attack level (lambda=0.15).  It deliberately disables any mechanism that
    forces a controller-state change.  Thus a failure of
    ``attack_target_control_state_changed_truth`` is a real warning that the
    formal weak attack is too small or poorly timed.  Duration and command-ramp
    semantics are aligned with the formal V2 design.
    """
    if attack_type not in {"benign", "sensor_bias", "sensor_dropout", "command_injection", "combined"}:
        raise ValueError(f"unsupported preexperiment attack_type {attack_type}")

    if attack_type == "benign":
        base = make_v2_challenge(
            split="preexperiment",
            scenario=scenario,
            attack_type="benign",
            intensity_label="benign",
            replicate=int(replicate),
            seed=seed,
            normal_operating_range_k=normal_operating_range_k,
            attack_delivery_probability=1.0,
        )
        return ChallengeSpec(**{
            **asdict(base),
            "challenge_id": f"pre_{scenario}_benign_r{int(replicate):02d}",
            "attack_start_s": PREEXPERIMENT_ATTACK_START_S,
            "timing_label": "fixed_120min_preexperiment",
            "challenge_family": "preexperiment_weak_boundary",
        })

    if not math.isclose(float(intensity), float(V2_INTENSITY_VALUES["weak"]), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"preexperiment is frozen to the lowest formal intensity {V2_INTENSITY_VALUES['weak']:.2f}; "
            f"got {float(intensity):.4f}"
        )

    # Reuse the formal weak-attack semantics, then freeze pre-experiment timing
    # to the 5 h diagnostic window.  This keeps magnitude/scope/ramp definitions
    # identical to the formal weak level while making plots directly comparable.
    base = make_v2_challenge(
        split="preexperiment",
        scenario=scenario,
        attack_type=attack_type,
        intensity_label="weak",
        replicate=int(replicate),
        seed=seed,
        normal_operating_range_k=normal_operating_range_k,
        attack_delivery_probability=1.0,
    )
    return ChallengeSpec(**{
        **asdict(base),
        "challenge_id": f"pre_{scenario}_{attack_type}_weak_r{int(replicate):02d}",
        "intensity_label": "weak_fixed_0.15",
        "intensity": float(V2_INTENSITY_VALUES["weak"]),
        "attack_start_s": PREEXPERIMENT_ATTACK_START_S,
        "attack_duration_s": PREEXPERIMENT_ATTACK_DURATION_S,
        "timing_label": "fixed_120min_preexperiment",
        "duration_label": "fixed_90min_preexperiment",
        "challenge_family": "preexperiment_weak_boundary",
        "expected_difficulty": "weak_boundary",
        "attack_delivered": True,
        "attack_delivery_probability": 1.0,
        "attack_delivery_draw": 0.0,
        "force_control_state_change": False,
    })


def build_preexperiment_challenges(
    seed: int,
    scenario: str = V2_FIXED_SCENARIO,
    replicates: int = 1,
    benign_count: int = 1,
) -> list[ChallengeSpec]:
    """Build the weak-boundary pre-experiment.

    ``replicates`` applies only to attacked challenges.  Benign is intentionally
    run once per physical scenario because, in the absence of an anomaly, no
    stochastic mechanism execution should be invoked.  The single benign
    challenge is still evaluated under every selected mechanism configuration
    so false triggers and configuration side effects remain observable.
    """
    if replicates < 1:
        raise ValueError("preexperiment attack replicates must be >= 1")
    if benign_count != 1:
        raise ValueError("preexperiment benign_count is frozen to 1")

    challenges: list[ChallengeSpec] = []
    benign_seed = stable_seed(f"preexperiment|{scenario}|benign|rep=1|{seed}")
    challenges.append(
        make_preexperiment_challenge(
            attack_type="benign",
            seed=benign_seed,
            scenario=scenario,
            replicate=1,
        )
    )
    for replicate in range(1, int(replicates) + 1):
        for attack_type in ("sensor_bias", "sensor_dropout", "command_injection", "combined"):
            challenges.append(
                make_preexperiment_challenge(
                    attack_type=attack_type,
                    seed=stable_seed(f"preexperiment|{scenario}|{attack_type}|rep={replicate}|{seed}"),
                    scenario=scenario,
                    replicate=replicate,
                )
            )
    return challenges

def build_trials_from_challenges(challenges: list[ChallengeSpec], config_ids: tuple[str, ...] = MECHANISM_CONFIGS) -> list[TrialSpec]:
    return [TrialSpec(challenge, MechanismConfig.from_id(config_id)) for challenge in challenges for config_id in config_ids]


def challenge_from_mapping(row: dict[str, Any]) -> ChallengeSpec:
    fields = ChallengeSpec.__dataclass_fields__
    kwargs: dict[str, Any] = {}
    for name, field_info in fields.items():
        if name not in row:
            continue
        value = row[name]
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        typ = field_info.type
        if typ in {int, "int"}:
            value = int(value)
        elif typ in {float, "float"}:
            value = float(value)
        elif typ in {bool, "bool"}:
            if isinstance(value, str):
                value = value.strip().lower() in {"1", "true", "yes"}
            else:
                value = bool(value)
        elif typ in {str, "str"}:
            value = str(value)
        kwargs[name] = value
    return ChallengeSpec(**kwargs)


def write_custom_manifests(root: Path, challenges: list[ChallengeSpec], config_ids: tuple[str, ...] = MECHANISM_CONFIGS) -> tuple[Path, Path, Path]:
    import pandas as pd

    root.mkdir(parents=True, exist_ok=True)
    trials = build_trials_from_challenges(challenges, config_ids)
    challenge_path = root / "challenge_manifest.csv"
    experiment_path = root / "experiment_manifest.csv"
    mechanism_path = root / "mechanism_configurations.csv"
    pd.DataFrame([asdict(c) for c in challenges]).to_csv(challenge_path, index=False)
    pd.DataFrame(
        [
            {
                "trial_id": t.trial_id,
                "challenge_id": t.challenge.challenge_id,
                "split": t.challenge.split,
                "scenario": t.challenge.scenario,
                "attack_type": t.challenge.attack_type,
                "intensity_label": t.challenge.intensity_label,
                "intensity": t.challenge.intensity,
                "mechanism_config": t.mechanism.config_id,
                **asdict(t.mechanism),
            }
            for t in trials
        ]
    ).to_csv(experiment_path, index=False)
    pd.DataFrame([
        {**asdict(MechanismConfig.from_id(cfg)),
         "p_red_exec": MECHANISM_EXECUTION_PROBABILITIES["M_red"],
         "p_cfg_exec": MECHANISM_EXECUTION_PROBABILITIES["M_cfg"],
         "p_iso_exec": MECHANISM_EXECUTION_PROBABILITIES["M_iso"],
         "p_rec_exec": MECHANISM_EXECUTION_PROBABILITIES["M_rec"]}
        for cfg in config_ids
    ]).to_csv(mechanism_path, index=False)
    return challenge_path, experiment_path, mechanism_path


def build_trials(split: str, pilot_replicates: int = 2, formal_replicates: int = 4) -> list[TrialSpec]:
    challenges = build_challenges(split, pilot_replicates, formal_replicates)
    return [TrialSpec(ch, MechanismConfig.from_id(cfg)) for ch in challenges for cfg in MECHANISM_CONFIGS]


def write_manifests(root: Path, split: str, pilot_replicates: int, formal_replicates: int) -> tuple[Path, Path, Path]:
    import pandas as pd

    root.mkdir(parents=True, exist_ok=True)
    challenges = build_challenges(split, pilot_replicates, formal_replicates)
    trials = build_trials(split, pilot_replicates, formal_replicates)
    challenge_path = root / "challenge_manifest.csv"
    experiment_path = root / "experiment_manifest.csv"
    mechanism_path = root / "mechanism_configurations.csv"
    pd.DataFrame([asdict(c) for c in challenges]).to_csv(challenge_path, index=False)
    pd.DataFrame(
        [
            {
                "trial_id": t.trial_id,
                "challenge_id": t.challenge.challenge_id,
                "split": t.challenge.split,
                "scenario": t.challenge.scenario,
                "attack_type": t.challenge.attack_type,
                "intensity_label": t.challenge.intensity_label,
                "mechanism_config": t.mechanism.config_id,
                **asdict(t.mechanism),
            }
            for t in trials
        ]
    ).to_csv(experiment_path, index=False)
    pd.DataFrame([asdict(MechanismConfig.from_id(cfg)) for cfg in MECHANISM_CONFIGS]).to_csv(mechanism_path, index=False)
    return challenge_path, experiment_path, mechanism_path
