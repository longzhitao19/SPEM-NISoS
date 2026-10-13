from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from . import VERSION
from .protocol import (
    MECHANISM_CONFIGS,
    MECHANISM_EXECUTION_PROBABILITIES,
    V2_ATTACK_DURATION_RANGE_S,
    V2_ATTACK_START_RANGE_S,
    V2_EVALUATION_WINDOW_S,
    V2_FIXED_SCENARIO,
    V2_POST_ATTACK_OBSERVATION_S,
    V2_TRIAL_DURATION_S,
    build_trials_from_challenges,
    build_v2_challenges,
)


def _check_service(base_url: str, timeout_s: float = 5.0) -> dict:
    url = base_url.rstrip('/') + '/version'
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            body = response.read(4096).decode('utf-8', errors='replace')
            return {'reachable': True, 'url': url, 'status': int(response.status), 'body_prefix': body[:500]}
    except Exception as exc:
        return {'reachable': False, 'url': url, 'error': repr(exc)}


def audit_formal_design(seed: int, replicates: int, step_s: int, trial_duration_s: int) -> dict:
    challenges = build_v2_challenges(
        split='formal',
        seed=int(seed),
        per_attack_per_intensity=int(replicates),
        benign_count=1,
        scenarios=(V2_FIXED_SCENARIO,),
        attack_delivery_probability=1.0,
    )
    trials = build_trials_from_challenges(challenges)
    attacked = [c for c in challenges if c.attack_type != 'benign']
    benign = [c for c in challenges if c.attack_type == 'benign']
    cells = Counter((c.attack_type, c.intensity_label) for c in attacked)
    errors: list[str] = []

    if step_s != 300:
        errors.append(f'step_s must be 300 for the frozen formal design, got {step_s}')
    if trial_duration_s != V2_TRIAL_DURATION_S:
        errors.append(f'trial_duration_s must be {V2_TRIAL_DURATION_S}, got {trial_duration_s}')
    if len(benign) != 1:
        errors.append(f'expected 1 benign challenge, got {len(benign)}')
    if len(cells) != 16 or set(cells.values()) != {int(replicates)}:
        errors.append(f'attack×intensity balance invalid: {dict(sorted(cells.items()))}')
    if len(attacked) != 16 * int(replicates):
        errors.append(f'expected {16 * int(replicates)} attacked challenges, got {len(attacked)}')
    if len(trials) != (16 * int(replicates) + 1) * len(MECHANISM_CONFIGS):
        errors.append(f'unexpected trial count: {len(trials)}')
    if len({c.challenge_id for c in challenges}) != len(challenges):
        errors.append('challenge_id values are not unique')
    if len({c.seed for c in challenges}) != len(challenges):
        errors.append('challenge seeds are not unique')
    if any(not c.attack_delivered for c in attacked):
        errors.append('formal main-effect design contains an undelivered attack')
    if any(bool(c.recovery_sensitive) for c in challenges):
        errors.append('unused recovery_sensitive nuisance flag must be frozen false')
    if any(c.scenario != V2_FIXED_SCENARIO for c in challenges):
        errors.append('formal design contains a non-frozen physical scenario')
    if any(c.attack_start_s < V2_ATTACK_START_RANGE_S[0] or c.attack_start_s > V2_ATTACK_START_RANGE_S[1] for c in attacked):
        errors.append('attack start is outside the frozen range')
    if any(c.attack_duration_s < V2_ATTACK_DURATION_RANGE_S[0] or c.attack_duration_s > V2_ATTACK_DURATION_RANGE_S[1] for c in attacked):
        errors.append('attack duration is outside the frozen range')
    if any(c.attack_start_s + c.attack_duration_s + V2_POST_ATTACK_OBSERVATION_S > trial_duration_s for c in attacked):
        errors.append('at least one challenge lacks the required 3 h post-attack observation')
    if V2_EVALUATION_WINDOW_S != 16200:
        errors.append(f'common Q_exec window expected 16200 s, got {V2_EVALUATION_WINDOW_S}')

    return {
        'status': 'PASS' if not errors else 'FAIL',
        'version': VERSION,
        'design_seed': int(seed),
        'formal_replicates_per_attack_intensity_cell': int(replicates),
        'physical_scenario': V2_FIXED_SCENARIO,
        'challenge_count': len(challenges),
        'attacked_challenge_count': len(attacked),
        'benign_challenge_count': len(benign),
        'mechanism_configuration_count': len(MECHANISM_CONFIGS),
        'trial_count': len(trials),
        'attack_intensity_cell_counts': {f'{a}|{i}': n for (a, i), n in sorted(cells.items())},
        'mechanism_execution_probabilities': dict(MECHANISM_EXECUTION_PROBABILITIES),
        'step_s': int(step_s),
        'trial_duration_s': int(trial_duration_s),
        'q_exec_common_evaluation_window_s': int(V2_EVALUATION_WINDOW_S),
        'minimum_post_attack_observation_s': int(V2_POST_ATTACK_OBSERVATION_S),
        'sample_first_attacked_challenge': asdict(attacked[0]) if attacked else None,
        'errors': errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1')
    parser.add_argument('--design-seed', type=int, default=20260915)
    parser.add_argument('--formal-replicates', type=int, default=20)
    parser.add_argument('--step-s', type=int, default=300)
    parser.add_argument('--trial-duration-s', type=int, default=V2_TRIAL_DURATION_S)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--skip-service', action='store_true')
    args = parser.parse_args()

    result = audit_formal_design(args.design_seed, args.formal_replicates, args.step_s, args.trial_duration_s)
    result['service'] = {'reachable': None, 'skipped': True} if args.skip_service else _check_service(args.base_url)
    if not args.skip_service and not result['service'].get('reachable', False):
        result['status'] = 'FAIL'
        result['errors'].append('BOPTEST service is not reachable at the requested base URL')

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    return 0 if result['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
