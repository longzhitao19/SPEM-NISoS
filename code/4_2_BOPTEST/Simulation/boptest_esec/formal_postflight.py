from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .analysis import analyze


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--expected-trials', type=int, default=5136)
    args = parser.parse_args()

    manifest_path = args.input / 'run_manifest.txt'
    if not manifest_path.exists():
        raise SystemExit('run_manifest.txt is missing')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    errors: list[str] = []
    if manifest.get('status') != 'complete_raw_boptest_esec_execution':
        errors.append(f"raw run status={manifest.get('status')}")
    if int(manifest.get('planned', -1)) != int(args.expected_trials):
        errors.append(f"planned={manifest.get('planned')} expected={args.expected_trials}")
    if int(manifest.get('completed', -1)) != int(args.expected_trials):
        errors.append(f"completed={manifest.get('completed')} expected={args.expected_trials}")
    if manifest.get('failures'):
        errors.append(f"raw run has {len(manifest.get('failures') or [])} failures")
    if errors:
        print(json.dumps({'status': 'FAIL', 'errors': errors}, ensure_ascii=False, indent=2))
        return 2

    analyze(args.input, args.output)
    diag_path = args.output / 'BOPTEST_EXPERIMENT_VALIDITY_DIAGNOSTICS.csv'
    if not diag_path.exists():
        print(json.dumps({'status': 'FAIL', 'errors': ['validity diagnostics file is missing']}, ensure_ascii=False, indent=2))
        return 2
    diag = pd.read_csv(diag_path)
    fails = diag[diag['status'].astype(str) == 'FAIL'] if 'status' in diag.columns else pd.DataFrame()
    warns = diag[diag['status'].astype(str) == 'WARN'] if 'status' in diag.columns else pd.DataFrame()
    result = {
        'status': 'PASS' if fails.empty else 'FAIL',
        'diagnostic_rows': int(len(diag)),
        'failures': fails.to_dict('records'),
        'warnings': warns.to_dict('records'),
        'analysis_output': str(args.output),
    }
    (args.output / 'FORMAL_POSTFLIGHT_SUMMARY.txt').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if fails.empty else 2


if __name__ == '__main__':
    raise SystemExit(main())
