"""Recompute the 270-minute worst zone-temperature deviation from raw trials.

Every attack challenge and all 16 matched configurations are retained. The
comparison uses the same configuration's benign trajectory at the same
physical time, sampled at the 54 evaluation-interval endpoints.
"""

from pathlib import Path
import pprint

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data"
RAW = ROOT / 'raw_data'
ZONES = [f"plant_post_hvac_reaZon{s}_TZon_y" for s in ("Cor", "Eas", "Nor", "Sou", "Wes")]
ATKS = ("sensor_bias", "sensor_dropout", "command_injection", "combined")
LEVELS = ("weak", "moderate", "strong", "extreme")
CFGS = [f"{i:04b}" for i in range(16)]
SEED = 20260922
B = 4000


def interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.integers(0, len(values), size=(B, len(values)))
    means = values[draws].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [.025, .975]))


def main() -> None:
    cols = [
        "trial_id", "challenge_id", "mechanism_config", "attack_type_truth",
        "intensity_label_truth", "time_s", "next_time_s", "attack_start_s_truth",
        "phi_exec_evaluation_end_s_truth", "final_command_k",
        "plant_diag_post_hvac_reaAhu_TSup_y",
    ] + ZONES
    ts = pd.read_csv(RAW / "physical_security_timeseries.csv", usecols=cols,
                     dtype={"mechanism_config": str})
    ts["mechanism_config"] = ts.mechanism_config.str.zfill(4)
    assert len(ts) == 5136 * 72
    assert (ts.next_time_s - ts.time_s).eq(300).all()

    benign = ts.loc[ts.attack_type_truth.eq("benign")].copy()
    assert benign.mechanism_config.nunique() == 16 and len(benign) == 16 * 72
    baseline = benign.loc[benign.mechanism_config.eq("0000")].sort_values("time_s")
    baseline_cols = ["time_s", "next_time_s", "final_command_k",
                     "plant_diag_post_hvac_reaAhu_TSup_y"] + ZONES
    baseline[baseline_cols].to_csv(DATA / "benign_baseline.csv", index=False)
    reference = benign.set_index(["mechanism_config", "next_time_s"])[ZONES]
    assert not reference.index.has_duplicates

    design = pd.read_csv(DATA / "challenge_design.csv").set_index("challenge_id")
    ids = [cid for attack in ATKS for level in LEVELS
           for cid in sorted(design[(design.attack_type == attack)
                                    & (design.intensity_label == level)].index)]
    assert len(ids) == 320
    ev = ts.loc[
        ts.challenge_id.isin(ids)
        & ts.time_s.ge(ts.attack_start_s_truth)
        & ts.time_s.lt(ts.phi_exec_evaluation_end_s_truth)
    ].copy()
    ev["k"] = ((ev.time_s - ev.attack_start_s_truth) // 300).astype(int)
    ix = pd.MultiIndex.from_product([ids, CFGS, range(54)],
                                    names=["challenge_id", "mechanism_config", "k"])
    ev = ev.set_index(["challenge_id", "mechanism_config", "k"]).loc[ix].reset_index()
    assert len(ev) == 320 * 16 * 54
    ref = reference.loc[pd.MultiIndex.from_frame(ev[["mechanism_config", "next_time_s"]])].to_numpy()
    absolute = np.abs(ev[ZONES].to_numpy() - ref).reshape(320, 16, 54, 5)
    assert np.isfinite(absolute).all()
    j = absolute.mean(axis=(2, 3))
    dmax = absolute.max(axis=(2, 3))
    old = pd.read_csv(DATA / "paired_results.csv", dtype={"config": str})
    old["config"] = old.config.str.zfill(4)
    old = old.set_index(["challenge_id", "config"]).loc[
        pd.MultiIndex.from_product([ids, CFGS])
    ]
    np.testing.assert_allclose(j.reshape(-1), old.J.to_numpy(), rtol=0, atol=1e-9)

    improvement = dmax[:, [0]] - dmax
    records = []
    for i, cid in enumerate(ids):
        for k, cfg in enumerate(CFGS):
            records.append({
                "challenge_id": cid, "attack": design.loc[cid, "attack_type"],
                "intensity": design.loc[cid, "intensity_label"], "config": cfg,
                "D_max_K": dmax[i, k], "D_max_U_K": dmax[i, 0],
                "D_max_reduction_K": improvement[i, k],
            })
    paired = pd.DataFrame.from_records(records)
    paired.to_csv(DATA / "paired_max_deviation.csv", index=False)

    rng = np.random.default_rng(SEED)
    condition = []
    for attack in ATKS:
        for level in LEVELS:
            x = paired[(paired.attack == attack) & (paired.intensity == level)]
            for cfg in ("0000", "1111"):
                y = x[x.config == cfg]
                values = y.D_max_K.to_numpy()
                lo, hi = interval(values, rng)
                reduction = y.D_max_reduction_K.to_numpy()
                rlo, rhi = interval(reduction, rng)
                condition.append({"attack": attack, "intensity": level,
                                  "config": cfg, "n": len(y),
                                  "D_max_mean_K": values.mean(),
                                  "D_max_low_K": lo, "D_max_high_K": hi,
                                  "D_max_reduction_mean_K": reduction.mean(),
                                  "D_max_reduction_low_K": rlo,
                                  "D_max_reduction_high_K": rhi})
    pd.DataFrame(condition).to_csv(DATA / "condition_max_deviation.csv", index=False)

    weights = np.zeros((B, 320))
    for cell in range(16):
        draw = rng.integers(0, 20, size=(B, 20))
        for b in range(B):
            weights[b, cell*20:cell*20+20] = np.bincount(draw[b], minlength=20) / 320
    overall = {}
    for label, x in (("U", dmax[:, 0]), ("RCIH", dmax[:, 15]),
                     ("paired_reduction", improvement[:, 15])):
        q = np.quantile(weights @ x, [.025, .975])
        overall[label] = {"mean_K": float(x.mean()),
                          "ci95_K": [float(q[0]), float(q[1])]}
    metadata = {
        "source": "physical_security_timeseries.csv",
        "reference": "same-configuration benign trajectory at matching next_time_s",
        "window": "270 min after attack start; 54 five-minute interval endpoints",
        "max_axes": "5 zones x 54 endpoints per challenge/configuration",
        "n_challenges": 320, "n_configurations": 16, "n_paired_rows": len(paired),
        "bootstrap": {"replicates": B, "unit": "full 16-configuration challenge block",
                      "strata": "4 attack types x 4 intensities"},
        "overall": overall,
    }
    (DATA / "max_deviation_summary.txt").write_text(
        pprint.pformat(metadata, sort_dicts=False), encoding="utf-8")
    print(pprint.pformat(metadata["overall"], sort_dicts=False))


if __name__ == "__main__":
    main()
