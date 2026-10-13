from __future__ import annotations

import argparse
import json
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import pag_risk_reduction
from .io_utils import read_frame, table_exists
from .protocol import MECHANISM_BITS, MECHANISM_CONFIGS, canonical_hash

PHYSICAL_SAFE_MIN_K = 293.15
PHYSICAL_SAFE_MAX_K = 297.15
DYNAMIC_RECOVERY_HOLD_S = 600
POST_ATTACK_WINDOW_S = 3600
POST_ATTACK_RECOVERY_WINDOW_S = 3 * 3600


def _ci(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return (np.nan, np.nan)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))




def _attacked_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the attacked reference set used by security/physical comparisons."""
    if "attack_type" in frame.columns:
        return frame[frame["attack_type"].astype(str) != "benign"].copy()
    if "disturbance_class" in frame.columns:
        return frame[frame["disturbance_class"].astype(str) == "D"].copy()
    return frame.copy()


def _standardized_summary(values: pd.DataFrame, value_col: str, strata_cols: list[str]) -> float:
    """Equal-weight cell mean under the frozen attack x intensity reference set."""
    if not strata_cols:
        return float(values[value_col].mean())
    cell = values.groupby(strata_cols, dropna=False)[value_col].mean()
    return float(cell.mean())

def paired_stats(outcomes: pd.DataFrame, n_boot: int = 2000, seed: int = 20260827) -> pd.DataFrame:
    outcomes = _attacked_only(outcomes)
    pivot = outcomes.pivot(index="challenge_id", columns="mechanism_config", values="Y")
    if "0000" not in pivot.columns:
        raise ValueError("paired outcomes require 0000 baseline")
    rng = np.random.default_rng(seed)
    challenges = pivot.index.to_numpy()
    rows = []
    for cfg in MECHANISM_CONFIGS:
        if cfg not in pivot.columns:
            continue
        y0 = pivot["0000"].to_numpy(int)
        ys = pivot[cfg].to_numpy(int)
        diff = ys - y0
        boot = []
        for _ in range(n_boot):
            sample = rng.choice(len(challenges), size=len(challenges), replace=True)
            boot.append(float(np.mean(diff[sample])))
        r = float(np.mean((y0 == 0) & (ys == 1)))
        h = float(np.mean((y0 == 1) & (ys == 0)))
        rows.append(
            {
                "mechanism_config": cfg,
                "P_safe": float(np.mean(ys)),
                "P_breach": float(1.0 - np.mean(ys)),
                "E_sec": float(np.mean(diff)),
                "E_sec_ci_low": _ci(np.asarray(boot))[0],
                "E_sec_ci_high": _ci(np.asarray(boot))[1],
                "R": r,
                "H": h,
                "R_minus_H": r - h,
                "identity_error": float(np.mean(diff) - (r - h)),
                "N_challenges": int(len(challenges)),
            }
        )
    return pd.DataFrame(rows)



def unified_security_effectiveness(
    outcomes: pd.DataFrame, n_boot: int = 4000, seed: int = 20260915
) -> pd.DataFrame:
    """Primary continuous effectiveness under the frozen equal-weight reference set.

    Each of the 16 attack-type x intensity cells receives equal reference weight.
    Bootstrap resampling is performed within each cell, so uncertainty reflects
    replicate variation without randomly changing the pre-registered cell weights.
    """
    attacked = _attacked_only(outcomes)
    if attacked.empty:
        return pd.DataFrame()
    if "phi_exec_evaluation_mean" not in attacked.columns:
        raise ValueError("continuous effectiveness requires phi_exec_evaluation_mean in paired_outcomes")
    attacked = attacked.copy()
    attacked["mechanism_config"] = attacked["mechanism_config"].astype(str).str.zfill(4)
    pivot = attacked.pivot(index="challenge_id", columns="mechanism_config", values="phi_exec_evaluation_mean")
    if "0000" not in pivot.columns:
        raise ValueError("continuous effectiveness requires paired 0000 baseline")

    meta_cols = [c for c in ["attack_type", "intensity_label"] if c in attacked.columns]
    meta = attacked.drop_duplicates("challenge_id").set_index("challenge_id")
    y_pivot = attacked.pivot(index="challenge_id", columns="mechanism_config", values="Y") if "Y" in attacked.columns else None
    rng = np.random.default_rng(seed)
    rows = []
    for cfg in MECHANISM_CONFIGS:
        if cfg not in pivot.columns:
            continue
        cols = ["0000"] if cfg == "0000" else ["0000", cfg]
        pair = pivot[cols].dropna()
        if pair.empty:
            continue
        work = pd.DataFrame(index=pair.index)
        work["q_u"] = pair["0000"].astype(float)
        work["q_p"] = work["q_u"] if cfg == "0000" else pair[cfg].astype(float)
        work["delta"] = work["q_p"] - work["q_u"]
        work["gain"] = np.maximum(work["delta"], 0.0)
        work["harm"] = np.maximum(-work["delta"], 0.0)
        work["improved"] = (work["delta"] > 1e-12).astype(float)
        work["harmed"] = (work["delta"] < -1e-12).astype(float)
        for col in meta_cols:
            work[col] = meta.loc[work.index, col].astype(str)
        strata = meta_cols if len(meta_cols) == 2 else []

        e_sec = _standardized_summary(work, "delta", strata)
        gain = _standardized_summary(work, "gain", strata)
        harm = _standardized_summary(work, "harm", strata)
        q_u = _standardized_summary(work, "q_u", strata)
        q_p = _standardized_summary(work, "q_p", strata)
        frac_improved = _standardized_summary(work, "improved", strata)
        frac_harmed = _standardized_summary(work, "harmed", strata)

        boot = []
        if strata:
            groups = [g for _, g in work.groupby(strata, sort=True, dropna=False)]
            for _ in range(n_boot):
                cell_means = []
                for g in groups:
                    idx = rng.integers(0, len(g), size=len(g))
                    cell_means.append(float(g["delta"].to_numpy(float)[idx].mean()))
                boot.append(float(np.mean(cell_means)))
        else:
            arr = work["delta"].to_numpy(float)
            for _ in range(n_boot):
                boot.append(float(arr[rng.integers(0, len(arr), size=len(arr))].mean()))
        ci_low, ci_high = _ci(np.asarray(boot))
        row = {
            "mechanism_config": cfg,
            "Q_exec_U": q_u,
            "Q_exec_P": q_p,
            "G_exec": gain,
            "H_exec": harm,
            "E_sec": e_sec,
            "E_sec_ci_low": ci_low,
            "E_sec_ci_high": ci_high,
            "identity_error": e_sec - (gain - harm),
            "N_attacked_challenges": int(len(work)),
            "N_reference_cells": int(work.groupby(strata).ngroups) if strata else 1,
            "reference_weighting": "equal_attack_type_x_intensity_cells" if strata else "equal_challenges",
            "bootstrap_scheme": "stratified_within_reference_cell" if strata else "challenge_bootstrap",
            "delta_exec_min": float(work["delta"].min()),
            "delta_exec_max": float(work["delta"].max()),
            "fraction_improved": frac_improved,
            "fraction_harmed": frac_harmed,
        }
        if y_pivot is not None and cfg in y_pivot.columns:
            ypair = y_pivot.loc[work.index, ["0000"] if cfg == "0000" else ["0000", cfg]].dropna()
            if not ypair.empty:
                yw = pd.DataFrame(index=ypair.index)
                yw["yu"] = ypair["0000"].astype(float)
                yw["yp"] = yw["yu"] if cfg == "0000" else ypair[cfg].astype(float)
                for col in strata:
                    yw[col] = meta.loc[yw.index, col].astype(str)
                row["P_safe_U_aux"] = _standardized_summary(yw, "yu", strata)
                row["P_safe_P_aux"] = _standardized_summary(yw, "yp", strata)
        rows.append(row)
    return pd.DataFrame(rows)


def time_local_security_effectiveness(ts: pd.DataFrame) -> pd.DataFrame:
    """Decompose paired instantaneous Phi_exec differences over time.

    The existing primary H_exec is a *challenge-level net* quantity:
    ``max(-(Q_P-Q_U), 0)``.  It intentionally becomes zero when later benefits
    outweigh an earlier protection-action cost.  This diagnostic preserves the
    same Q_exec evaluation window but separates positive and negative
    instantaneous paired differences before integration:

        G_exec_time = (1/T) integral max(Phi_P-Phi_U, 0) dt
        H_exec_time = (1/T) integral max(Phi_U-Phi_P, 0) dt

    Therefore ``E_sec_time = G_exec_time-H_exec_time`` equals the paired
    Q_exec difference while transient protection-induced harm remains visible.
    """
    if ts.empty:
        return pd.DataFrame()
    required = {
        "challenge_id",
        "mechanism_config",
        "attack_type_truth",
        "intensity_label_truth",
        "attack_start_s_truth",
        "time_s",
        "next_time_s",
        "phi_exec",
    }
    missing = sorted(required.difference(ts.columns))
    if missing:
        raise ValueError(f"time-local effectiveness requires time-series columns: {missing}")

    work = ts.copy()
    work["mechanism_config"] = work["mechanism_config"].astype(str).str.zfill(4)
    work = work[work["attack_type_truth"].astype(str) != "benign"].copy()
    if work.empty:
        return pd.DataFrame()

    if "phi_exec_evaluation_end_s_truth" not in work.columns:
        # Legacy-compatible fallback: use all samples from attack onset onward.
        work["phi_exec_evaluation_end_s_truth"] = np.inf

    meta = (
        work[["challenge_id", "attack_type_truth", "intensity_label_truth"]]
        .drop_duplicates("challenge_id")
        .set_index("challenge_id")
    )
    baseline = work[work["mechanism_config"] == "0000"][
        [
            "challenge_id", "time_s", "next_time_s", "attack_start_s_truth",
            "phi_exec_evaluation_end_s_truth", "phi_exec",
        ]
    ].rename(columns={"phi_exec": "phi_u", "next_time_s": "next_time_u"})
    if baseline.empty:
        raise ValueError("time-local effectiveness requires paired 0000 baseline")

    rows: list[dict[str, object]] = []
    for cfg in MECHANISM_CONFIGS:
        protected = work[work["mechanism_config"] == cfg][
            ["challenge_id", "time_s", "next_time_s", "phi_exec"]
        ].rename(columns={"phi_exec": "phi_p", "next_time_s": "next_time_p"})
        if protected.empty:
            continue
        pair = baseline.merge(protected, on=["challenge_id", "time_s"], how="inner", validate="one_to_one")
        if pair.empty:
            continue
        pair = pair[
            (pair["time_s"].astype(float) >= pair["attack_start_s_truth"].astype(float))
            & (pair["time_s"].astype(float) < pair["phi_exec_evaluation_end_s_truth"].astype(float))
        ].copy()
        if pair.empty:
            continue
        pair["step_end"] = np.minimum(
            np.minimum(pair["next_time_u"].astype(float), pair["next_time_p"].astype(float)),
            pair["phi_exec_evaluation_end_s_truth"].astype(float),
        )
        pair["dt"] = np.maximum(0.0, pair["step_end"] - pair["time_s"].astype(float))
        pair = pair[pair["dt"] > 0].copy()
        pair["dphi"] = pair["phi_p"].astype(float) - pair["phi_u"].astype(float)
        pair["gain_area"] = np.maximum(pair["dphi"], 0.0) * pair["dt"]
        pair["harm_area"] = np.maximum(-pair["dphi"], 0.0) * pair["dt"]
        pair["delta_area"] = pair["dphi"] * pair["dt"]

        challenge = pair.groupby("challenge_id", as_index=True).agg(
            evaluation_s=("dt", "sum"),
            gain_area=("gain_area", "sum"),
            harm_area=("harm_area", "sum"),
            delta_area=("delta_area", "sum"),
        )
        challenge = challenge[challenge["evaluation_s"] > 0].copy()
        if challenge.empty:
            continue
        challenge["G_exec_time"] = challenge["gain_area"] / challenge["evaluation_s"]
        challenge["H_exec_time"] = challenge["harm_area"] / challenge["evaluation_s"]
        challenge["E_sec_time"] = challenge["delta_area"] / challenge["evaluation_s"]
        challenge["time_fraction_harmed"] = 0.0
        challenge["time_fraction_improved"] = 0.0
        # Duration fractions are calculated separately so they are not weighted
        # by the magnitude of dPhi.
        duration = pair.assign(
            harmed_dt=np.where(pair["dphi"] < -1e-12, pair["dt"], 0.0),
            improved_dt=np.where(pair["dphi"] > 1e-12, pair["dt"], 0.0),
        ).groupby("challenge_id").agg(harmed_dt=("harmed_dt", "sum"), improved_dt=("improved_dt", "sum"))
        challenge.loc[duration.index, "time_fraction_harmed"] = duration["harmed_dt"] / challenge.loc[duration.index, "evaluation_s"]
        challenge.loc[duration.index, "time_fraction_improved"] = duration["improved_dt"] / challenge.loc[duration.index, "evaluation_s"]
        challenge["attack_type"] = meta.loc[challenge.index, "attack_type_truth"].astype(str)
        challenge["intensity_label"] = meta.loc[challenge.index, "intensity_label_truth"].astype(str)
        strata = ["attack_type", "intensity_label"]

        g = _standardized_summary(challenge, "G_exec_time", strata)
        h = _standardized_summary(challenge, "H_exec_time", strata)
        e = _standardized_summary(challenge, "E_sec_time", strata)
        rows.append({
            "mechanism_config": cfg,
            "G_exec_time": g,
            "H_exec_time": h,
            "E_sec_time": e,
            "time_identity_error": e - (g - h),
            "time_fraction_improved": _standardized_summary(challenge, "time_fraction_improved", strata),
            "time_fraction_harmed": _standardized_summary(challenge, "time_fraction_harmed", strata),
            "N_time_local_challenges": int(len(challenge)),
            "N_time_local_reference_cells": int(challenge.groupby(strata).ngroups),
        })
    return pd.DataFrame(rows)



def cumulative_execution_quality(ts: pd.DataFrame) -> pd.DataFrame:
    """Build trial-level cumulative Q_exec(t) trajectories from Phi_exec(t).

    The runner's static Q_exec integrates Phi_exec over complete fixed steps
    whose ``time_s`` is at or after disturbance onset.  This function mirrors
    that definition exactly and reports the cumulative mean after every step:

        Q_exec(t_k) = sum_j Phi_exec_j * dt_j / sum_j dt_j.

    Time is aligned by elapsed seconds from attack onset so trials with different
    absolute attack starts remain directly comparable.
    """
    if ts.empty:
        return pd.DataFrame()
    required = {
        "trial_id", "challenge_id", "mechanism_config", "attack_type_truth",
        "attack_start_s_truth", "time_s", "next_time_s", "phi_exec",
    }
    missing = sorted(required.difference(ts.columns))
    if missing:
        raise ValueError(f"dynamic Q_exec requires time-series columns: {missing}")

    rows: list[pd.DataFrame] = []
    work = ts.copy()
    work["mechanism_config"] = work["mechanism_config"].astype(str).str.zfill(4)
    work = work[work["attack_type_truth"].astype(str) != "benign"].copy()
    for trial_id, g in work.groupby("trial_id", sort=False):
        g = g.sort_values("time_s").copy()
        attack_start = int(g["attack_start_s_truth"].iloc[0])
        if "phi_exec_evaluation_end_s_truth" in g.columns:
            evaluation_end = int(g["phi_exec_evaluation_end_s_truth"].iloc[0])
            eval_g = g[(g["time_s"].astype(int) >= attack_start) & (g["time_s"].astype(int) < evaluation_end)].copy()
        else:
            eval_g = g[g["time_s"].astype(int) >= attack_start].copy()
        if eval_g.empty:
            continue
        step = (eval_g["next_time_s"].astype(float) - eval_g["time_s"].astype(float)).to_numpy()
        phi = eval_g["phi_exec"].astype(float).to_numpy()
        valid = np.isfinite(step) & (step > 0) & np.isfinite(phi)
        if not bool(np.all(valid)):
            eval_g = eval_g.loc[valid].copy()
            step = step[valid]
            phi = phi[valid]
        if eval_g.empty:
            continue
        cumulative_duration = np.cumsum(step)
        cumulative_auc = np.cumsum(phi * step)
        out = pd.DataFrame({
            "trial_id": str(trial_id),
            "challenge_id": str(eval_g["challenge_id"].iloc[0]),
            "mechanism_config": str(eval_g["mechanism_config"].iloc[0]).zfill(4),
            "attack_type": str(eval_g["attack_type_truth"].iloc[0]),
            "intensity_label": str(eval_g["intensity_label_truth"].iloc[0]) if "intensity_label_truth" in eval_g.columns else "unknown",
            "intensity": float(eval_g["intensity_truth"].iloc[0]) if "intensity_truth" in eval_g.columns else np.nan,
            "elapsed_from_attack_start_s": (eval_g["next_time_s"].astype(int).to_numpy() - attack_start),
            "evaluation_duration_s": cumulative_duration.astype(int),
            "Q_exec_t": cumulative_auc / cumulative_duration,
            "phi_exec_current": phi,
        })
        if "attack_end_s_truth" in eval_g.columns:
            attack_end = int(eval_g["attack_end_s_truth"].iloc[0])
            out["phase"] = np.where(eval_g["next_time_s"].astype(int).to_numpy() <= attack_end, "attack", "recovery")
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def dynamic_security_effectiveness(
    q_time: pd.DataFrame,
    n_boot: int = 800,
    seed: int = 20260915,
) -> pd.DataFrame:
    """Paired dynamic effectiveness with frozen equal cell weights."""
    if q_time.empty:
        return pd.DataFrame()
    required = {"challenge_id", "mechanism_config", "elapsed_from_attack_start_s", "Q_exec_t"}
    missing = sorted(required.difference(q_time.columns))
    if missing:
        raise ValueError(f"dynamic E_sec requires cumulative-Q columns: {missing}")

    q = q_time.copy()
    q["mechanism_config"] = q["mechanism_config"].astype(str).str.zfill(4)
    rng = np.random.default_rng(seed)
    rows = []
    elapsed_values = sorted(int(x) for x in q["elapsed_from_attack_start_s"].dropna().unique())
    for elapsed in elapsed_values:
        block = q[q["elapsed_from_attack_start_s"].astype(int) == elapsed].copy()
        pivot = block.pivot(index="challenge_id", columns="mechanism_config", values="Q_exec_t")
        if "0000" not in pivot.columns:
            continue
        meta_cols = [c for c in ["attack_type", "intensity_label"] if c in block.columns]
        meta = block.drop_duplicates("challenge_id").set_index("challenge_id")
        strata = meta_cols if len(meta_cols) == 2 else []
        for cfg in sorted(set(pivot.columns).intersection(MECHANISM_CONFIGS)):
            cols = ["0000"] if cfg == "0000" else ["0000", cfg]
            pair = pivot[cols].dropna()
            if pair.empty:
                continue
            work = pd.DataFrame(index=pair.index)
            work["q_u"] = pair["0000"].astype(float)
            work["q_p"] = work["q_u"] if cfg == "0000" else pair[cfg].astype(float)
            work["delta"] = work["q_p"] - work["q_u"]
            work["gain"] = np.maximum(work["delta"], 0.0)
            work["harm"] = np.maximum(-work["delta"], 0.0)
            work["improved"] = (work["delta"] > 1e-12).astype(float)
            work["harmed"] = (work["delta"] < -1e-12).astype(float)
            for col in strata:
                work[col] = meta.loc[work.index, col].astype(str)
            e_sec = _standardized_summary(work, "delta", strata)
            gain = _standardized_summary(work, "gain", strata)
            harm = _standardized_summary(work, "harm", strata)
            boot = []
            if n_boot > 0:
                if strata:
                    groups = [g for _, g in work.groupby(strata, sort=True, dropna=False)]
                    for _ in range(int(n_boot)):
                        cell_means = []
                        for g in groups:
                            arr = g["delta"].to_numpy(float)
                            cell_means.append(float(arr[rng.integers(0, len(arr), size=len(arr))].mean()))
                        boot.append(float(np.mean(cell_means)))
                else:
                    arr = work["delta"].to_numpy(float)
                    for _ in range(int(n_boot)):
                        boot.append(float(arr[rng.integers(0, len(arr), size=len(arr))].mean()))
            ci_low, ci_high = _ci(np.asarray(boot)) if boot else (np.nan, np.nan)
            rows.append({
                "elapsed_from_attack_start_s": int(elapsed),
                "elapsed_from_attack_start_h": float(elapsed / 3600.0),
                "mechanism_config": cfg,
                "Q_exec_U_t": _standardized_summary(work, "q_u", strata),
                "Q_exec_P_t": _standardized_summary(work, "q_p", strata),
                "G_exec_t": gain,
                "H_exec_t": harm,
                "E_sec_t": e_sec,
                "E_sec_t_ci_low": ci_low,
                "E_sec_t_ci_high": ci_high,
                "identity_error": e_sec - (gain - harm),
                "N_pairs": int(len(work)),
                "N_reference_cells": int(work.groupby(strata).ngroups) if strata else 1,
                "fraction_improved": _standardized_summary(work, "improved", strata),
                "fraction_harmed": _standardized_summary(work, "harmed", strata),
            })
    return pd.DataFrame(rows)


def factorial_effects(esec: pd.DataFrame) -> pd.DataFrame:
    value = dict(zip(esec.mechanism_config.astype(str).str.zfill(4), esec.E_sec))
    if any(cfg not in value for cfg in MECHANISM_CONFIGS):
        return pd.DataFrame()
    rows = []
    for idx, name in enumerate(MECHANISM_BITS):
        deltas = []
        for cfg in MECHANISM_CONFIGS:
            if cfg[idx] == "0":
                on = cfg[:idx] + "1" + cfg[idx + 1 :]
                deltas.append(value[on] - value[cfg])
        rows.append({"effect": name, "order": 1, "estimate": float(np.mean(deltas))})
    for i, j in combinations(range(4), 2):
        vals = []
        for cfg in MECHANISM_CONFIGS:
            if cfg[i] == "0" and cfg[j] == "0":
                c00 = cfg
                c10 = cfg[:i] + "1" + cfg[i + 1 :]
                c01 = cfg[:j] + "1" + cfg[j + 1 :]
                c11 = c10[:j] + "1" + c10[j + 1 :]
                vals.append(value[c11] - value[c10] - value[c01] + value[c00])
        rows.append({"effect": f"{MECHANISM_BITS[i]}:{MECHANISM_BITS[j]}", "order": 2, "estimate": float(np.mean(vals))})
    return pd.DataFrame(rows)



def shapley_from_full_factorial(values: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Compute four-mechanism Shapley contributions for a full 2^4 coalition table."""
    value = dict(zip(values["mechanism_config"].astype(str).str.zfill(4), values[value_col].astype(float)))
    if any(cfg not in value for cfg in MECHANISM_CONFIGS):
        return pd.DataFrame()
    rows = []
    n = len(MECHANISM_BITS)
    for j, name in enumerate(MECHANISM_BITS):
        contribution = 0.0
        for cfg in MECHANISM_CONFIGS:
            if cfg[j] == "1":
                continue
            k = cfg.count("1")
            weight = factorial(k) * factorial(n - k - 1) / factorial(n)
            with_j = cfg[:j] + "1" + cfg[j + 1 :]
            contribution += weight * (value[with_j] - value[cfg])
        rows.append({"mechanism": name, "shapley": float(contribution), "coalition_value": value_col})
    total = float(sum(row["shapley"] for row in rows))
    target = float(value["1111"] - value["0000"])
    for row in rows:
        row["efficiency_target"] = target
        row["efficiency_error"] = total - target
    return pd.DataFrame(rows)

def unpaired_baseline(outcomes: pd.DataFrame) -> pd.DataFrame:
    outcomes = _attacked_only(outcomes)
    rows = []
    base = outcomes[outcomes.mechanism_config == "0000"].Y.to_numpy(int)
    for cfg in sorted(outcomes.mechanism_config.unique()):
        ys = outcomes[outcomes.mechanism_config == cfg].Y.to_numpy(int)
        p0 = float(np.mean(base))
        ps = float(np.mean(ys))
        se = float(np.sqrt(ps * (1 - ps) / len(ys) + p0 * (1 - p0) / len(base))) if len(ys) and len(base) else np.nan
        rows.append({"method": "unpaired_probability_difference", "mechanism_config": cfg, "estimate": ps - p0, "standard_error": se})
    return pd.DataFrame(rows)


def paired_unpaired_ablation(outcomes: pd.DataFrame) -> pd.DataFrame:
    outcomes = _attacked_only(outcomes)
    joined = outcomes.merge(
        outcomes[outcomes.mechanism_config == "0000"][["challenge_id", "Y"]].rename(columns={"Y": "Y_0000"}),
        on="challenge_id",
        how="inner",
        validate="many_to_one",
    )
    rows = []
    for cfg in MECHANISM_CONFIGS:
        sub = joined[joined.mechanism_config == cfg]
        if sub.empty:
            continue
        diff = sub["Y"].to_numpy(float) - sub["Y_0000"].to_numpy(float)
        ps = float(sub["Y"].mean())
        p0 = float(sub["Y_0000"].mean())
        n = int(sub["challenge_id"].nunique())
        paired_se = float(np.std(diff, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        unpaired_se = float(np.sqrt(ps * (1 - ps) / n + p0 * (1 - p0) / n)) if n else np.nan
        rows.append(
            {
                "mechanism_config": cfg,
                "paired_E_sec": float(np.mean(diff)),
                "paired_standard_error": paired_se,
                "unpaired_probability_difference": ps - p0,
                "unpaired_standard_error": unpaired_se,
                "unpaired_over_paired_se": unpaired_se / paired_se if paired_se > 0 else np.nan,
                "N_challenges": n,
            }
        )
    return pd.DataFrame(rows)


def attach_manifest(input_dir: Path, outcomes: pd.DataFrame) -> pd.DataFrame:
    challenge_path = input_dir / "challenge_manifest.csv"
    if not challenge_path.exists():
        return outcomes
    challenges = pd.read_csv(challenge_path, dtype={"challenge_id": str})
    keep = ["challenge_id", "split", "scenario", "attack_type", "disturbance_class", "intensity_label", "intensity", "attack_target", "seed", "attack_delivered", "attack_start_s", "attack_duration_s"]
    keep = [name for name in keep if name in challenges.columns]
    return outcomes.merge(challenges[keep].drop_duplicates("challenge_id"), on="challenge_id", how="left", validate="many_to_one")


def condition_results(outcomes: pd.DataFrame) -> pd.DataFrame:
    group_cols = [col for col in ["split", "scenario", "attack_type", "intensity_label", "mechanism_config"] if col in outcomes.columns]
    if not group_cols:
        group_cols = ["mechanism_config"]
    return outcomes.groupby(group_cols).Y.agg(["mean", "count"]).reset_index().rename(columns={"mean": "P_safe", "count": "N"})


def criterion_validity(input_dir: Path, outcomes: pd.DataFrame) -> pd.DataFrame:
    outcomes = _attacked_only(outcomes)
    physical_path = input_dir / "physical_criteria"
    if not table_exists(physical_path):
        return pd.DataFrame()
    physical = read_frame(physical_path)
    joined = outcomes[["trial_id", "mechanism_config", "Y"]].merge(physical, on="trial_id", how="inner")
    metric_cols = [
        "mean_abs_temperature_deviation_k",
        "max_abs_temperature_deviation_k",
        "comfort_violation_fraction",
        "time_outside_safe_envelope_fraction",
        "any_zone_outside_safe_envelope_fraction",
        "max_consecutive_any_zone_violation_s",
        "physical_recovery_envelope_fraction_last_quarter",
    ]
    rows = []
    for metric in metric_cols:
        if metric not in joined.columns or joined[metric].notna().sum() < 2 or joined["Y"].nunique() < 2:
            continue
        pair = joined[["Y", metric]].dropna()
        # Constant whole-window comfort diagnostics produce undefined
        # correlations and were previously emitted as NaN rows.  Such rows are
        # not evidence of criterion validity, so omit them from the correlation
        # table.  Baseline-relative and attack-window physical effects are
        # reported separately by the dedicated paired analyses.
        if pair[metric].nunique() < 2:
            continue
        rows.append(
            {
                "metric": metric,
                "pearson_with_Y": float(pair.corr(method="pearson").iloc[0, 1]),
                "spearman_with_Y": float(pair.corr(method="spearman").iloc[0, 1]),
                "N_trials": int(len(pair)),
            }
        )
    return pd.DataFrame(rows)



def paired_physical_effects(input_dir: Path, outcomes: pd.DataFrame, n_boot: int = 2000, seed: int = 20260914) -> pd.DataFrame:
    """Quantify paired plant-side improvement against 0000 for the same scenario.

    Positive improvement means physically better performance. These metrics are
    independent external criteria and do not feed the online defense runtime.
    """
    physical_path = input_dir / "physical_criteria"
    if not table_exists(physical_path):
        return pd.DataFrame()
    physical = read_frame(physical_path)
    outcomes = _attacked_only(outcomes)
    joined = outcomes[["trial_id", "challenge_id", "mechanism_config"]].merge(physical, on="trial_id", how="inner")
    metrics = {
        "mean_abs_temperature_deviation_k": -1.0,
        "max_abs_temperature_deviation_k": -1.0,
        "time_outside_safe_envelope_fraction": -1.0,
        "any_zone_outside_safe_envelope_fraction": -1.0,
        "max_consecutive_any_zone_violation_s": -1.0,
        "physical_recovery_envelope_fraction_last_quarter": 1.0,
    }
    rng = np.random.default_rng(seed)
    rows = []
    for metric, direction in metrics.items():
        if metric not in joined.columns:
            continue
        pivot = joined.pivot(index="challenge_id", columns="mechanism_config", values=metric)
        if "0000" not in pivot.columns:
            continue
        for cfg in MECHANISM_CONFIGS:
            if cfg not in pivot.columns:
                continue
            pair = pivot[["0000", cfg]].dropna()
            if pair.empty:
                continue
            # direction=-1: smaller is better => baseline - defended.
            # direction=+1: larger is better => defended - baseline.
            if direction < 0:
                improvement = pair["0000"].to_numpy(float) - pair[cfg].to_numpy(float)
            else:
                improvement = pair[cfg].to_numpy(float) - pair["0000"].to_numpy(float)
            boot = []
            n = len(improvement)
            for _ in range(n_boot):
                idx = rng.integers(0, n, size=n)
                boot.append(float(np.mean(improvement[idx])))
            ci_low, ci_high = _ci(np.asarray(boot))
            rows.append({
                "mechanism_config": cfg,
                "metric": metric,
                "mean_paired_physical_improvement": float(np.mean(improvement)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "fraction_improved": float(np.mean(improvement > 0)),
                "fraction_unchanged": float(np.mean(np.isclose(improvement, 0.0))),
                "N_pairs": int(n),
            })
    return pd.DataFrame(rows)

def _first_sustained_safe_time(times: np.ndarray, safe: np.ndarray, attack_end_s: int, hold_s: int) -> float:
    post = np.flatnonzero(times >= attack_end_s)
    if post.size == 0:
        return float("nan")
    if times.size >= 2:
        diffs = np.diff(times)
        step_s = int(np.nanmedian(diffs[diffs > 0])) if np.any(diffs > 0) else 300
    else:
        step_s = 300
    needed = max(1, int(np.ceil(hold_s / step_s)))
    for pos in post:
        end = pos + needed
        if end <= safe.size and bool(np.all(safe[pos:end])):
            return float(max(0, int(times[pos]) - int(attack_end_s)))
    return float("nan")


def dynamic_physical_metrics(input_dir: Path) -> pd.DataFrame:
    """Derive attack-window and post-attack plant metrics from the synchronized time series."""
    ts_path = input_dir / "physical_security_timeseries"
    if not table_exists(ts_path):
        ts_path = input_dir / "chain_state_timeseries"
    if not table_exists(ts_path):
        return pd.DataFrame()
    ts = read_frame(ts_path)
    temp_cols = [col for col in ts.columns if col.startswith("plant_post_hvac_reaZon") and col.endswith("_TZon_y")]
    if len(temp_cols) != 5:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for trial_id, group in ts.groupby("trial_id", sort=False):
        group = group.sort_values("next_time_s").copy()
        temps = group[temp_cols].to_numpy(float)
        if temps.size == 0 or not np.isfinite(temps).all():
            continue
        times = group["next_time_s"].to_numpy(int)
        attack_start_s = int(group["attack_start_s_truth"].iloc[0])
        attack_end_s = int(group["attack_end_s_truth"].iloc[0])
        deviation = np.abs(temps - 295.15)
        any_violation = np.any((temps < PHYSICAL_SAFE_MIN_K) | (temps > PHYSICAL_SAFE_MAX_K), axis=1)
        safe = ~any_violation
        attack_mask = (times > attack_start_s) & (times <= attack_end_s)
        post1h_mask = (times > attack_end_s) & (times <= attack_end_s + POST_ATTACK_WINDOW_S)
        post3h_mask = (times > attack_end_s) & (times <= attack_end_s + POST_ATTACK_RECOVERY_WINDOW_S)
        recovery_time = _first_sustained_safe_time(times, safe, attack_end_s, DYNAMIC_RECOVERY_HOLD_S)
        row = {
            "trial_id": trial_id,
            "challenge_id": str(group["challenge_id"].iloc[0]),
            "mechanism_config": str(group["mechanism_config"].iloc[0]).zfill(4),
            "attack_type": str(group["attack_type_truth"].iloc[0]) if "attack_type_truth" in group.columns else "unknown",
            "intensity_label": str(group["intensity_label_truth"].iloc[0]) if "intensity_label_truth" in group.columns else "unknown",
            "attack_start_s": attack_start_s,
            "attack_end_s": attack_end_s,
            "attack_duration_s": max(0, attack_end_s - attack_start_s),
            "attack_window_mean_abs_temperature_deviation_k": float(np.mean(deviation[attack_mask])) if np.any(attack_mask) else np.nan,
            "attack_window_max_abs_temperature_deviation_k": float(np.max(deviation[attack_mask])) if np.any(attack_mask) else np.nan,
            "attack_window_any_zone_violation_fraction": float(np.mean(any_violation[attack_mask])) if np.any(attack_mask) else np.nan,
            "post_attack_1h_mean_abs_temperature_deviation_k": float(np.mean(deviation[post1h_mask])) if np.any(post1h_mask) else np.nan,
            "post_attack_1h_any_zone_safe_fraction": float(np.mean(safe[post1h_mask])) if np.any(post1h_mask) else np.nan,
            "post_attack_3h_mean_abs_temperature_deviation_k": float(np.mean(deviation[post3h_mask])) if np.any(post3h_mask) else np.nan,
            "post_attack_3h_any_zone_safe_fraction": float(np.mean(safe[post3h_mask])) if np.any(post3h_mask) else np.nan,
            "physical_recovery_observed": int(np.isfinite(recovery_time)),
            "physical_recovery_time_s": recovery_time,
            "physical_recovery_hold_s": DYNAMIC_RECOVERY_HOLD_S,
            "recovery_censor_time_s": int(max(0, times[-1] - attack_end_s)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def paired_dynamic_physical_effects(dynamic: pd.DataFrame, n_boot: int = 2000, seed: int = 20260914) -> pd.DataFrame:
    if dynamic.empty:
        return pd.DataFrame()
    if "attack_type" in dynamic.columns:
        dynamic = dynamic[dynamic["attack_type"].astype(str) != "benign"].copy()
    metrics = {
        "attack_window_mean_abs_temperature_deviation_k": -1.0,
        "attack_window_max_abs_temperature_deviation_k": -1.0,
        "attack_window_any_zone_violation_fraction": -1.0,
        "post_attack_1h_mean_abs_temperature_deviation_k": -1.0,
        "post_attack_1h_any_zone_safe_fraction": 1.0,
        "post_attack_3h_mean_abs_temperature_deviation_k": -1.0,
        "post_attack_3h_any_zone_safe_fraction": 1.0,
        "physical_recovery_time_s": -1.0,
    }
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for metric, direction in metrics.items():
        pivot = dynamic.pivot(index="challenge_id", columns="mechanism_config", values=metric)
        if "0000" not in pivot.columns:
            continue
        for cfg in MECHANISM_CONFIGS:
            if cfg not in pivot.columns:
                continue
            pair = pivot[["0000", cfg]].dropna()
            if pair.empty:
                continue
            improvement = pair[cfg].to_numpy(float) - pair["0000"].to_numpy(float)
            if direction < 0:
                improvement = -improvement
            n = len(improvement)
            boot = [float(np.mean(improvement[rng.integers(0, n, size=n)])) for _ in range(n_boot)]
            ci_low, ci_high = _ci(np.asarray(boot))
            rows.append({
                "mechanism_config": cfg,
                "metric": metric,
                "mean_paired_dynamic_improvement": float(np.mean(improvement)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "fraction_improved": float(np.mean(improvement > 0)),
                "N_pairs": int(n),
            })
    return pd.DataFrame(rows)


def benign_relative_physical_metrics(input_dir: Path) -> pd.DataFrame:
    """Plant trajectory distance from the benign trajectory of the same mechanism.

    This is the preferred physical external criterion for the fixed peak-heat
    scenario because it respects the actual BOPTEST operating trajectory rather
    than imposing a fixed 20--24 C band that can be violated even by benign runs.
    """
    ts_path = input_dir / "physical_security_timeseries"
    if not table_exists(ts_path):
        return pd.DataFrame()
    ts = read_frame(ts_path)
    if "attack_type_truth" not in ts.columns:
        return pd.DataFrame()
    ts["mechanism_config"] = ts["mechanism_config"].astype(str).str.zfill(4)
    temp_cols = [col for col in ts.columns if col.startswith("plant_post_hvac_reaZon") and col.endswith("_TZon_y")]
    if len(temp_cols) != 5:
        return pd.DataFrame()
    refs: dict[str, pd.DataFrame] = {}
    benign = ts[ts["attack_type_truth"] == "benign"].copy()
    for cfg, g in benign.groupby("mechanism_config"):
        refs[cfg] = g.groupby("next_time_s", as_index=True)[temp_cols].mean().sort_index()
    rows: list[dict[str, object]] = []
    for trial_id, g in ts.groupby("trial_id", sort=False):
        g = g.sort_values("next_time_s")
        cfg = str(g["mechanism_config"].iloc[0]).zfill(4)
        ref = refs.get(cfg)
        if ref is None:
            continue
        gg = g.set_index("next_time_s")
        common = gg.index.intersection(ref.index)
        if common.empty:
            continue
        cur = gg.loc[common, temp_cols].to_numpy(float)
        base = ref.loc[common, temp_cols].to_numpy(float)
        delta = np.abs(cur - base)
        per_time = np.mean(delta, axis=1)
        times = common.to_numpy(int)
        attack_start = int(g["attack_start_s_truth"].iloc[0])
        attack_end = int(g["attack_end_s_truth"].iloc[0])
        attack_mask = (times > attack_start) & (times <= attack_end) if attack_end > attack_start else np.zeros(len(times), dtype=bool)
        post_mask = times > attack_end
        rows.append({
            "trial_id": trial_id,
            "challenge_id": str(g["challenge_id"].iloc[0]),
            "mechanism_config": cfg,
            "attack_type": str(g["attack_type_truth"].iloc[0]),
            "trajectory_distance_vs_benign_k": float(np.mean(per_time)),
            "trajectory_max_distance_vs_benign_k": float(np.max(delta)),
            "attack_trajectory_distance_vs_benign_k": float(np.mean(per_time[attack_mask])) if np.any(attack_mask) else np.nan,
            "post_attack_trajectory_distance_vs_benign_k": float(np.mean(per_time[post_mask])) if np.any(post_mask) else np.nan,
        })
    return pd.DataFrame(rows)



def physical_metric_sensitivity(input_dir: Path) -> pd.DataFrame:
    """Rank BOPTEST plant temperatures by attack sensitivity.

    Candidate metrics are restricted to plant-side temperatures already logged
    by the runner: five zone temperatures and available supply-air/supply-water
    temperatures.  Sensitivity is the mean absolute deviation from the benign
    trajectory during the attack window.  Because these candidates share kelvin
    units, their magnitudes are directly comparable.  The 1111 column reports
    whether full protection reduces that physical deviation.
    """
    ts_path = input_dir / "physical_security_timeseries"
    if not table_exists(ts_path):
        return pd.DataFrame()
    ts = read_frame(ts_path)
    if ts.empty or "attack_type_truth" not in ts.columns:
        return pd.DataFrame()
    ts = ts.copy()
    ts["mechanism_config"] = ts["mechanism_config"].astype(str).str.zfill(4)

    zone_cols = [
        c for c in ts.columns
        if c.startswith("plant_post_hvac_reaZon") and c.endswith("_TZon_y")
    ]
    supply_cols = [
        c for c in ts.columns
        if c.startswith("plant_diag_post_") and c.endswith("_TSup_y")
    ]
    candidate_cols = zone_cols + supply_cols
    if not candidate_cols:
        return pd.DataFrame()

    refs: dict[str, pd.DataFrame] = {}
    benign = ts[ts["attack_type_truth"].astype(str) == "benign"].copy()
    for cfg, g in benign.groupby("mechanism_config", sort=False):
        refs[str(cfg).zfill(4)] = (
            g.groupby("next_time_s", as_index=True)[candidate_cols]
            .mean(numeric_only=True)
            .sort_index()
        )

    detail_rows: list[dict[str, object]] = []
    attacked = ts[ts["attack_type_truth"].astype(str) != "benign"].copy()
    for trial_id, g in attacked.groupby("trial_id", sort=False):
        g = g.sort_values("next_time_s")
        cfg = str(g["mechanism_config"].iloc[0]).zfill(4)
        ref = refs.get(cfg)
        if ref is None:
            continue
        gg = g.set_index("next_time_s")
        common = gg.index.intersection(ref.index)
        if common.empty:
            continue
        attack_start = int(g["attack_start_s_truth"].iloc[0])
        attack_end = int(g["attack_end_s_truth"].iloc[0])
        times = common.to_numpy(int)
        attack_mask = (times > attack_start) & (times <= attack_end)
        post_mask = times > attack_end
        if not np.any(attack_mask):
            continue
        for metric in candidate_cols:
            cur = gg.loc[common, metric].to_numpy(float)
            base = ref.loc[common, metric].to_numpy(float)
            finite = np.isfinite(cur) & np.isfinite(base)
            amask = attack_mask & finite
            pmask = post_mask & finite
            if not np.any(amask):
                continue
            delta = np.abs(cur - base)
            family = "zone_temperature" if metric in zone_cols else "supply_temperature"
            detail_rows.append({
                "trial_id": str(trial_id),
                "challenge_id": str(g["challenge_id"].iloc[0]),
                "attack_type": str(g["attack_type_truth"].iloc[0]),
                "intensity_label": str(g["intensity_label_truth"].iloc[0]) if "intensity_label_truth" in g.columns else "unknown",
                "intensity": float(g["intensity_truth"].iloc[0]) if "intensity_truth" in g.columns else np.nan,
                "mechanism_config": cfg,
                "physical_metric": metric,
                "metric_family": family,
                "attack_mean_abs_deviation_vs_benign_k": float(np.mean(delta[amask])),
                "attack_max_abs_deviation_vs_benign_k": float(np.max(delta[amask])),
                "post_attack_mean_abs_deviation_vs_benign_k": float(np.mean(delta[pmask])) if np.any(pmask) else np.nan,
            })
    if not detail_rows:
        return pd.DataFrame()
    detail = pd.DataFrame(detail_rows)

    rows: list[dict[str, object]] = []
    attacks = list(dict.fromkeys(detail["attack_type"].astype(str).tolist()))
    for attack in attacks + ["ALL"]:
        block = detail if attack == "ALL" else detail[detail["attack_type"].astype(str) == attack]
        for metric, m in block.groupby("physical_metric", sort=False):
            base = m[m["mechanism_config"] == "0000"]
            full = m[m["mechanism_config"] == "1111"]
            if base.empty:
                continue
            bmean = float(base["attack_mean_abs_deviation_vs_benign_k"].mean())
            bmax = float(base["attack_max_abs_deviation_vs_benign_k"].mean())
            fmean = float(full["attack_mean_abs_deviation_vs_benign_k"].mean()) if not full.empty else np.nan
            improvement = bmean - fmean if np.isfinite(fmean) else np.nan
            reduction = improvement / bmean if np.isfinite(improvement) and bmean > 1e-12 else np.nan
            rows.append({
                "attack_type": attack,
                "physical_metric": metric,
                "metric_family": str(m["metric_family"].iloc[0]),
                "attack_mean_deviation_0000_k": bmean,
                "attack_max_deviation_0000_k": bmax,
                "attack_mean_deviation_1111_k": fmean,
                "full_protection_improvement_k": improvement,
                "full_protection_reduction_fraction": reduction,
                "N_0000_trials": int(len(base)),
                "N_1111_trials": int(len(full)),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["sensitivity_rank_within_attack"] = (
        out.groupby("attack_type")["attack_mean_deviation_0000_k"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    return out.sort_values(["attack_type", "sensitivity_rank_within_attack", "physical_metric"]).reset_index(drop=True)

def paired_benign_relative_effects(relative: pd.DataFrame, n_boot: int = 2000, seed: int = 20260914) -> pd.DataFrame:
    if relative.empty:
        return pd.DataFrame()
    metrics = (
        "trajectory_distance_vs_benign_k",
        "trajectory_max_distance_vs_benign_k",
        "attack_trajectory_distance_vs_benign_k",
        "post_attack_trajectory_distance_vs_benign_k",
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    attacked = relative[relative["attack_type"] != "benign"].copy()
    for metric in metrics:
        pivot = attacked.pivot(index="challenge_id", columns="mechanism_config", values=metric)
        if "0000" not in pivot.columns:
            continue
        for cfg in MECHANISM_CONFIGS:
            if cfg not in pivot.columns:
                continue
            pair = pivot[["0000", cfg]].dropna()
            if pair.empty:
                continue
            improvement = pair["0000"].to_numpy(float) - pair[cfg].to_numpy(float)
            n = len(improvement)
            boot = [float(np.mean(improvement[rng.integers(0, n, size=n)])) for _ in range(n_boot)]
            ci_low, ci_high = _ci(np.asarray(boot))
            rows.append({
                "mechanism_config": cfg, "metric": metric,
                "mean_paired_improvement_toward_benign_k": float(np.mean(improvement)),
                "ci_low": ci_low, "ci_high": ci_high,
                "fraction_improved": float(np.mean(improvement > 0)), "N_pairs": int(n),
            })
    return pd.DataFrame(rows)



def experiment_validity_diagnostics(
    esec: pd.DataFrame, condition: pd.DataFrame, benign_relative: pd.DataFrame, outcomes: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Surface structural experiment problems before results are interpreted.

    These checks are descriptive diagnostics.  They do not alter E_sec and do
    not use physical outcomes to tune the online protection runtime.
    """
    rows: list[dict[str, object]] = []

    if outcomes is not None and not outcomes.empty:
        attacked_out = _attacked_only(outcomes)
        attacked_out = attacked_out.copy()
        attacked_out["mechanism_config"] = attacked_out["mechanism_config"].astype(str).str.zfill(4)
        per_challenge = attacked_out.groupby("challenge_id")["mechanism_config"].nunique()
        rows.append({
            "diagnostic": "full_2x4_mechanism_factorial_complete",
            "value": int(per_challenge.min()) if len(per_challenge) else 0,
            "status": "PASS" if len(per_challenge) and bool((per_challenge == len(MECHANISM_CONFIGS)).all()) else "FAIL",
            "detail": f"attacked_challenges={len(per_challenge)};min_configs={int(per_challenge.min()) if len(per_challenge) else 0};max_configs={int(per_challenge.max()) if len(per_challenge) else 0}",
        })
        if {"attack_type", "intensity_label"}.issubset(attacked_out.columns):
            base = attacked_out[attacked_out["mechanism_config"] == "0000"].copy()
            cell_counts = base.groupby(["attack_type", "intensity_label"])["challenge_id"].nunique()
            expected_cells = 16
            # Balance is a structural property of the realized design.  Do not
            # hard-code N=20 here: the same frozen formal design is intentionally
            # run with N=1 for screening before the final N=20 campaign.  Final
            # campaign size is enforced separately by formal_postflight through
            # --expected-trials (5136 for N=20, 272 for N=1 screening).
            expected_n = int(cell_counts.iloc[0]) if len(cell_counts) else 0
            pass_cells = (
                len(cell_counts) == expected_cells
                and expected_n >= 1
                and bool((cell_counts == expected_n).all())
            )
            rows.append({
                "diagnostic": "balanced_attack_intensity_reference_set",
                "value": int(len(cell_counts)),
                "status": "PASS" if pass_cells else "FAIL",
                "detail": f"cells={len(cell_counts)};counts={sorted(set(int(x) for x in cell_counts.tolist())) if len(cell_counts) else []};expected_per_cell={expected_n}",
            })
        base = attacked_out[attacked_out["mechanism_config"] == "0000"].copy()
        if "attack_delivered" in base.columns:
            delivered = base["attack_delivered"].astype(bool)
            rows.append({
                "diagnostic": "all_formal_attacks_delivered",
                "value": int(delivered.sum()),
                "status": "PASS" if len(delivered) and bool(delivered.all()) else "FAIL",
                "detail": f"delivered={int(delivered.sum())}/{len(delivered)}",
            })
        if "attack_target_control_state_changed_truth" in base.columns:
            changed = base["attack_target_control_state_changed_truth"].fillna(0).astype(int) == 1
            rows.append({
                "diagnostic": "all_attacks_change_target_control_state",
                "value": int(changed.sum()),
                "status": "PASS" if len(changed) and bool(changed.all()) else "FAIL",
                "detail": f"changed={int(changed.sum())}/{len(changed)}",
            })
        if "unprotected_final_control_state_changed_truth" in base.columns:
            changed = base["unprotected_final_control_state_changed_truth"].fillna(0).astype(int) == 1
            rows.append({
                "diagnostic": "all_attacks_change_unprotected_final_control_state",
                "value": int(changed.sum()),
                "status": "PASS" if len(changed) and bool(changed.all()) else "FAIL",
                "detail": f"changed={int(changed.sum())}/{len(changed)}",
            })
        if "phi_exec_evaluation_window_s" in attacked_out.columns:
            windows = attacked_out["phi_exec_evaluation_window_s"].dropna().astype(int)
            rows.append({
                "diagnostic": "common_phi_exec_evaluation_window",
                "value": int(windows.nunique()),
                "status": "PASS" if len(windows) and windows.nunique() == 1 else "FAIL",
                "detail": f"windows_s={sorted(windows.unique().tolist())[:12]}",
            })

    if not esec.empty and {"mechanism_config", "E_sec"}.issubset(esec.columns):
        value = {str(r.mechanism_config).zfill(4): float(r.E_sec) for r in esec.itertuples()}
        iso_deltas = []
        for cfg in sorted(value):
            if len(cfg) == 4 and cfg[2] == "0":
                with_iso = cfg[:2] + "1" + cfg[3:]
                if with_iso in value:
                    iso_deltas.append(value[with_iso] - value[cfg])
        if iso_deltas:
            rows.append({
                "diagnostic": "isolation_marginal_effect_nonzero",
                "value": float(max(abs(x) for x in iso_deltas)),
                "status": "PASS" if any(abs(x) > 1e-12 for x in iso_deltas) else "FAIL",
                "detail": f"max_abs_toggle_delta_E_sec={max(abs(x) for x in iso_deltas):.6g};pairs={len(iso_deltas)}",
            })

    if not benign_relative.empty:
        rel = benign_relative.copy()
        rel["mechanism_config"] = rel["mechanism_config"].astype(str).str.zfill(4)
        drop = rel[(rel["attack_type"] == "sensor_dropout") & (rel["mechanism_config"] == "0000")]
        if not drop.empty and "attack_trajectory_distance_vs_benign_k" in drop.columns:
            vals = drop["attack_trajectory_distance_vs_benign_k"].astype(float).to_numpy()
            mean = float(np.nanmean(vals))
            nonzero_fraction = float(np.mean(np.abs(vals) > 1e-12))
            rows.append({
                "diagnostic": "sensor_dropout_has_unprotected_physical_effect",
                "value": mean,
                "status": "PASS" if nonzero_fraction > 0.0 else "FAIL",
                "detail": f"mean_attack_window_distance_k={mean:.6g};nonzero_fraction={nonzero_fraction:.3f};N={len(vals)}",
            })

        # Combined attacks can exhibit genuine non-additive interaction, but a
        # severe collapse relative to both constituent families is worth
        # flagging because the current command-attack rule acts on the
        # sensor-corrupted command and can partially counteract it.  This is a
        # warning only; it never changes the attack or protection runtime.
        base0000 = rel[rel["mechanism_config"] == "0000"]
        means = (base0000[base0000["attack_type"].isin(["sensor_bias", "command_injection", "combined"])]
                 .groupby("attack_type")["attack_trajectory_distance_vs_benign_k"].mean())
        if {"sensor_bias", "command_injection", "combined"}.issubset(set(means.index)):
            constituent_floor = float(min(means["sensor_bias"], means["command_injection"]))
            combined_mean = float(means["combined"])
            ratio = combined_mean / constituent_floor if constituent_floor > 1e-12 else float("nan")
            rows.append({
                "diagnostic": "combined_attack_physical_effect_not_collapsed",
                "value": ratio,
                "status": "PASS" if (not np.isfinite(ratio) or ratio >= 0.75) else "WARN",
                "detail": f"combined_to_weaker_constituent_ratio={ratio:.3f};combined={combined_mean:.6g};constituent_floor={constituent_floor:.6g}",
            })

    if not condition.empty and {"attack_type", "intensity_label", "mechanism_config", "P_safe"}.issubset(condition.columns):
        c = condition.copy()
        c["mechanism_config"] = c["mechanism_config"].astype(str).str.zfill(4)
        attacked = c[c["attack_type"] != "benign"]
        negative = []
        for r in attacked.itertuples():
            cfg = str(r.mechanism_config).zfill(4)
            if cfg[3] != "0":
                continue
            with_rec = cfg[:3] + "1"
            partner = attacked[(attacked["attack_type"] == r.attack_type)
                               & (attacked["intensity_label"] == r.intensity_label)
                               & (attacked["mechanism_config"] == with_rec)]
            if partner.empty:
                continue
            delta = float(partner.iloc[0]["P_safe"] - r.P_safe)
            if delta < -1e-12:
                negative.append((str(r.attack_type), str(r.intensity_label), cfg, with_rec, delta))
        rows.append({
            "diagnostic": "recovery_toggle_negative_binary_cells",
            "value": int(len(negative)),
            "status": "PASS" if not negative else "WARN",
            "detail": ";".join(f"{a}/{i}:{c0}->{c1}:{d:+.3f}" for a, i, c0, c1, d in negative[:12]),
        })

    return pd.DataFrame(rows)

def baseline_comparison(outcomes: pd.DataFrame) -> pd.DataFrame:
    attacked = _attacked_only(outcomes)
    base = unpaired_baseline(attacked)
    pag = pag_risk_reduction(attacked) if {"attack_type", "intensity"}.issubset(attacked.columns) else pd.DataFrame()
    if not pag.empty:
        for col in ("standard_error",):
            pag[col] = np.nan
        base["mean_base_risk"] = np.nan
        base["mean_defended_risk"] = np.nan
        return pd.concat([base, pag[base.columns]], ignore_index=True)
    return base


def analyze(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = read_frame(input_dir / "paired_outcomes")
    outcomes["mechanism_config"] = outcomes["mechanism_config"].astype(str).str.zfill(4)
    outcomes = attach_manifest(input_dir, outcomes)
    legacy_esec = paired_stats(outcomes)
    esec = unified_security_effectiveness(outcomes)
    factorial_source = esec[["mechanism_config", "E_sec"]] if not esec.empty else legacy_esec
    factorial = factorial_effects(factorial_source)
    theory_shapley = shapley_from_full_factorial(esec, "E_sec") if not esec.empty else pd.DataFrame()
    baseline = baseline_comparison(outcomes)
    ablation = paired_unpaired_ablation(outcomes)
    condition = condition_results(outcomes)
    criterion = criterion_validity(input_dir, outcomes)
    physical_effects = paired_physical_effects(input_dir, outcomes)
    dynamic_physical = dynamic_physical_metrics(input_dir)
    dynamic_effects = paired_dynamic_physical_effects(dynamic_physical)
    benign_relative = benign_relative_physical_metrics(input_dir)
    benign_relative_effects = paired_benign_relative_effects(benign_relative)
    validity_diagnostics = experiment_validity_diagnostics(esec, condition, benign_relative, outcomes=outcomes)
    ts_path = input_dir / "physical_security_timeseries"
    if not table_exists(ts_path):
        ts_path = input_dir / "chain_state_timeseries"
    if table_exists(ts_path):
        ts_frame = read_frame(ts_path)
        q_time = cumulative_execution_quality(ts_frame)
        dynamic_esec = dynamic_security_effectiveness(q_time)
        time_local_esec = time_local_security_effectiveness(ts_frame)
    else:
        q_time = pd.DataFrame()
        dynamic_esec = pd.DataFrame()
        time_local_esec = pd.DataFrame()
    if not time_local_esec.empty and not esec.empty:
        esec = esec.merge(time_local_esec, on="mechanism_config", how="left", validate="one_to_one")
    physical_sensitivity = physical_metric_sensitivity(input_dir)
    esec.to_csv(output_dir / "BOPTEST_SECURITY_PROTECTION_EFFECTIVENESS.csv", index=False)
    legacy_esec.to_csv(output_dir / "BOPTEST_ESEC_RESULTS_LEGACY_ALL_SCENARIOS.csv", index=False)
    if not theory_shapley.empty:
        theory_shapley.to_csv(output_dir / "BOPTEST_SECURITY_EFFECTIVENESS_SHAPLEY.csv", index=False)
    factorial.to_csv(output_dir / "BOPTEST_FACTORIAL_EFFECTS.csv", index=False)
    baseline.to_csv(output_dir / "BOPTEST_BASELINE_COMPARISON.csv", index=False)
    ablation.to_csv(output_dir / "BOPTEST_PAIRED_UNPAIRED_ABLATION.csv", index=False)
    condition.to_csv(output_dir / "BOPTEST_CONDITION_RESULTS.csv", index=False)
    if not criterion.empty:
        criterion.to_csv(output_dir / "BOPTEST_CRITERION_VALIDITY.csv", index=False)
    if not physical_effects.empty:
        physical_effects.to_csv(output_dir / "BOPTEST_PAIRED_PHYSICAL_EFFECTS.csv", index=False)
    if not dynamic_physical.empty:
        dynamic_physical.to_csv(output_dir / "BOPTEST_DYNAMIC_PHYSICAL_METRICS.csv", index=False)
    if not dynamic_effects.empty:
        dynamic_effects.to_csv(output_dir / "BOPTEST_PAIRED_DYNAMIC_PHYSICAL_EFFECTS.csv", index=False)
    if not benign_relative.empty:
        benign_relative.to_csv(output_dir / "BOPTEST_BENIGN_RELATIVE_PHYSICAL_METRICS.csv", index=False)
    if not benign_relative_effects.empty:
        benign_relative_effects.to_csv(output_dir / "BOPTEST_PAIRED_BENIGN_RELATIVE_EFFECTS.csv", index=False)
    if not q_time.empty:
        q_time.to_csv(output_dir / "BOPTEST_DYNAMIC_Q_EXEC_TIMESERIES.csv", index=False)
    if not dynamic_esec.empty:
        dynamic_esec.to_csv(output_dir / "BOPTEST_DYNAMIC_SECURITY_EFFECTIVENESS.csv", index=False)
    if not time_local_esec.empty:
        time_local_esec.to_csv(output_dir / "BOPTEST_TIME_LOCAL_SECURITY_EFFECTIVENESS.csv", index=False)
    if not physical_sensitivity.empty:
        physical_sensitivity.to_csv(output_dir / "BOPTEST_PHYSICAL_METRIC_SENSITIVITY.csv", index=False)
    if not validity_diagnostics.empty:
        validity_diagnostics.to_csv(output_dir / "BOPTEST_EXPERIMENT_VALIDITY_DIAGNOSTICS.csv", index=False)
    report = [
        "# BOPTEST Statistical Report",
        "",
        f"Outcome rows: {len(outcomes)}",
        f"Challenge blocks: {outcomes.challenge_id.nunique()}",
        f"Mechanism configurations: {outcomes.mechanism_config.nunique()}",
        "",
        "Primary estimand: the 16 attack-type x intensity cells form a frozen equal-weight reference set. Continuous security protection effectiveness is E_sec=sum_h pi_h E[Q_exec(P)-Q_exec(U)|h]=G_exec-H_exec with pi_h=1/16. Q_exec uses the common attack-relative evaluation horizon. Bootstrap resampling is stratified within reference cells. Binary Y is auxiliary only.",
        "",
        "## Security Protection Effectiveness",
        esec.to_markdown(index=False) if not esec.empty else "No attacked paired scenarios available.",
        "",
        "## Factorial Effects",
        factorial.to_markdown(index=False),
        "",
        "## Security-effectiveness Shapley",
        theory_shapley.to_markdown(index=False) if not theory_shapley.empty else "Full 2^4 coalition table unavailable.",
        "",
        "## Baseline Comparison",
        baseline.to_markdown(index=False),
        "",
        "## Paired vs Unpaired Ablation",
        ablation.to_markdown(index=False),
    ]
    # Absolute comfort-envelope validity is retained in raw diagnostics only.
    # Formal plant-side validation is based on deviation from the matched benign
    # BOPTEST trajectory and is reported below.
    if not physical_effects.empty:
        report.extend(["", "## Paired Physical Improvement", physical_effects.to_markdown(index=False)])
    if not dynamic_effects.empty:
        report.extend(["", "## Paired Dynamic Physical Improvement", dynamic_effects.to_markdown(index=False)])
    if not benign_relative_effects.empty:
        report.extend(["", "## Paired Improvement Toward Benign Physical Trajectory", benign_relative_effects.to_markdown(index=False)])
    if not dynamic_esec.empty:
        report.extend(["", "## Dynamic Security Protection Effectiveness", dynamic_esec.to_markdown(index=False)])
    if not time_local_esec.empty:
        report.extend([
            "",
            "## Time-local Protection Gain and Harm",
            "G_exec_time/H_exec_time decompose instantaneous paired Phi_exec differences before time integration, so short protection-action harm remains visible even when the full-trial net H_exec is zero.",
            time_local_esec.to_markdown(index=False),
        ])
    if not physical_sensitivity.empty:
        top = physical_sensitivity[physical_sensitivity["sensitivity_rank_within_attack"] <= 3]
        report.extend(["", "## Most Attack-Sensitive BOPTEST Physical Metrics", top.to_markdown(index=False)])
    if not validity_diagnostics.empty:
        report.extend(["", "## Experiment Validity Diagnostics", validity_diagnostics.to_markdown(index=False)])
    (output_dir / "BOPTEST_STATISTICAL_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    final_report = [
        "# BOPTEST Final Experiment Report",
        "",
        "## Conclusion",
        "This report summarizes the currently analyzed BOPTEST run directory. Treat the result as final only if the input directory contains the frozen development and formal protocol outputs.",
        "",
        f"Challenge blocks: {outcomes.challenge_id.nunique()}",
        f"Outcome rows: {len(outcomes)}",
        "",
        "## Primary Security Protection Effectiveness",
        esec.to_markdown(index=False) if not esec.empty else "Unavailable",
        "",
        "## Mechanism Effects",
        factorial.to_markdown(index=False),
        "",
        "## Shapley Contributions",
        theory_shapley.to_markdown(index=False) if not theory_shapley.empty else "Unavailable",
        "",
        "## Baselines",
        baseline.to_markdown(index=False),
        "",
        "## Paired vs Unpaired Ablation",
        ablation.to_markdown(index=False),
        "",
    ]
    if not validity_diagnostics.empty:
        final_report.extend([
            "",
            "## Experiment Validity Diagnostics",
            validity_diagnostics.to_markdown(index=False),
        ])
        if (validity_diagnostics["status"].astype(str) == "FAIL").any():
            final_report.extend(["", "Result status: NOT READY TO FREEZE. At least one structural validity diagnostic failed."])
    final_report.extend([
        "",
        "## Interpretation Guard",
        "E_sec is computed from paired continuous Phi_exec trajectories: Q_exec=(1/T) integral Phi_exec dt and E_sec=mean(Q_exec(P)-Q_exec(U)). Positive and negative changes are decomposed as G_exec and H_exec. Binary Y remains a boundary diagnostic only. Plant-side validation uses matched benign-relative trajectories and does not feed the online protection runtime.",
    ])
    (output_dir / "BOPTEST_FINAL_EXPERIMENT_REPORT.md").write_text("\n".join(final_report), encoding="utf-8")
    (output_dir / "analysis_manifest.txt").write_text(
        json.dumps({"status": "complete", "outcome_hash": canonical_hash(outcomes.to_dict("records"))}, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
