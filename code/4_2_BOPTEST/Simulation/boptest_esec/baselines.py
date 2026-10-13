from __future__ import annotations

import pandas as pd


def pag_component_probability(intensity: float) -> float:
    """Map frozen BOPTEST attack intensity to a PAG component probability."""
    return float(max(0.0, min(1.0, intensity)))


def pag_defended_risk(row: pd.Series) -> float:
    """Construct-comparison PAG risk under the frozen BOPTEST topology mapping.

    This is a probabilistic attack graph construct, not an E_sec estimate. It
    treats attack success as reachability to an unsafe final command through
    compromised S or A elements, then removes paths when the mechanism
    configuration contains a graph-level mitigation for that compromised path.
    """
    attack_type = str(row["attack_type"])
    if attack_type == "benign":
        return 0.0
    p = pag_component_probability(float(row["intensity"]))
    cfg = str(row["mechanism_config"]).zfill(4)
    m_red, m_cfg, m_iso, _m_rec = (bit == "1" for bit in cfg)

    sensor_risk = 0.0
    command_risk = 0.0
    if attack_type in {"sensor_bias", "sensor_dropout", "combined"}:
        sensor_mitigated = m_iso or m_red or m_cfg
        sensor_risk = 0.0 if sensor_mitigated else p
    if attack_type in {"command_injection", "combined"}:
        command_mitigated = m_iso or m_red or m_cfg
        command_risk = 0.0 if command_mitigated else p
    return float(1.0 - (1.0 - sensor_risk) * (1.0 - command_risk))


def pag_risk_reduction(outcomes: pd.DataFrame) -> pd.DataFrame:
    required = {"challenge_id", "mechanism_config", "attack_type", "intensity"}
    missing = required.difference(outcomes.columns)
    if missing:
        raise ValueError(f"PAG baseline requires columns: {sorted(missing)}")
    work = outcomes.copy()
    work["mechanism_config"] = work["mechanism_config"].astype(str).str.zfill(4)
    work["pag_defended_risk"] = work.apply(pag_defended_risk, axis=1)
    base = (
        work[work["mechanism_config"] == "0000"][["challenge_id", "pag_defended_risk"]]
        .rename(columns={"pag_defended_risk": "pag_base_risk"})
        .copy()
    )
    work = work.merge(base, on="challenge_id", how="left", validate="many_to_one")
    work["estimate"] = work["pag_base_risk"] - work["pag_defended_risk"]
    return (
        work.groupby("mechanism_config", as_index=False)
        .agg(estimate=("estimate", "mean"), mean_defended_risk=("pag_defended_risk", "mean"), mean_base_risk=("pag_base_risk", "mean"))
        .assign(method="pag_attack_graph_risk_reduction")
        [["method", "mechanism_config", "estimate", "mean_base_risk", "mean_defended_risk"]]
    )
