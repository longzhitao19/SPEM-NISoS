import hashlib
"""Reproduce Section 4.1.5 from the current analytical model and compare_al.

Run: python analyze_method_comparison.py
The legacy W_SoSRM_gain column is deliberately excluded. No R/H values from
the older scene-net decomposition are imported into the current manuscript.
"""
from pathlib import Path
import itertools
import pprint
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

ROOT = Path(__file__).resolve().parent
COMPARISON = ROOT / "comparison_reference"
OUT = ROOT / "comparison"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "source"))
import caculate_th as model


def calculate():
    original = pd.read_csv(COMPARISON / "example_data/all_48_comparison_scores.csv")
    cols = ["E_Q", "terminal_gain", "binary_tau_050", "binary_tau_060", "binary_tau_070"]
    rows = []
    for topology in model.TOPOLOGIES:
        for config in model.ALL_SUBSETS:
            q_p = np.array([model.q_chain(c, config, topology) for c in model.SCENARIOS])
            q_u = np.array([model.q_chain(c, (), topology) for c in model.SCENARIOS])
            terminal = np.mean([
                model.phi_exec(model.T, c, config, topology)
                - model.phi_exec(model.T, c, (), topology) for c in model.SCENARIOS
            ])
            row = {"topology": topology, "config": model.config_name(config),
                   "E_Q": float(np.mean(q_p - q_u)), "terminal_gain": terminal}
            for theta in (0.50, 0.60, 0.70):
                row[f"binary_tau_{round(theta * 100):03d}"] = float(np.mean(
                    (q_p >= theta).astype(int) - (q_u >= theta).astype(int)))
            rows.append(row)
    scores = pd.DataFrame(rows)
    joined = scores.merge(original[["topology", "config", *cols]],
                          on=["topology", "config"], suffixes=("", "_frozen"),
                          validate="one_to_one")
    assert len(joined) == len(scores) == 48
    errors = {}
    for col in cols:
        errors[col] = float(np.max(np.abs(joined[col] - joined[col + "_frozen"])))
        np.testing.assert_allclose(joined[col], joined[col + "_frozen"], atol=1e-12, rtol=0)
    scores.to_csv(OUT / "all_48_scores_verified.csv", index=False, encoding="utf-8-sig")

    # Every eligible path has exactly four links and four internal nodes.
    # Double counting path-unit incidence gives the equal-weight mean loss
    # without assuming an undocumented adjacency matrix.
    topo = pd.read_csv(COMPARISON / "example_data/A_topology_comparison.csv")
    topo["W_link_recomputed"] = 1 - 2 / topo.n_links
    topo["W_scene_recomputed"] = 1 - 4 / (topo.n_links + topo.n_eligible_nodes)
    for src, new in [("W_link_only", "W_link_recomputed"),
                     ("W_scene_node_link", "W_scene_recomputed")]:
        np.testing.assert_allclose(topo[src], topo[new], atol=1e-12, rtol=0)
    full = scores[scores.config == "RCIH"].set_index("topology")
    np.testing.assert_allclose(topo.E_Q_RCIH, full.loc[topo.topology].E_Q, atol=1e-12, rtol=0)
    # The table only uses E_Q, so the older scene-net R/H columns are omitted.
    topo = topo[["topology", "n_nodes_total", "n_eligible_nodes", "n_links",
                 "n_nominal_functional_paths", "W_link_recomputed", "W_scene_recomputed", "E_Q_RCIH"]]
    topo.to_csv(OUT / "topology_comparison_verified.csv", index=False)
    rank_rows = []
    for col in ["W_link_recomputed", "W_scene_recomputed"]:
        rank_rows.append({"method": col,
                          "spearman_rho": float(spearmanr(topo[col], topo.E_Q_RCIH).statistic),
                          "kendall_tau_b": float(kendalltau(topo[col], topo.E_Q_RCIH).statistic)})
    pd.DataFrame(rank_rows).to_csv(OUT / "rank_consistency_verified.csv", index=False)

    pairs = list(itertools.combinations(range(48), 2))
    distinct = [(i, j) for i, j in pairs if not np.isclose(
        scores.E_Q[i], scores.E_Q[j], atol=1e-12, rtol=0)]
    metrics = []
    for col in cols:
        x = scores[col].to_numpy()
        ties = sum(np.isclose(x[i], x[j], atol=1e-12, rtol=0) for i, j in pairs)
        conflicts = sum(np.isclose(x[i], x[j], atol=1e-12, rtol=0) for i, j in distinct)
        n_unique = len(np.unique(x.round(12)))
        metrics.append({"source_column": col, "n_unique_scores": n_unique,
                        "resolution_ratio": n_unique / 48,
                        "tied_pairs": int(ties), "total_pairs": len(pairs),
                        "global_pair_tie_rate": ties / len(pairs),
                        "conflicting_pairs": int(conflicts), "reference_distinct_pairs": len(distinct),
                        "conflicting_tie_rate_vs_reference": conflicts / len(distinct),
                        "pairwise_discrimination_retention": 1 - conflicts / len(distinct)})
    disc = pd.DataFrame(metrics)
    frozen_disc = pd.read_csv(COMPARISON / "example_results/XX3_discrimination.csv").set_index("source_column")
    check_cols = ["n_unique_scores", "resolution_ratio", "global_pair_tie_rate",
                  "conflicting_tie_rate_vs_reference", "pairwise_discrimination_retention"]
    np.testing.assert_allclose(disc.set_index("source_column")[check_cols],
                               frozen_disc.loc[cols, check_cols], atol=1e-12, rtol=0)
    disc.to_csv(OUT / "discrimination_verified.csv", index=False)
    audit = {"status": "PASS", "evaluated_objects": 48, "scenarios_per_object": 2,
             "scenario_weights": {"F": 0.5, "C": 0.5}, "window": [0, 10],
             "max_absolute_errors_vs_compare_al": errors, "total_pairs": len(pairs),
             "reference_distinct_pairs": len(distinct), "rank_results": rank_rows,
             "W_verification": "Path-unit incidence identity; topology counts from frozen input",
             "excluded_legacy_column": "W_SoSRM_gain",
             "R_H_convention": "Existing manuscript timewise positive/negative integrals retained",
             "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                 ROOT / "source/caculate_th.py",
                 COMPARISON / "example_data/A_topology_comparison.csv",
                 COMPARISON / "example_data/all_48_comparison_scores.csv"]}}
    (OUT / "comparison_audit.txt").write_text(pprint.pformat(audit, sort_dicts=False), encoding="utf-8")
    print(disc.to_string(index=False))
    print(pprint.pformat(audit, sort_dicts=False))
    return disc



if __name__ == '__main__':
    calculate()
