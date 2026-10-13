import itertools
import math
import numpy as np
import pandas as pd


# ============================================================
# 1. Global parameters
# ============================================================

MECHS = ("R", "C", "I", "H")
TOPOLOGIES = ("G_B", "G_D", "G_E")
SCENARIOS = ("F", "C")

SCENARIO_WEIGHT = {
    "F": 0.5,
    "C": 0.5,
}

# Observation window
T = 10.0

# Disturbance interval
T_ATTACK_START = 2.0
T_ATTACK_END = 6.0

# Strict safety response deadline
T_STRICT_DEADLINE = 3.5

# Mechanism completion time
T_MECH = {
    "I": 2.5,
    "R": 2.9,
    "C": 3.4,
    "H": 7.5,
}

# Raw-state normalization constants
D_MAX = 1.0
N_REQ = 5
C_REQ = 100.0

L_TARGET = 0.20
L_MAX = 1.00


# ============================================================
# 2. State normalization
# ============================================================

def timeliness_score(latency):
    """
    Convert raw path delay L into normalized timeliness ell.
    """
    if latency <= L_TARGET:
        return 1.0

    if latency >= L_MAX:
        return 0.0

    return (
        (L_MAX - latency)
        / (L_MAX - L_TARGET)
    )


def state_from_raw(
    d,
    n_available,
    capacity,
    latency,
):
    """
    Construct b, a, s, ell and Phi from raw quantities.
    """

    b = 1.0 - min(
        d / D_MAX,
        1.0,
    )

    a = min(
        n_available / N_REQ,
        1.0,
    )

    s = min(
        capacity / C_REQ,
        1.0,
    )

    ell = timeliness_score(
        latency
    )

    phi = min(
        b,
        a,
        s,
        ell,
    )

    return {
        "d": d,
        "n_available": n_available,
        "capacity": capacity,
        "latency": latency,
        "b": b,
        "a": a,
        "s": s,
        "ell": ell,
        "phi": phi,
    }


# ============================================================
# 3. Basic states
# ============================================================

NORMAL_STATE = state_from_raw(
    d=0.0,
    n_available=5,
    capacity=100.0,
    latency=0.20,
)

F_DAMAGED_STATE = state_from_raw(
    d=0.0,
    n_available=1,
    capacity=70.0,
    latency=0.20,
)

C_COMPROMISED_STATE = state_from_raw(
    d=0.65,
    n_available=5,
    capacity=80.0,
    latency=0.20,
)


# ============================================================
# 4. Topology-specific alternative paths
#
# G_B:
#   no redundancy/reconfiguration path
#
# G_D:
#   redundancy:
#       S2-G2-D2-A2-O1
#   reconfiguration:
#       S2-G2-D1-A1-O1
#
# G_E:
#   redundancy:
#       S2-G2-D2-A2-O1
#   reconfiguration:
#       S1-G2-D1-A1-O1
# ============================================================

PATH_STATE = {

    "G_B": {
        "R": None,
        "C": None,
    },

    "G_D": {

        "R": state_from_raw(
            d=0.0,
            n_available=5,
            capacity=85.0,
            latency=0.28,
        ),

        "C": state_from_raw(
            d=0.0,
            n_available=5,
            capacity=90.0,
            latency=0.40,
        ),
    },

    "G_E": {

        "R": state_from_raw(
            d=0.0,
            n_available=5,
            capacity=85.0,
            latency=0.28,
        ),

        "C": state_from_raw(
            d=0.0,
            n_available=5,
            capacity=95.0,
            latency=0.28,
        ),
    },
}


# ============================================================
# 5. Operational path-selection rule
# ============================================================

def select_alternative_path(
    topology,
    active_mechanisms,
    t,
):
    """
    Predefined operational selection rule.

    1. Only mechanisms that have completed can provide a path.
    2. Select the path with smaller latency.
    3. If latency is equal, select the path with higher capacity.

    The final E_Q is NOT used to choose a path.
    """

    active_mechanisms = set(
        active_mechanisms
    )

    candidates = []

    for mechanism in ("R", "C"):

        path = PATH_STATE[
            topology
        ][mechanism]

        if path is None:
            continue

        if mechanism not in active_mechanisms:
            continue

        if t < T_MECH[mechanism]:
            continue

        candidates.append(
            (
                mechanism,
                path,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            x[1]["latency"],
            -x[1]["capacity"],
            x[0],
        )
    )

    return candidates[0][1]


# ============================================================
# 6. Phi_exec(t)
# ============================================================

def phi_exec(
    t,
    scenario,
    active_mechanisms,
    topology,
):
    """
    Actual executed secure-chain state Phi_exec(t).
    """

    active_mechanisms = set(
        active_mechanisms
    )

    # Normal period
    if t < T_ATTACK_START:
        return NORMAL_STATE["phi"]

    # Autonomous recovery restores the primary chain
    if (
        "H" in active_mechanisms
        and t >= T_MECH["H"]
    ):
        return NORMAL_STATE["phi"]

    # --------------------------------------------------------
    # F scenario:
    # functional availability damage
    # --------------------------------------------------------

    if scenario == "F":

        alternative = (
            select_alternative_path(
                topology,
                active_mechanisms,
                t,
            )
        )

        if alternative is not None:
            return alternative["phi"]

        # No protection:
        # damaged state persists even after attack termination.
        return F_DAMAGED_STATE["phi"]

    # --------------------------------------------------------
    # C scenario:
    # compromised trust/integrity state
    # --------------------------------------------------------

    if scenario == "C":

        isolation_complete = (
            "I" in active_mechanisms
            and t >= T_MECH["I"]
        )

        # Before isolation, the compromised primary chain
        # remains the actual chain.
        if not isolation_complete:
            return C_COMPROMISED_STATE[
                "phi"
            ]

        # After isolation, only an established safe
        # alternative chain can provide service.
        alternative = (
            select_alternative_path(
                topology,
                active_mechanisms,
                t,
            )
        )

        if alternative is None:
            # No executable secure chain exists.
            return 0.0

        return alternative["phi"]

    raise ValueError(
        f"Unknown scenario: {scenario}"
    )


# ============================================================
# 7. Exact time integration
# ============================================================

def change_points(
    active_mechanisms,
):
    """
    All possible state-transition times.
    """

    points = {
        0.0,
        T_ATTACK_START,
        T_ATTACK_END,
        T,
    }

    for mechanism in active_mechanisms:

        t = T_MECH[mechanism]

        if 0 < t < T:
            points.add(t)

    return sorted(points)


def q_chain(
    scenario,
    active_mechanisms,
    topology,
):
    """
    Q_chain = 1/T int Phi_exec(t) dt

    Phi is piecewise constant, so exact interval
    integration is used.
    """

    points = change_points(
        active_mechanisms
    )

    area = 0.0

    for left, right in zip(
        points[:-1],
        points[1:],
    ):

        middle = (
            left + right
        ) / 2.0

        area += (
            phi_exec(
                middle,
                scenario,
                active_mechanisms,
                topology,
            )
            * (right - left)
        )

    return area / T


# ============================================================
# 8. Dynamic effectiveness E_Q
# ============================================================

def e_q(
    active_mechanisms,
    topology,
):

    active_mechanisms = set(
        active_mechanisms
    )

    value = 0.0

    for scenario in SCENARIOS:

        q_p = q_chain(
            scenario,
            active_mechanisms,
            topology,
        )

        q_u = q_chain(
            scenario,
            set(),
            topology,
        )

        value += (
            SCENARIO_WEIGHT[scenario]
            * (q_p - q_u)
        )

    return value


# ============================================================
# 9. R_Q and H_Q decomposition
# ============================================================

def r_q_h_q(
    active_mechanisms,
    topology,
):

    active_mechanisms = set(
        active_mechanisms
    )

    points = change_points(
        active_mechanisms
    )

    R_Q = 0.0
    H_Q = 0.0

    for scenario in SCENARIOS:

        benefit_area = 0.0
        harm_area = 0.0

        for left, right in zip(
            points[:-1],
            points[1:],
        ):

            middle = (
                left + right
            ) / 2.0

            phi_p = phi_exec(
                middle,
                scenario,
                active_mechanisms,
                topology,
            )

            phi_u = phi_exec(
                middle,
                scenario,
                set(),
                topology,
            )

            diff = phi_p - phi_u

            benefit_area += (
                max(diff, 0.0)
                * (right - left)
            )

            harm_area += (
                max(-diff, 0.0)
                * (right - left)
            )

        R_Q += (
            SCENARIO_WEIGHT[scenario]
            * benefit_area
            / T
        )

        H_Q += (
            SCENARIO_WEIGHT[scenario]
            * harm_area
            / T
        )

    return R_Q, H_Q


# ============================================================
# 10. Strict binary outcome
# ============================================================

def safe_alternative_before_deadline(
    active_mechanisms,
    topology,
):

    active_mechanisms = set(
        active_mechanisms
    )

    for mechanism in ("R", "C"):

        if mechanism not in active_mechanisms:
            continue

        if PATH_STATE[
            topology
        ][mechanism] is None:
            continue

        if (
            T_MECH[mechanism]
            <= T_STRICT_DEADLINE
        ):
            return True

    return False


def strict_y(
    scenario,
    active_mechanisms,
    topology,
):

    active_mechanisms = set(
        active_mechanisms
    )

    alternative_ok = (
        safe_alternative_before_deadline(
            active_mechanisms,
            topology,
        )
    )

    if scenario == "F":

        return int(
            alternative_ok
        )

    if scenario == "C":

        isolation_ok = (
            "I" in active_mechanisms
            and T_MECH["I"]
            <= T_STRICT_DEADLINE
        )

        return int(
            isolation_ok
            and alternative_ok
        )

    raise ValueError(
        scenario
    )


def tau_y(
    active_mechanisms,
    topology,
):
    """
    Baseline Y_U = 0 in both F and C.
    """

    return sum(

        SCENARIO_WEIGHT[scenario]
        * strict_y(
            scenario,
            active_mechanisms,
            topology,
        )

        for scenario in SCENARIOS
    )


# ============================================================
# 11. 16 configurations
# ============================================================

CONFIGS = (

    (),

    ("R",),
    ("C",),
    ("I",),
    ("H",),

    ("R", "C"),
    ("R", "I"),
    ("R", "H"),
    ("C", "I"),
    ("C", "H"),
    ("I", "H"),

    ("R", "C", "I"),
    ("R", "C", "H"),
    ("R", "I", "H"),
    ("C", "I", "H"),

    ("R", "C", "I", "H"),
)


def config_name(
    config,
):

    if not config:
        return "∅"

    return "".join(
        m for m in MECHS if m in config
    )


def main_result_table():

    rows = []

    for config in CONFIGS:

        row = {
            "configuration":
                config_name(config)
        }

        for topology in TOPOLOGIES:

            R_Q, H_Q = r_q_h_q(
                config,
                topology,
            )

            row[
                f"{topology}_QF"
            ] = q_chain(
                "F",
                config,
                topology,
            )

            row[
                f"{topology}_QC"
            ] = q_chain(
                "C",
                config,
                topology,
            )

            row[
                f"{topology}_RQ"
            ] = R_Q

            row[
                f"{topology}_HQ"
            ] = H_Q

            row[
                f"{topology}_EQ"
            ] = e_q(
                config,
                topology,
            )

            row[
                f"{topology}_tauY"
            ] = tau_y(
                config,
                topology,
            )

        rows.append(row)

    return pd.DataFrame(
        rows
    )


# ============================================================
# 12. Shapley contribution
# ============================================================

def shapley(
    topology,
):

    n = len(
        MECHS
    )

    result = {}

    for mechanism in MECHS:

        others = [
            x
            for x in MECHS
            if x != mechanism
        ]

        contribution = 0.0

        for r in range(
            len(others) + 1
        ):

            for subset_tuple in (
                itertools.combinations(
                    others,
                    r,
                )
            ):

                subset = set(
                    subset_tuple
                )

                weight = (
                    math.factorial(r)
                    * math.factorial(
                        n - r - 1
                    )
                    / math.factorial(n)
                )

                marginal = (

                    e_q(
                        subset
                        | {mechanism},
                        topology,
                    )

                    -

                    e_q(
                        subset,
                        topology,
                    )
                )

                contribution += (
                    weight
                    * marginal
                )

        result[
            mechanism
        ] = contribution

    return result


# ============================================================
# 13. Pairwise interaction
# ============================================================

def interaction(
    topology,
    m1,
    m2,
):

    return (

        e_q(
            {m1, m2},
            topology,
        )

        - e_q(
            {m1},
            topology,
        )

        - e_q(
            {m2},
            topology,
        )

        + e_q(
            set(),
            topology,
        )
    )


# ============================================================
# 14. Reliability-parameter uncertainty
# ============================================================

ALL_SUBSETS = [

    {
        MECHS[j]
        for j in range(
            len(MECHS)
        )
        if (
            mask >> j
        ) & 1
    }

    for mask in range(
        2 ** len(MECHS)
    )
]


# Deterministic analytical truth
DET_EQ = {

    topology: {

        frozenset(subset):
            e_q(
                subset,
                topology,
            )

        for subset
        in ALL_SUBSETS
    }

    for topology
    in TOPOLOGIES
}


def expected_e_q_given_p(
    configured,
    topology,
    p,
):
    """
    Conditional expectation over all realized
    mechanism-success subsets.

    No finite-N experiment is used here.
    """

    configured = list(
        set(configured)
    )

    total = 0.0

    for mask in range(
        1 << len(configured)
    ):

        realized = set()

        probability = 1.0

        for j, mechanism in enumerate(
            configured
        ):

            if (
                mask >> j
            ) & 1:

                realized.add(
                    mechanism
                )

                probability *= (
                    p[mechanism]
                )

            else:

                probability *= (
                    1.0
                    - p[mechanism]
                )

        total += (

            probability

            * DET_EQ[
                topology
            ][
                frozenset(realized)
            ]
        )

    return total


def uncertainty_analysis(
    n_draws=20000,
    seed=20260914,
):

    rng = (
        np.random.default_rng(
            seed
        )
    )

    rows = []

    for p_bar in (
        0.70,
        0.85,
        0.95,
    ):

        for kappa in (
            50,
            100,
            200,
        ):

            alpha = (
                p_bar
                * kappa
            )

            beta = (
                (1.0 - p_bar)
                * kappa
            )

            samples = rng.beta(
                alpha,
                beta,
                size=(
                    n_draws,
                    len(MECHS),
                ),
            )

            values = {
                topology:
                    np.empty(
                        n_draws
                    )
                for topology
                in TOPOLOGIES
            }

            for i, sample in enumerate(
                samples
            ):

                p = dict(
                    zip(
                        MECHS,
                        sample,
                    )
                )

                for topology in TOPOLOGIES:

                    values[
                        topology
                    ][i] = (
                        expected_e_q_given_p(
                            MECHS,
                            topology,
                            p,
                        )
                    )

            row = {
                "p_bar": p_bar,
                "kappa": kappa,
            }

            for topology in TOPOLOGIES:

                v = values[
                    topology
                ]

                row[
                    f"{topology}_mean"
                ] = np.mean(v)

                row[
                    f"{topology}_q025"
                ] = np.quantile(
                    v,
                    0.025,
                )

                row[
                    f"{topology}_q975"
                ] = np.quantile(
                    v,
                    0.975,
                )

            rows.append(row)

    return pd.DataFrame(
        rows
    )


# ============================================================
# 15. Mathematical self-checks
# ============================================================

def self_checks():

    tol = 1e-12

    # ---------- Table 1 ----------

    assert abs(
        NORMAL_STATE["phi"]
        - 1.00
    ) < tol

    assert abs(
        F_DAMAGED_STATE["phi"]
        - 0.20
    ) < tol

    assert abs(
        C_COMPROMISED_STATE["phi"]
        - 0.35
    ) < tol

    assert abs(
        PATH_STATE[
            "G_D"
        ]["R"]["phi"]
        - 0.85
    ) < tol

    assert abs(
        PATH_STATE[
            "G_D"
        ]["C"]["phi"]
        - 0.75
    ) < tol

    assert abs(
        PATH_STATE[
            "G_E"
        ]["R"]["phi"]
        - 0.85
    ) < tol

    assert abs(
        PATH_STATE[
            "G_E"
        ]["C"]["phi"]
        - 0.90
    ) < tol

    # ---------- Baseline Q ----------

    for topology in TOPOLOGIES:

        assert abs(
            q_chain(
                "F",
                set(),
                topology,
            )
            - 0.36
        ) < tol

        assert abs(
            q_chain(
                "C",
                set(),
                topology,
            )
            - 0.48
        ) < tol

    # ---------- Bounds and identity ----------

    for topology in TOPOLOGIES:

        for config in CONFIGS:

            for scenario in SCENARIOS:

                q = q_chain(
                    scenario,
                    config,
                    topology,
                )

                assert (
                    -tol
                    <= q
                    <= 1.0 + tol
                )

            R_Q, H_Q = r_q_h_q(
                config,
                topology,
            )

            assert abs(
                e_q(
                    config,
                    topology,
                )
                -
                (
                    R_Q
                    - H_Q
                )
            ) < tol

    # ---------- Shapley efficiency ----------

    for topology in TOPOLOGIES:

        sv = shapley(
            topology
        )

        assert abs(
            sum(
                sv.values()
            )
            -
            e_q(
                MECHS,
                topology,
            )
        ) < tol

    # ---------- Frozen main values ----------

    assert abs(
        e_q(
            MECHS,
            "G_B",
        )
        - 0.09375
    ) < tol

    assert abs(
        e_q(
            MECHS,
            "G_D",
        )
        - 0.43875
    ) < tol

    assert abs(
        e_q(
            MECHS,
            "G_E",
        )
        - 0.45925
    ) < tol

    print(
        "All mathematical self-checks passed."
    )


# ============================================================
# 16. Run
# ============================================================

if __name__ == "__main__":

    self_checks()

    main_df = (
        main_result_table()
    )

    print(
        "\n=== Main results ==="
    )

    print(
        main_df
        .round(5)
        .to_string(
            index=False
        )
    )

    print(
        "\n=== Shapley ==="
    )

    for topology in TOPOLOGIES:

        values = shapley(
            topology
        )

        print(
            topology,
            {
                k: round(v, 6)
                for k, v
                in values.items()
            },
            "sum =",
            round(
                sum(
                    values.values()
                ),
                6,
            ),
        )

    print(
        "\n=== G_E interactions ==="
    )

    for pair in (
        ("R", "I"),
        ("C", "I"),
        ("R", "C"),
    ):

        print(
            pair,
            round(
                interaction(
                    "G_E",
                    *pair,
                ),
                6,
            ),
        )

    uncertainty_df = (
        uncertainty_analysis()
    )

    print(
        "\n=== Parameter uncertainty ==="
    )

    print(
        uncertainty_df
        .round(6)
        .to_string(
            index=False
        )
    )

    main_df.to_csv(
        "theoretical_case_main.csv",
        index=False,
    )

    uncertainty_df.to_csv(
        "theoretical_case_uncertainty.csv",
        index=False,
    )
