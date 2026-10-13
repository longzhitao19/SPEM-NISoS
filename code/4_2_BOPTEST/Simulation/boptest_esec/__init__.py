"""BOPTEST implementation of paired endogenous-resilience E_sec experiments."""

# v0.8.11 keeps the v0.8.10 attack-validity fixes and adds explicit protection
# action transients.  A path selected in ``finish`` cannot receive execution
# credit until it has actually been used by the next ``prepare`` call, and
# reconfiguration/recovery handovers may briefly reduce execution availability.
# The version bump intentionally invalidates older checkpoints.
VERSION = "boptest-esec-0.8.11-protection-transient-harm"
