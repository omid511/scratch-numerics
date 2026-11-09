#!/usr/bin/env python3
"""Check a frozen kinematic contrast and its conditional algebraic bounds."""
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
from struct import pack

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.json"
RESULTS = HERE / "results.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if RESULTS.exists():
        raise FileExistsError(f"Refusing to overwrite {RESULTS}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    grid = protocol["grid"]
    hz = grid["samples_per_second"]
    end, tc, lead, branch = (F(grid[k]) for k in
                             ("end_seconds", "crossing_seconds", "lead_seconds", "branch_seconds"))
    assert (hz, end, tc, lead, branch) == (1000, F(156, 5), 24, 1, 23)
    n, crossing_index, branch_index, lead_samples = (int(x * hz) for x in (end, tc, branch, lead))
    assert crossing_index - branch_index == lead_samples == 1000
    assert F(crossing_index, hz) - F(branch_index, hz) == lead

    shared = [F(9, 10) + F(i, hz * 240) for i in range(n + 1)]
    rho_b = shared[branch_index]
    benign = shared[:branch_index + 1]
    for i in range(branch_index + 1, n + 1):
        s = F(i - branch_index, hz)
        benign.append(rho_b + ((s - s * s / 2) / 240 if s <= 1 else F(1, 480)))

    # Compare the actual binary64 representations, not a numerical tolerance.
    prefix_crossing = b"".join(pack("=d", float(v)) for v in shared[:branch_index + 1])
    prefix_benign = b"".join(pack("=d", float(v)) for v in benign[:branch_index + 1])
    assert prefix_crossing == prefix_benign
    assert shared[crossing_index] == 1
    assert all(x < 1 for x in shared[:crossing_index])
    assert next(i for i, x in enumerate(shared) if float(x) >= 1.0) == crossing_index
    assert all(x < 1 and float(x) < 1.0 for x in benign)
    assert any(shared[i] != benign[i] for i in range(branch_index + 1, n + 1))
    # Derivatives of the linear, quadratic and constant pieces at both joins.
    linear_rate = F(1, 240)
    branch_rate = lambda s: (1 - s) / 240
    assert linear_rate == branch_rate(F(0))
    assert branch_rate(F(1)) == 0
    assert rho_b + (F(1) - F(1, 2)) / 240 == rho_b + F(1, 480)
    assert min(shared + benign) == F(9, 10)
    assert max(shared + benign) == F(103, 100)
    assert 0 <= branch_rate(F(1)) <= branch_rate(F(0)) == linear_rate
    assert all(0 <= (series[i + 1] - series[i]) * hz <= linear_rate
               for series in (shared, benign) for i in range(n))

    target = F(protocol["timely_success_target"])
    limit = F(protocol["nuisance_episode_limit_per_hour"])
    assert (target, limit) == (F(9, 10), 1)
    rows = []
    for alpha_text in protocol["exploratory_alpha"]:
        alpha = F(alpha_text)
        q = max(F(0), 1 - (1 - target) / alpha)
        for lambda_text in protocol["exploratory_benign_rearmed_opportunities_per_hour"]:
            rate = F(lambda_text)
            bound = rate * q
            rows.append({"alpha": float(alpha), "lambda_b_per_hour": float(rate),
                         "q_min": float(q), "nuisance_lower_bound_per_hour": float(bound),
                         "necessary_compatibility": bound <= limit})
    assert max(F(0), 1 - (1 - target) / F(1, 10)) == 0
    assert max(F(0), 1 - (1 - target) / F(1)) == F(9, 10)
    result = {
        "protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__).resolve()),
        "status": "all self-checks passed",
        "checks": {
            "sample_count": n + 1, "branch_index": branch_index,
            "crossing_index": crossing_index, "lead_samples": lead_samples,
            "prefix_float64_byte_identity_through_branch": True,
            "first_crossing_sample_seconds_exact": str(F(crossing_index, hz)),
            "crossing_value_exact": str(shared[crossing_index]),
            "benign_max_exact": str(max(benign)),
            "futures_differ_after_branch": True,
            "first_derivative_at_branch_left_and_right_exact": str(linear_rate),
            "first_derivative_at_branch_end_left_and_right_exact": "0",
            "rate_bound_per_second_exact": str(linear_rate),
            "ratio_min_exact": "9/10", "ratio_max_exact": "103/100",
            "q_min_at_alpha_0_1": 0.0, "q_min_at_alpha_1": 0.9
        },
        "assumptions": protocol["assumptions"], "derivation": protocol["derivation"],
        "limitations": protocol["limitations"], "exploratory_grid": rows
    }
    with RESULTS.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"PASS: {RESULTS}")
    print(f"protocol_sha256={result['protocol_sha256']}")
    print(f"script_sha256={result['script_sha256']}")
    print(f"results_sha256={sha256(RESULTS)}")


if __name__ == "__main__":
    main()
