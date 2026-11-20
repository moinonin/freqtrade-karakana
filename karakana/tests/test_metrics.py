import numpy as np

from karakana.metrics import (
    calculate_moment_sgo_metrics,
    compute_sgo_matrix,
    evaluate_structural_geometry,
)


def test_compute_sgo_matrix_matches_theoretical_outer_product():
    matrix = compute_sgo_matrix(
        [
            [2.0, -1.0],
            [0.0, -3.0],
        ],
        utility_power=1.0,
    )

    expected = np.array(
        [
            [1.0, 1.0, 2.0],
            [1.0, 2.0, 1.0],
            [2.0, 1.0, 5.0],
        ]
    )

    np.testing.assert_allclose(matrix, expected)


def test_compute_sgo_matrix_defaults_to_utility_power_1_1():
    matrix = compute_sgo_matrix([[4.0, -9.0]])

    expected_p = 4.0**1.1
    expected_n = 9.0**1.1

    np.testing.assert_allclose(matrix[0, 1], expected_p)
    np.testing.assert_allclose(matrix[0, 2], expected_n)
    np.testing.assert_allclose(matrix[1, 1], expected_p * expected_p)
    np.testing.assert_allclose(matrix[1, 2], expected_p * expected_n)


def test_compute_sgo_matrix_is_exported_from_package_root():
    import karakana

    matrix = karakana.compute_sgo_matrix([[1.0, -1.0]])

    assert matrix.shape == (3, 3)


def test_moment_sgo_distinguishes_coupled_from_decoupled_units():
    coupled = calculate_moment_sgo_metrics(
        [
            [10.0, -10.0],
            [10.0, -10.0],
            [10.0, -10.0],
            [10.0, -10.0],
        ],
        utility_power=1.0,
    )
    decoupled = calculate_moment_sgo_metrics(
        [
            [10.0, 0.0],
            [10.0, 0.0],
            [0.0, -10.0],
            [0.0, -10.0],
        ],
        utility_power=1.0,
    )

    assert coupled["e_pn"] == 100.0
    assert decoupled["e_pn"] == 0.0
    assert coupled["rho_pn_sq"] > decoupled["rho_pn_sq"]
    assert decoupled["rho_pn_sq"] == 0.0


def test_moment_sgo_ideal_preserves_marginals_and_zeros_cross_moment():
    metrics = calculate_moment_sgo_metrics(
        [
            [3.0, -2.0],
            [4.0, 0.0],
        ]
    )
    matrix = np.asarray(metrics["moment_matrix"])
    ideal = np.asarray(metrics["moment_ideal_matrix"])

    np.testing.assert_allclose(np.diag(ideal), np.diag(matrix))
    assert ideal[0, 1] == matrix[0, 1]
    assert ideal[0, 2] == matrix[0, 2]
    assert ideal[1, 2] == 0.0
    assert ideal[2, 1] == 0.0


def test_alpha_s_sil_is_zero_when_negative_ideal_alpha_equals_s_observed():
    # In the legacy n_matrix ideal, ideal_alpha is -avg_win. If observed losses
    # are zero, s_observed is +avg_win and the opposite-sign alignment loss is 0.
    report = evaluate_structural_geometry(
        np.array([2.0, 2.0]),
        np.array([0.0]),
        utility_power=1.0,
    )

    assert report["ideal_cofactors"][1][2] < 0.0
    assert -report["ideal_cofactors"][1][2] == report["n_matrix"][0][2]
    assert report["alpha_s_sil"] < 1e-8


def test_alpha_s_mse_alias_matches_alpha_s_sil():
    report = evaluate_structural_geometry(
        np.array([10.0, 12.0, 14.0]),
        np.array([-3.0, -4.0]),
    )

    assert report["alpha_s_mse"] == report["alpha_s_sil"]


def test_evaluate_empty_outcomes_raises_clear_error():
    try:
        evaluate_structural_geometry(np.array([]), np.array([]))
    except ValueError as exc:
        assert "0 outcomes" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty outcomes")


def test_positive_alpha_weight_penalizes_positive_alpha_only():
    base = evaluate_structural_geometry(
        np.array([1.0]),
        np.array([-10.0, -12.0, -14.0]),
        positive_alpha_weight=0.0,
    )
    penalized = evaluate_structural_geometry(
        np.array([1.0]),
        np.array([-10.0, -12.0, -14.0]),
        positive_alpha_weight=2.0,
    )

    assert base["structural_alpha"] > 0.0
    assert penalized["positive_alpha_penalty"] > 0.0
    assert penalized["ranking_score"] > base["ranking_score"]


def test_positive_alpha_weight_does_not_penalize_non_positive_alpha():
    base = evaluate_structural_geometry(
        np.array([10.0, 12.0, 14.0]),
        np.array([-3.0, -4.0]),
        positive_alpha_weight=0.0,
    )
    penalized = evaluate_structural_geometry(
        np.array([10.0, 12.0, 14.0]),
        np.array([-3.0, -4.0]),
        positive_alpha_weight=2.0,
    )
    # With a non-positive structural alpha the ranking penalty is inert: the
    # geometry metrics must be identical regardless of the penalty weight.
    assert base["positive_alpha_penalty"] == penalized["positive_alpha_penalty"] == 0.0
    assert base["structural_alpha"] == penalized["structural_alpha"]
    assert base["normalized_structure_term"] == penalized["normalized_structure_term"]
    assert base["ranking_score"] == penalized["ranking_score"]


def test_positive_alpha_weight_from_profile():
    from karakana.config import get_config
    from karakana.metrics import RANKING_PROFILES

    # Setup outcomes that produce a positive alpha
    # [1.0] vs [-10, -12, -14] gives positive alpha
    pos = np.array([1.0])
    neg = np.array([-10.0, -12.0, -14.0])

    # default (None) should use the configured global ranking profile
    report_default = evaluate_structural_geometry(pos, neg)
    default_profile = get_config.get("ranking_profile", "conservative")
    assert report_default["positive_alpha_weight"] == RANKING_PROFILES[default_profile]["positive_alpha_weight"]

    # aggressive should use 0.13
    report_aggressive = evaluate_structural_geometry(pos, neg, ranking_profile="aggressive")
    assert report_aggressive["positive_alpha_weight"] == 0.13

    # conservative should use 0.07
    report_conservative = evaluate_structural_geometry(pos, neg, ranking_profile="conservative")
    assert report_conservative["positive_alpha_weight"] == 0.07

    # explicit override should still work
    report_override = evaluate_structural_geometry(pos, neg, positive_alpha_weight=0.5)
    assert report_override["positive_alpha_weight"] == 0.5
