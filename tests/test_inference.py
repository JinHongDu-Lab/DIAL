import numpy as np
import pytest

from dial_judge.baselines import fit_human_only_btl, fit_staged_structured_calibration
from dial_judge.benchmarks import fit_dial
from dial_judge.data import comparisons_to_aggregated, comparisons_to_order_aggregated, pool_pairs
from dial_judge.gacv import make_centering_basis, pack_reduced
from dial_judge.inference import (
    bias_projection_operator,
    bias_sensitivity,
    calibration_restriction_test,
    comparison_laplacian,
    human_only_uq,
    joint_sandwich,
    llm_score_covariance,
    misalignment_widened_contrasts,
    noncentrality_interval,
    pairwise_contrast_matrix,
    population_excess_risk,
)
from dial_judge.simulate import (
    compute_human_score,
    compute_score_matrix,
    generate_study_calibration,
    generate_study_parameters,
    generate_study_position_effects,
    generate_random_human_comparisons,
    generate_random_llm_comparisons,
)


@pytest.fixture(scope="module")
def small_problem():
    N, K, r = 8, 4, 1
    seed = 3
    mu, gamma, U, V = generate_study_parameters(N, K, r, random_seed=seed)
    b = generate_study_position_effects(K, random_seed=seed + 1)
    c_mu, c_v = generate_study_calibration(V, random_seed=seed + 2)
    S = compute_score_matrix(mu, gamma, U, V)
    s0 = compute_human_score(mu, V, c_mu, c_v)
    llm = generate_random_llm_comparisons(S, b, 4000, random_seed=seed + 3)
    hum = generate_random_human_comparisons(s0, 400, random_seed=seed + 4)
    n_ijk, y_ijk = comparisons_to_aggregated(llm, N, K)
    n_order, y_order = comparisons_to_order_aggregated(llm, N, K)
    return dict(N=N, K=K, r=r, mu=mu, gamma=gamma, U=U, V=V, b=b, S=S, s0=s0, pairs=pool_pairs(hum), n_ijk=n_ijk, y_ijk=y_ijk, n_order=n_order, y_order=y_order)


def test_study_parameters_satisfy_constraints():
    N, K, r = 12, 5, 2
    mu, gamma, U, V = generate_study_parameters(N, K, r, random_seed=0)
    assert abs(mu.sum()) < 1e-10 and abs(mu @ mu - N) < 1e-8
    assert abs(gamma.sum() - K) < 1e-10
    assert np.allclose(V.T @ np.ones(N), 0) and np.allclose(mu @ V, 0)
    assert np.allclose(V.T @ V / N, np.eye(r))
    assert np.allclose(U.sum(axis=0), 0)


def test_random_llm_sampler_swap_fraction():
    S = np.zeros((2, 5))
    b = np.zeros(2)
    recs = generate_random_llm_comparisons(S, b, 5000, random_seed=1, swap_fraction=0.2)
    a = np.array([rec[4] for rec in recs])
    assert abs(np.mean(a == -1) - 0.2) < 0.03
    assert all(rec[1] < rec[2] for rec in recs)
    recs0 = generate_random_llm_comparisons(S, b, 100, random_seed=1, swap_fraction=0.0)
    assert all(rec[4] == 1 for rec in recs0)


def test_excess_risk_nonnegative_and_zero_at_truth(small_problem):
    s0 = small_problem["s0"]
    assert population_excess_risk(s0, s0) == pytest.approx(0.0, abs=1e-12)
    rng = np.random.default_rng(0)
    for _ in range(5):
        s = s0 + rng.normal(size=s0.size)
        assert population_excess_risk(s, s0) > 0


def test_human_only_uq_covariance_is_centered_and_psd(small_problem):
    N, pairs = small_problem["N"], small_problem["pairs"]
    h = fit_human_only_btl(N, pairs)
    uq = human_only_uq(N, pairs, h["s_H"])
    cov = uq["covariance"]
    assert np.allclose(cov @ np.ones(N), 0, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(cov) > -1e-10)
    assert np.all(uq["contrasts"]["se"] > 0)


def test_llm_score_covariance_matches_fisher_at_truth(small_problem):
    """Under the true parameters the centered score covariance equals the Fisher information."""
    p = small_problem
    N, K, r = p["N"], p["K"], p["r"]
    jb, ib = make_centering_basis(K), make_centering_basis(N)
    zeta = pack_reduced(p["gamma"], p["mu"], p["U"], p["V"], p["b"], 1.0, np.zeros(r), True)
    VL, n_L = llm_score_covariance(zeta, N, K, r, jb, ib, True, p["n_order"], p["y_order"], p["n_ijk"], p["y_ijk"])
    assert np.allclose(VL, VL.T)
    assert np.all(np.linalg.eigvalsh(VL) > -1e-10)
    # compare with expected information sum_c (n_c/n_L) p(1-p) x x^T via a large-sample surrogate:
    # replace y_c by its expectation n_c * p_c -> second moment equals p(1-p) n_c and g_bar = 0.
    from dial_judge.inference import _llm_cell_design
    from scipy.special import expit

    X, n_c, _, eta = _llm_cell_design(zeta, N, K, r, jb, ib, True, p["n_order"], p["y_order"], p["n_ijk"], p["y_ijk"])
    pc = expit(eta)
    fisher = (X * (n_c * pc * (1 - pc))[:, None]).T @ X / n_L
    rel = np.linalg.norm(VL - fisher) / np.linalg.norm(fisher)
    assert rel < 0.15  # sampling noise in y_c at n_L = 4000


def test_joint_sandwich_shapes_and_tau0_is_narrower(small_problem):
    p = small_problem
    N, K, r = p["N"], p["K"], p["r"]
    st = fit_staged_structured_calibration(N, K, r, p["n_ijk"], p["y_ijk"], p["pairs"], n_order=p["n_order"], y_order=p["y_order"])
    jt = fit_dial(N, K, r, p["n_ijk"], p["y_ijk"], p["pairs"], n_order=p["n_order"], y_order=p["y_order"], init_params=(st["gamma"], st["mu"], st["U"], st["V"], st["b"]))
    sw = joint_sandwich(jt, N, K, r, p["pairs"], p["n_ijk"], p["y_ijk"], p["n_order"], p["y_order"])
    sw0 = joint_sandwich(jt, N, K, r, p["pairs"], p["n_ijk"], p["y_ijk"], p["n_order"], p["y_order"], llm_variation=False)
    assert sw["cov_s"].shape == (N, N) and sw["cov_S"].shape == (K * N, K * N) and sw["cov_b"].shape == (K, K)
    assert np.allclose(sw["cov_s"] @ np.ones(N), 0, atol=1e-6)
    assert np.all(sw["contrasts"]["se"] > 0) and np.all(sw["b"]["se"] > 0)
    # dropping LLM sampling variation can only shrink the covariance
    assert np.all(sw0["contrasts"]["se"] <= sw["contrasts"]["se"] + 1e-12)
    assert np.all(sw0["b"]["se"] <= sw["b"]["se"] + 1e-12)


def test_joint_warm_start_not_worse_than_staged_objective(small_problem):
    p = small_problem
    N, K, r = p["N"], p["K"], p["r"]
    from dial_judge.dial_model import human_nll_and_grad, pairs_to_arrays
    from dial_judge.hja import negative_log_likelihood

    st = fit_staged_structured_calibration(N, K, r, p["n_ijk"], p["y_ijk"], p["pairs"], n_order=p["n_order"], y_order=p["y_order"], tol=1e-7, max_steps=400)
    jt = fit_dial(N, K, r, p["n_ijk"], p["y_ijk"], p["pairs"], n_order=p["n_order"], y_order=p["y_order"], init_params=(st["gamma"], st["mu"], st["U"], st["V"], st["b"]), tol=1e-8, max_steps=300)
    n_L, n_H = float(p["n_order"].sum()), float(sum(q[2] for q in p["pairs"]))
    lam = n_L / n_H
    pa = pairs_to_arrays(p["pairs"])

    def Q(f):
        ll = negative_log_likelihood(f["mu"], f["gamma"], f["U"], f["V"], p["n_ijk"], p["y_ijk"], b=f["b"], n_order=p["n_order"], y_order=p["y_order"])
        lh = human_nll_and_grad(f["alpha_H"], f["a"], f["mu"], f["V"], pa)[0]
        return lh / n_H + lam * ll / n_L

    assert Q(jt) <= Q(st) + 1e-9
    assert jt["fit_info"]["converged"]


def test_pairwise_contrast_matrix():
    D = pairwise_contrast_matrix(4)
    assert D.shape == (6, 4) and np.allclose(D.sum(axis=1), 0)


@pytest.fixture(scope="module")
def misaligned_problem():
    """A larger, genuinely misaligned problem (c_v_sd=0.5, the misaligned simulation config) on
    which GACV reliably selects an interior lambda > 0 for align='mu', unlike small_problem's
    scale where it can select the lam=0 endpoint."""
    N, K, r = 10, 4, 1
    seed = 0
    mu, gamma, U, V = generate_study_parameters(N, K, r, random_seed=seed)
    b = generate_study_position_effects(K, random_seed=seed + 1)
    c_mu, c_v = generate_study_calibration(V, c_v_sd=0.5, random_seed=seed + 2)
    S = compute_score_matrix(mu, gamma, U, V)
    s0 = compute_human_score(mu, V, c_mu, c_v)
    llm = generate_random_llm_comparisons(S, b, 20000, random_seed=seed + 3, swap_fraction=0.25)
    hum = generate_random_human_comparisons(s0, 800, random_seed=seed + 4)
    n_ijk, y_ijk = comparisons_to_aggregated(llm, N, K)
    n_order, y_order = comparisons_to_order_aggregated(llm, N, K)
    return dict(N=N, K=K, r=r, s0=s0, pairs=pool_pairs(hum), n_ijk=n_ijk, y_ijk=y_ijk, n_order=n_order, y_order=y_order)


def test_comparison_laplacian_matches_human_only_info(small_problem):
    N, pairs = small_problem["N"], small_problem["pairs"]
    h = fit_human_only_btl(N, pairs)
    uq = human_only_uq(N, pairs, h["s_H"])
    L = comparison_laplacian(N, pairs, h["s_H"], normalize=False)
    n_H = sum(q[2] for q in pairs)
    B = make_centering_basis(N)
    cov_from_L = B @ np.linalg.pinv(B.T @ L @ B, rcond=1e-10) @ B.T
    assert np.allclose(cov_from_L, uq["covariance"])
    L_norm = comparison_laplacian(N, pairs, h["s_H"], normalize=True)
    assert np.allclose(L_norm * n_H, L)
    assert np.allclose(L.sum(axis=1), 0, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(L) > -1e-8)


def test_bias_projection_operator_fixes_col_W_and_eigenvalues_bounded(misaligned_problem):
    """Local misalignment: A_lambda h = h exactly for h in col(W) at every lambda, and its
    eigenvalues (w.r.t. the L inner product) lie in [0, 1]; both checked with finite-sample
    slack since A_lambda is a plug-in built from a fitted joint DIAL model."""
    p = misaligned_problem
    N, K, r = p["N"], p["K"], p["r"]
    from dial_judge.baselines import fit_consensus_only_calibrated
    from dial_judge.gacv import select_lambda

    st = fit_consensus_only_calibrated(N, K, r, p["n_ijk"], p["y_ijk"], p["pairs"], n_order=p["n_order"], y_order=p["y_order"])
    sel = select_lambda(N, K, r, p["pairs"], n_ijk_llm=p["n_ijk"], y_ijk_llm=p["y_ijk"], n_order=p["n_order"], y_order=p["y_order"],
                         staged_fit=st, return_all=True, align="mu")
    assert sel["lam"] > 0
    sw = joint_sandwich(sel, N, K, r, p["pairs"], p["n_ijk"], p["y_ijk"], p["n_order"], p["y_order"], cluster="pair")
    mu_hat = sel["mu"]
    L = comparison_laplacian(N, p["pairs"], mu_hat, normalize=True)
    A = bias_projection_operator(sw["dot_s0"], sw["hessian"], L)
    resid = A @ mu_hat - mu_hat
    assert np.max(np.abs(resid)) < 0.25 * np.max(np.abs(mu_hat))
    w, Vv = np.linalg.eigh(L)
    w = np.maximum(w, 0.0)
    Lh = Vv @ np.diag(np.sqrt(w)) @ Vv.T
    w_safe = np.where(w > 1e-9, w, 1.0)
    Lph = Vv @ np.diag(np.where(w > 1e-9, 1.0 / np.sqrt(w_safe), 0.0)) @ Vv.T
    M = Lh @ A @ Lph
    ev = np.linalg.eigvalsh((M + M.T) / 2.0)
    assert ev.min() > -0.1 and ev.max() < 1.2


def test_misalignment_widened_contrasts_widens_and_improves_coverage(misaligned_problem):
    """The interval-widening fix of "Interval validity under local misalignment" only widens
    (never narrows) the plain sandwich intervals, and does not hurt coverage against the true
    contrasts on a problem with a genuine misalignment (small_problem's default c_v_sd=0.5)."""
    p = misaligned_problem
    N, K, r = p["N"], p["K"], p["r"]
    from dial_judge.baselines import fit_consensus_only_calibrated
    from dial_judge.gacv import select_lambda

    st = fit_consensus_only_calibrated(N, K, r, p["n_ijk"], p["y_ijk"], p["pairs"], n_order=p["n_order"], y_order=p["y_order"])
    sel = select_lambda(N, K, r, p["pairs"], n_ijk_llm=p["n_ijk"], y_ijk_llm=p["y_ijk"], n_order=p["n_order"], y_order=p["y_order"],
                         staged_fit=st, return_all=True, align="mu")
    sw = joint_sandwich(sel, N, K, r, p["pairs"], p["n_ijk"], p["y_ijk"], p["n_order"], p["y_order"], cluster="pair")
    W = sel["mu"].reshape(-1, 1)
    ct = calibration_restriction_test(N, p["pairs"], W)
    _, delta_hi_raw = noncentrality_interval(ct["stat"], ct["df"], level=0.95)
    delta_hi = delta_hi_raw / (2.0 * sw["n_H"])
    wc = misalignment_widened_contrasts(sw, N, p["pairs"], ct["s_full"], delta_hi)
    assert np.all(wc["kappa"] >= -1e-9)
    assert np.all(wc["lower_widened"] <= sw["contrasts"]["lower"] + 1e-9)
    assert np.all(wc["upper_widened"] >= sw["contrasts"]["upper"] - 1e-9)
    D = pairwise_contrast_matrix(N)
    tc = D @ p["s0"]
    cov_raw = np.mean((tc >= sw["contrasts"]["lower"]) & (tc <= sw["contrasts"]["upper"]))
    cov_widened = np.mean((tc >= wc["lower_widened"]) & (tc <= wc["upper_widened"]))
    assert cov_widened >= cov_raw - 1e-9


def test_noncentrality_interval_endpoints():
    lo, hi = noncentrality_interval(2.0, 8, level=0.90)
    assert lo == 0.0 and hi >= 0.0
    lo2, hi2 = noncentrality_interval(50.0, 8, level=0.90)
    assert lo2 > 0.0 and hi2 > lo2


def test_cluster_robust_llm_covariance_psd_and_close_to_iid_without_clustering(small_problem):
    p = small_problem
    N, K, r = p["N"], p["K"], p["r"]
    jb, ib = make_centering_basis(K), make_centering_basis(N)
    zeta = pack_reduced(p["gamma"], p["mu"], p["U"], p["V"], p["b"], 1.0, np.zeros(r), True)
    V_iid, _ = llm_score_covariance(zeta, N, K, r, jb, ib, True, p["n_order"], p["y_order"], p["n_ijk"], p["y_ijk"])
    V_cl, _ = llm_score_covariance(zeta, N, K, r, jb, ib, True, p["n_order"], p["y_order"], p["n_ijk"], p["y_ijk"], cluster="pair")
    assert np.allclose(V_cl, V_cl.T) and np.all(np.linalg.eigvalsh(V_cl) > -1e-10)
    # at the truth with independent binomial cells both estimate the same matrix; with only
    # K * N(N-1)/2 = 112 clusters the entrywise noise is large, so compare the total variance.
    assert abs(np.trace(V_cl) / np.trace(V_iid) - 1.0) < 0.25
