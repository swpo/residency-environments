"""Manual check of the PhySim ensemble scores (not part of CI).

The fair estimator is unbiased and "plugin" reproduces the original scores bit for bit.
The repository keeps per-environment unit tests out of ``tests/`` (AGENTS.md), so this
lives here and is run by hand from the repository root:

    uv pip install -e ./environments/physim pytest
    pytest scripts/physim/check_scoring.py

The bit-identity checks need the full git history (the pre-fair scorer blobs).
"""

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
current = pytest.importorskip("physim.blobround6_eval")
legacy = pytest.importorskip("physim.legacy_v1.blobround6_eval")

ROOT = Path(__file__).resolve().parents[2]
# Git blob ids of both scorer modules before the fair estimator (upstream/main at 5149b69).
PRE_FAIR_BLOBS = {
    "physim.blobround6_eval": "03219556fc95235b6a93d331f01bda281cc2beef",
    "physim.legacy_v1.blobround6_eval": "04f24b7352e791c8e7d4b25c5fbafc6f52620f49",
}
SCORERS = pytest.mark.parametrize("scorer", [current, legacy], ids=["current", "legacy_v1"])
MEMBERS = pytest.mark.parametrize("m", [2, 4, 16, 64])

CASE = {
    "id": "fair-check",
    "actions": [],
    "queries": [{"sensor": "device0", "t": [0, 1.0]}, {"sensor": "global", "t": [0.5]}],
}
GROUPS = [
    {
        "id": "mixed",
        "selectors": [
            {"query": 0, "time_index": 1, "port": 3, "slot": 5},
            {"query": 0, "time_index": 0, "port": 7, "slot": 12},
            {"query": 1, "time_index": 0, "port": 0, "slot": 1},
        ],
        "scales": [1.0, 0.25, 0.5],
        "unit": "native",
    },
    {
        "id": "single",
        "selectors": [{"query": 0, "time_index": 0, "port": 0, "slot": 0}],
        "scales": [2.0],
        "unit": "native",
    },
]


# Verbatim copies of the scoring functions before the fair estimator (identical in both modules).
def old_marginal_crps(pred, truth):
    """Empirical-distribution CRPS, averaged over independent truth observations."""
    m, n = len(pred), len(truth)
    x, y = pred.reshape(m, -1), truth.reshape(n, -1)
    cross = np.zeros(x.shape[1], dtype=np.float64)
    # One observation at a time avoids M*N*coordinates temporary storage.
    for observation in y:
        cross += np.abs(x - observation).mean(axis=0) / n
    ordered = np.sort(x, axis=0)
    weights = (2 * np.arange(m, dtype=float) - m + 1)[:, None]
    # Exactly half the mean distance over all ordered pairs, including i=j.
    half_self = (weights * ordered).sum(axis=0) / (m * m)
    return (cross - half_self).reshape(pred.shape[1:])


def old_mean_rms_distance(x, y):
    total = 0.0
    for row in x:
        delta = y - row
        # hypot.reduce is stable for finite extreme values until the true norm
        # exceeds float64; the final finite-score guard gives a clear failure.
        total += float(np.hypot.reduce(delta, axis=1).mean()) / math.sqrt(x.shape[1]) / len(x)
    return total


def old_energy_score(pred, truth):
    return old_mean_rms_distance(pred, truth) - 0.5 * old_mean_rms_distance(pred, pred)


def pre_fair_module(scorer, tmp_path, monkeypatch):
    """The scorer module as committed before the fair estimator, loaded from its git blob."""
    blob = PRE_FAIR_BLOBS[scorer.__name__]
    try:
        source = subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "blob", blob], capture_output=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git blob {blob} of the pre-change scorer is not available")
    path = tmp_path / "pre_fair_blobround6_eval.py"
    path.write_bytes(source)
    # A sibling module name keeps the original relative import of blobround6.
    name = scorer.__name__.rsplit(".", 1)[0] + "._pre_fair_blobround6_eval"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


def samples(rng, members, shapes):
    return {"samples": [rng.normal(size=(members, *shape)) for shape in shapes]}


class SeededPredictor:
    calls = 0

    def predict(self, actions, queries, n_samples=64, seed=0):
        self.calls += 1
        return samples(np.random.default_rng(seed), n_samples, current.validate_case(CASE))


class SeededTruth:
    def sample_truth(self, actions, queries, n_samples, truth_seed):
        return samples(np.random.default_rng(truth_seed), n_samples, current.validate_case(CASE))


def gaussian_rms_mean_distance(d):
    """E|X - X'| / sqrt(d) for independent X, X' ~ N(0, I_d)."""
    return 2 * math.exp(math.lgamma((d + 1) / 2) - math.lgamma(d / 2)) / math.sqrt(d)


def assert_within_se(values, expected, z=4.0):
    values = np.asarray(values)
    se = values.std(ddof=1) / math.sqrt(values.size)
    assert abs(values.mean() - expected) < z * se, (values.mean(), expected, se)


# (c) The plugin estimator reproduces the original scorer bit for bit.
@SCORERS
@pytest.mark.parametrize("m", [1, 2, 3, 7, 64])
def test_plugin_functions_match_original_exactly(scorer, m):
    rng = np.random.default_rng(100 + m)
    for d, n in ((1, 1), (5, 2), (40, 3)):
        pred, truth = rng.normal(size=(m, d)) * 3 + 1, rng.normal(size=(n, d))
        assert scorer._energy_score(pred, truth, "plugin") == old_energy_score(pred, truth)
    pred, truth = rng.normal(size=(m, 3, 12, 13)), rng.normal(size=(2, 3, 12, 13))
    new, old = scorer._marginal_crps(pred, truth, "plugin"), old_marginal_crps(pred, truth)
    assert new.shape == old.shape and (new == old).all()


@SCORERS
@pytest.mark.parametrize("m,n", [(1, 1), (2, 2), (5, 3), (64, 2)])
def test_plugin_report_matches_pre_change_scorer(scorer, m, n, tmp_path, monkeypatch):
    pre = pre_fair_module(scorer, tmp_path, monkeypatch)
    rng = np.random.default_rng(7 * m + n)
    shapes = scorer.validate_case(CASE)
    pred, truth = samples(rng, m, shapes), samples(rng, n, shapes)
    old = pre.score_case(CASE, pred, truth, groups=GROUPS, n_samples=m, n_truth=n)
    new = scorer.score_case(CASE, pred, truth, groups=GROUPS, n_samples=m, n_truth=n, estimator="plugin")
    assert new == dict(old, estimator="plugin")
    options = dict(groups=GROUPS, n_samples=m, n_truth=n, predictor_seed=3, truth_seed=5)
    old = pre.evaluate_trusted_predictor(SeededPredictor(), SeededTruth(), CASE, **options)
    new = scorer.evaluate_trusted_predictor(SeededPredictor(), SeededTruth(), CASE, **options, estimator="plugin")
    assert new == dict(old, estimator="plugin")


@pytest.mark.parametrize("estimator", ["fair", "plugin"])
def test_legacy_scorer_matches_current(estimator):
    rng = np.random.default_rng(11)
    pred, truth = rng.normal(size=(6, 9)), rng.normal(size=(2, 9))
    assert legacy._energy_score(pred, truth, estimator) == current._energy_score(pred, truth, estimator)
    assert (legacy._marginal_crps(pred, truth, estimator) == current._marginal_crps(pred, truth, estimator)).all()


# (a) Fair scores are unbiased whatever the ensemble size; plugin is biased by E|X - X'| / (2m).
@MEMBERS
def test_fair_energy_is_unbiased_in_ensemble_size(m):
    d, n, reps = 3, 2, 60000 // m
    rng = np.random.default_rng(1000 + m)
    pred, truth = rng.standard_normal((reps, m, d)), rng.standard_normal((reps, n, d))
    fair = [current._energy_score(x, y, "fair") for x, y in zip(pred, truth)]
    plugin = [current._energy_score(x, y, "plugin") for x, y in zip(pred, truth)]
    # Truths share the forecast law, so the expected score is E|X - X'| / 2 for every m.
    spread = gaussian_rms_mean_distance(d)
    assert_within_se(fair, spread / 2)
    assert_within_se(np.subtract(plugin, fair), spread / (2 * m))


@MEMBERS
def test_fair_crps_is_unbiased_in_ensemble_size(m):
    # Each column is an independent replicate; truths share the N(0, 1) forecast law.
    reps = 40000
    rng = np.random.default_rng(2000 + m)
    pred, truth = rng.standard_normal((m, reps)), rng.standard_normal((2, reps))
    fair = current._marginal_crps(pred, truth, "fair")
    plugin = current._marginal_crps(pred, truth, "plugin")
    assert_within_se(fair, 1 / math.sqrt(math.pi))
    assert_within_se(plugin - fair, 1 / (math.sqrt(math.pi) * m))


def test_plugin_bias_is_detectable_at_small_ensembles():
    rng = np.random.default_rng(3)
    pred, truth = rng.standard_normal((2, 40000)), rng.standard_normal((1, 40000))
    plugin = current._marginal_crps(pred, truth, "plugin")
    se = plugin.std(ddof=1) / math.sqrt(plugin.size)
    assert plugin.mean() - 1 / math.sqrt(math.pi) > 10 * se


# (b) Fair CRPS against the closed form for a Gaussian forecast and a fixed observation.
@MEMBERS
@pytest.mark.parametrize("y", [-1.0, 0.3, 2.5])
def test_fair_crps_matches_gaussian_closed_form(m, y):
    mu, sigma, reps = 0.3, 1.7, 40000
    rng = np.random.default_rng(int(3000 + 10 * m + 10 * y))
    pred = rng.normal(mu, sigma, size=(m, reps))
    fair = current._marginal_crps(pred, np.full((1, reps), y), "fair")
    z = (y - mu) / sigma
    phi = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    cdf = (1 + math.erf(z / math.sqrt(2))) / 2
    assert_within_se(fair, sigma * (z * (2 * cdf - 1) + 2 * phi - 1 / math.sqrt(math.pi)))


@SCORERS
def test_fair_is_the_default_and_matches_brute_force(scorer):
    rng = np.random.default_rng(17)
    shapes = scorer.validate_case(CASE)
    pred, truth = samples(rng, 5, shapes), samples(rng, 3, shapes)
    report = scorer.score_case(CASE, pred, truth, groups=GROUPS, n_samples=5, n_truth=3)
    assert report["estimator"] == "fair"
    assert report["scoring_convention"] == scorer.SCORING_CONVENTIONS["fair"]
    for group, joint in zip(GROUPS, report["joint_energy"]):
        x, y = (
            np.column_stack([a[s["query"]][:, s["time_index"], s["port"], s["slot"]] for s in group["selectors"]])
            / np.asarray(group["scales"])
            for a in (pred["samples"], truth["samples"])
        )
        norm = math.sqrt(x.shape[1])
        cross = np.linalg.norm(x[:, None] - y[None], axis=-1).mean() / norm
        self_pairs = np.linalg.norm(x[:, None] - x[None], axis=-1).sum() / (5 * 4) / norm
        assert joint["energy_score"] == pytest.approx(cross - self_pairs / 2, rel=1e-12)
    x, y = pred["samples"][1], truth["samples"][1]
    brute = np.abs(x[:, None] - y[None]).mean((0, 1)) - np.abs(x[:, None] - x[None]).sum((0, 1)) / (2 * 5 * 4)
    np.testing.assert_allclose(
        report["marginal_crps"][1]["crps_by_port_slot_mean_over_times"], brute.mean(axis=0), rtol=1e-12, atol=1e-15
    )


# (d) The fair estimator needs two prediction members; plugin keeps accepting one.
@SCORERS
def test_fair_requires_two_prediction_members(scorer):
    rng = np.random.default_rng(19)
    one, truth = rng.normal(size=(1, 4)), rng.normal(size=(2, 4))
    with pytest.raises(scorer.EvaluationError, match="at least two prediction members"):
        scorer._energy_score(one, truth)
    with pytest.raises(scorer.EvaluationError, match="at least two prediction members"):
        scorer._marginal_crps(one, truth, "fair")
    assert scorer._energy_score(one, truth, "plugin") == old_energy_score(one, truth)
    shapes = scorer.validate_case(CASE)
    pred, observed = samples(rng, 1, shapes), samples(rng, 1, shapes)
    with pytest.raises(scorer.EvaluationError, match="at least two prediction members"):
        scorer.score_case(CASE, pred, observed, groups=GROUPS, n_samples=1, n_truth=1)
    assert scorer.score_case(CASE, pred, observed, groups=GROUPS, n_samples=1, n_truth=1, estimator="plugin")
    # One truth realization is still fine for the fair estimator.
    report = scorer.score_case(CASE, samples(rng, 2, shapes), observed, groups=GROUPS, n_samples=2, n_truth=1)
    assert report["estimator"] == "fair"
    predictor = SeededPredictor()
    with pytest.raises(scorer.EvaluationError, match="at least two prediction members"):
        scorer.evaluate_trusted_predictor(predictor, SeededTruth(), CASE, groups=GROUPS, n_samples=1, truth_seed=1)
    assert predictor.calls == 0


# (e) Unknown estimators are refused before any scoring work.
@SCORERS
@pytest.mark.parametrize("estimator", ["unbiased", "Fair", "", None, 1])
def test_unknown_estimator_is_rejected(scorer, estimator):
    rng = np.random.default_rng(23)
    pred, truth = rng.normal(size=(4, 3)), rng.normal(size=(2, 3))
    with pytest.raises(scorer.EvaluationError, match="estimator must be one of"):
        scorer._energy_score(pred, truth, estimator)
    with pytest.raises(scorer.EvaluationError, match="estimator must be one of"):
        scorer._marginal_crps(pred, truth, estimator)
    shapes = scorer.validate_case(CASE)
    with pytest.raises(scorer.EvaluationError, match="estimator must be one of"):
        scorer.score_case(
            CASE,
            samples(rng, 4, shapes),
            samples(rng, 2, shapes),
            groups=GROUPS,
            n_samples=4,
            n_truth=2,
            estimator=estimator,
        )
    predictor = SeededPredictor()
    with pytest.raises(scorer.EvaluationError, match="estimator must be one of"):
        scorer.evaluate_trusted_predictor(
            predictor, SeededTruth(), CASE, groups=GROUPS, n_samples=4, n_truth=2, estimator=estimator
        )
    assert predictor.calls == 0


def test_grading_uses_and_records_the_fair_estimator():
    evaluation = pytest.importorskip("physim.evaluation")
    assert evaluation.SCORING_ESTIMATOR == "fair"
    source = Path(evaluation.__file__).read_text()
    assert "estimator=SCORING_ESTIMATOR" in source and "scoring_estimator=SCORING_ESTIMATOR" in source


@SCORERS
@pytest.mark.parametrize("estimator", ["fair", "plugin"])
def test_cli_passes_the_estimator_through(scorer, estimator, monkeypatch, tmp_path):
    seen = {}

    def fake_score_case(*args, **kwargs):
        seen["estimator"] = kwargs.get("estimator")
        return {"ok": True}

    monkeypatch.setattr(scorer, "validate_plan", lambda *a, **k: None)
    monkeypatch.setattr(scorer, "read_json_data", lambda *a, **k: {})
    monkeypatch.setattr(scorer, "read_sample_bundle", lambda *a, **k: {})
    monkeypatch.setattr(scorer, "score_case", fake_score_case)
    out = tmp_path / "report.json"
    argv = [
        "--case",
        "c",
        "--groups",
        "g",
        "--prediction",
        "p",
        "--truth",
        "t",
        "--out",
        str(out),
        "--estimator",
        estimator,
    ]
    assert scorer.main(argv) == 0
    assert seen["estimator"] == estimator


def test_precision_reward_treats_rounding_level_negatives_as_zero():
    rewards = pytest.importorskip("physim.rewards")
    assert rewards.precision_reward(-4.4e-16) == 1.0
    assert rewards.precision_reward(0.0) == 1.0
    with pytest.raises(ValueError):
        rewards.precision_reward(-1e-6)
