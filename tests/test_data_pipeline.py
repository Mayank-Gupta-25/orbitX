import numpy as np
import pandas as pd
import pytest

from data.kelvins_stats import build_artifacts
from data.screening import rtn_components, select_targets
from orbitx.dynamics import burn_displacement, mean_motion
from orbitx.generator import EncounterGenerator
from orbitx.uncertainty import CovarianceProfile


def make_synthetic_cdms(n_events=300, seed=0) -> pd.DataFrame:
    """A fake table with the same columns as the Kelvins train_data.csv (the ones OrbitX uses)."""
    rng = np.random.default_rng(seed)
    rows = []
    for eid in range(n_events):
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v) * rng.uniform(8e3, 15e3)
        a = np.cross(v, rng.normal(size=3))
        a /= np.linalg.norm(a)
        miss = a * 10 ** rng.uniform(2.5, 4.7)
        base = np.array([40.0, 400.0, 35.0]) * 10 ** rng.uniform(-0.2, 0.4)
        final_risk = rng.uniform(-12, -2)
        for tau in np.sort(rng.uniform(0.2, 6.5, size=rng.integers(5, 9))):
            sig = base * (1.0 + tau * np.array([0.1, 1.2, 0.05]))
            row = {
                "event_id": eid, "mission_id": 1, "time_to_tca": tau, "risk": final_risk,
                "miss_distance": np.linalg.norm(miss), "relative_speed": np.linalg.norm(v),
                "c_object_type": "DEBRIS",
            }
            for ax, i in zip("rtn", range(3)):
                row[f"relative_position_{ax}"] = miss[i]
                row[f"relative_velocity_{ax}"] = v[i]
                row[f"t_sigma_{ax}"] = sig[i] * 0.6
                row[f"c_sigma_{ax}"] = sig[i] * 0.8
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    art = build_artifacts(make_synthetic_cdms())
    out = tmp_path_factory.mktemp("processed")
    art["events"].to_parquet(out / "events.parquet", index=False)
    art["series"].to_parquet(out / "cdm_sigma_series.parquet", index=False)
    art["profile"].save(out / "cov_profile.npz")
    return out, art


def test_artifacts_basic(processed):
    _, art = processed
    ev = art["events"]
    assert len(ev) == 300
    frac = ev["split"].value_counts(normalize=True)
    assert 0.65 < frac["train"] < 0.75 and 0.10 < frac["val"] < 0.20 and 0.10 < frac["test"] < 0.20
    assert ev["hard"].sum() > 0 and (ev.loc[ev["hard"], "split"] == "train").any()
    assert set(art["report"]["pc_validation_vs_ESA_risk"]) >= {"hbr_5m", "hbr_10m", "n_informative_events"}


def test_profile_grows_with_time_to_tca(processed):
    prof: CovarianceProfile = processed[1]["profile"]
    assert prof.sigma(5.0)[1] > prof.sigma(0.3)[1]            # along-track sigma shrinks toward TCA
    assert np.all(prof.sigma(0.0) > 0)


def test_profile_save_load_roundtrip(processed, tmp_path):
    prof = processed[1]["profile"]
    prof.save(tmp_path / "p.npz")
    again = CovarianceProfile.load(tmp_path / "p.npz")
    assert np.allclose(prof.sigma(2.0), again.sigma(2.0))


def test_generator_hard_events_and_geometry(processed):
    gen = EncounterGenerator.from_processed(processed[0], "train", p_hard=1.0, seed=1)
    for _ in range(50):
        c = gen.sample_conjunction(1000.0)
        assert c.hard
        assert abs(c.miss_rtn @ c.vrel_rtn) / (np.linalg.norm(c.miss_rtn) * np.linalg.norm(c.vrel_rtn) + 1e-12) < 1e-9
        assert c.true_pc() >= 1e-30


def test_generator_p_hard_zero_gives_natural_events(processed):
    gen = EncounterGenerator.from_processed(processed[0], "val", p_hard=0.0, seed=2)
    assert not any(gen.sample_conjunction(0.0).hard for _ in range(30))


def test_episode_sampling_constraints(processed):
    gen = EncounterGenerator.from_processed(processed[0], "train", seed=3)
    for _ in range(30):
        conj = gen.sample_episode(horizon_s=72 * 3600, n_conj=3, min_tca_s=12 * 3600, min_sep_s=6 * 3600)
        t = [c.tca_s for c in conj]
        assert t == sorted(t) and 1 <= len(t) <= 3
        assert all(12 * 3600 <= x <= 72 * 3600 - 3600 for x in t)
        assert all(b - a >= 6 * 3600 for a, b in zip(t, t[1:]))


def test_generator_is_reproducible(processed):
    a = EncounterGenerator.from_processed(processed[0], "train", seed=7).sample_episode(n_conj=2)
    b = EncounterGenerator.from_processed(processed[0], "train", seed=7).sample_episode(n_conj=2)
    assert all(np.allclose(x.miss_rtn, y.miss_rtn) and x.tca_s == y.tca_s for x, y in zip(a, b))


def test_uncertainty_shrinks_toward_tca(processed):
    gen = EncounterGenerator.from_processed(processed[0], "train", seed=4)
    c = gen.sample_conjunction(5 * 86400.0)
    assert c.sigma_rtn(3 * 86400.0)[1] > c.sigma_rtn(0.3 * 86400.0)[1]


def test_estimate_noise_matches_sigma(processed):
    gen = EncounterGenerator.from_processed(processed[0], "train", seed=5)
    c = gen.sample_conjunction(5 * 86400.0)
    rng = np.random.default_rng(0)
    tau = 2 * 86400.0
    errs = np.array([c.sample_estimate(tau, rng)[0] - c.miss_rtn for _ in range(4000)])
    assert np.allclose(errs.std(axis=0), c.sigma_rtn(tau), rtol=0.1)


def test_burn_away_reduces_true_pc(processed):
    gen = EncounterGenerator.from_processed(processed[0], "train", p_hard=1.0, hard_k_max=0.5, seed=6)
    n = mean_motion(550e3)
    for _ in range(20):
        c = gen.sample_conjunction(2 * 86400.0)
        away = -3.0 * c.sigma_final_rtn().max() * c.miss_rtn / (np.linalg.norm(c.miss_rtn) + 1e-9)
        assert c.true_pc(away) < c.true_pc()
    assert burn_displacement([0, 0.01, 0], 86400.0, n)[1] < -2000        # a 1 cm/s burn a day early moves ~2.6 km


def test_estimated_pc_uses_given_sigma(processed):
    gen = EncounterGenerator.from_processed(processed[0], "train", p_hard=1.0, hard_k_max=0.5, seed=8)
    c = gen.sample_conjunction(86400.0)
    wide = c.estimated_pc(c.miss_rtn, c.sigma_final_rtn() * 20)
    tight = c.estimated_pc(c.miss_rtn, c.sigma_final_rtn())
    assert wide < tight                                       # wider covariance dilutes a near-centred encounter


def test_rtn_components_and_target_selection():
    r, v = np.array([7000e3, 0, 0]), np.array([0, 7.5e3, 0])
    assert np.allclose(rtn_components(r, v, np.array([1.0, 2.0, 3.0])), [1.0, 2.0, 3.0])
    fake = [
        {"name": "GOOD-1", "catnr": 1, "ecc": 0.001, "perigee_km": 700.0, "epoch_jd": 10.0},
        {"name": "SOME DEB", "catnr": 2, "ecc": 0.001, "perigee_km": 700.0, "epoch_jd": 10.0},
        {"name": "ECCENTRIC", "catnr": 3, "ecc": 0.3, "perigee_km": 700.0, "epoch_jd": 10.0},
        {"name": "STALE", "catnr": 4, "ecc": 0.001, "perigee_km": 700.0, "epoch_jd": 1.0},
        {"name": "LOW", "catnr": 5, "ecc": 0.001, "perigee_km": 300.0, "epoch_jd": 10.0},
    ]
    picked = select_targets(fake, start_jd=10.5, n=5, seed=0, alt_min=600, alt_max=900, name_contains=None)
    assert [p["name"] for p in picked] == ["GOOD-1"]
