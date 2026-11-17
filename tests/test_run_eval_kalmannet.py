import numpy as np

import run_eval
from tracker.kalmannet import KalmanNet
from tracker.neural import GRUTracker
from tracker.scenarios import SCENARIOS, STRESS, TARGET, random_radar
from tracker.simulate import simulate_batch


def tracks(n=4):
    mode, radar = SCENARIOS["maneuvers"]
    return simulate_batch(n, mode, radar, TARGET, seed=1)


def test_the_kalmannet_column_appears_only_when_a_model_is_given():
    plain = run_eval.method_rmses(tracks(), GRUTracker())
    with_net = run_eval.method_rmses(tracks(), GRUTracker(), KalmanNet(hidden=8))
    assert "KalmanNet" not in plain and "KalmanNet" in with_net
    assert set(with_net) - {"KalmanNet"} == set(plain)
    assert np.isfinite(with_net["KalmanNet"]) and 10 < with_net["KalmanNet"] < 2000


def test_the_stress_radar_is_noisier_than_anything_in_training():
    rng = np.random.default_rng(0)
    draws = [random_radar(rng) for _ in range(500)]
    radar = STRESS[1]
    assert radar.sigma_r > max(r.sigma_r for r in draws)
    assert radar.sigma_theta > max(r.sigma_theta for r in draws)


def test_run_stress_gives_one_value_per_seed_for_every_method_and_the_verdict_table_prints(capsys):
    res = run_eval.run_stress(2, 3, GRUTracker(), KalmanNet(hidden=8))
    assert "KalmanNet" in res and all(len(v) == 2 for v in res.values())
    run_eval.print_kalmannet_verdicts({"very_noisy": res})
    out = capsys.readouterr().out
    assert "very_noisy" in out and "vs IMM" in out
