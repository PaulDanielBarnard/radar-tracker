import numpy as np
import torch

from tracker.kalman import H_POS, ConstantVelocityKF, cv_matrices
from tracker.kalmannet import (VMAX, KalmanNet, estimate, load_kalmannet, save_kalmannet,
                               steady_state_gain, tracks_to_inputs)
from tracker.simulate import RadarConfig, TargetConfig, simulate_batch

T = TargetConfig().n_steps


def tracks(n=6, mode="random", p_detect=0.8, seed=0):
    return simulate_batch(n, mode, RadarConfig(p_detect=p_detect), TargetConfig(), seed=seed)


def test_the_steady_state_gain_is_a_real_fixed_point_and_has_the_expected_shape():
    K = steady_state_gain(100.0)
    assert K.shape == (4, 2)
    np.testing.assert_allclose(K, steady_state_gain(100.0, dt=1.0), atol=1e-9)
    assert 0.0 < K[0, 0] < 1.0 and 0.0 < K[2, 0] < VMAX and abs(K[0, 1]) < 1e-9
    assert steady_state_gain(1.0)[0, 0] < K[0, 0]                       # calmer filter trusts plots less


def test_the_untrained_network_outputs_exactly_the_starting_gain():
    model = KalmanNet()
    _, gains = model(*tracks_to_inputs(tracks())[:3], return_gains=True)
    expected = torch.tensor(steady_state_gain(100.0), dtype=torch.float32)
    np.testing.assert_allclose(gains[0, 5].detach().numpy(), expected.numpy(), atol=2e-3)
    np.testing.assert_allclose(gains[:, 1:].detach().numpy(), np.broadcast_to(expected, gains[:, 1:].shape), atol=2e-3)


def test_shapes_and_the_first_estimate_is_the_first_plot_with_zero_velocity():
    z, det, static, truth = tracks_to_inputs(tracks(4))
    est = KalmanNet()(z, det, static)
    assert est.shape == (4, T, 4) and truth.shape == (4, T, 2)
    assert torch.equal(est[:, 0, :2], z[:, 0]) and torch.all(est[:, 0, 2:] == 0)


def test_the_gain_stays_inside_its_bounds_even_for_wild_weights():
    model = KalmanNet(hidden=16)
    torch.manual_seed(0)
    with torch.no_grad():
        model.head.weight.normal_(0, 5.0)
        model.head.bias.normal_(0, 50.0)
    _, gains = model(*tracks_to_inputs(tracks())[:3], return_gains=True)
    assert torch.all(gains[:, :, :2] >= 0) and torch.all(gains[:, :, :2] <= 1)
    assert torch.all(gains[:, :, 2:] >= 0) and torch.all(gains[:, :, 2:] <= VMAX + 1e-6)
    assert torch.all(gains[:, :, 0, 1] == 0) and torch.all(gains[:, :, 3, 0] == 0)    # axes stay decoupled
    assert torch.isfinite(model(*tracks_to_inputs(tracks())[:3])).all()


def test_gradients_reach_every_part_of_the_network():
    model = KalmanNet(hidden=16, layers=2)
    torch.manual_seed(1)
    with torch.no_grad():
        model.head.weight.normal_(0, 0.02)            # a zero head would (correctly) block the GRU gradient
    z, det, static, truth = tracks_to_inputs(tracks(4))
    loss = ((model(z, det, static)[:, :, :2] - truth) ** 2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_a_constant_gain_network_equals_a_plain_constant_gain_filter():
    model = KalmanNet(hidden=8)
    K = np.array([[0.4, 0.0], [0.0, 0.3], [0.02, 0.0], [0.0, 0.03]])         # inside the bounds
    with torch.no_grad():
        model.head.weight.zero_()
        a, b = np.array([0.4, 0.3]), np.array([0.02, 0.03]) / VMAX
        model.head.bias.copy_(torch.tensor(np.concatenate([np.log(a / (1 - a)), np.log(b / (1 - b))]),
                                           dtype=torch.float32))
    z, det, static, _ = tracks_to_inputs(tracks(3, p_detect=0.7))
    got = model(z, det, static).detach().numpy()

    F, _ = cv_matrices(100.0, 1.0)
    for b in range(3):                                                        # reference filter, plain numpy
        x = np.concatenate([z[b, 0].numpy(), [0.0, 0.0]])
        for k in range(1, T):
            x = F @ x
            if det[b, k] > 0:
                x = x + K @ (z[b, k].numpy() - x[:2])
            np.testing.assert_allclose(got[b, k], x, rtol=1e-3, atol=0.05)


def test_a_missed_scan_coasts_on_the_prediction():
    z, det, static, _ = tracks_to_inputs(tracks(2, p_detect=1.0))
    det = det.clone()
    det[:, 20:30] = 0.0
    est = KalmanNet()(z, det, static).detach()
    F = torch.tensor(cv_matrices(100.0, 1.0)[0], dtype=torch.float32)
    for k in range(20, 30):
        torch.testing.assert_close(est[:, k], est[:, k - 1] @ F.T)


def test_before_training_it_is_in_the_same_league_as_the_q100_kalman_filter():
    trs = tracks(40, "random", 0.9, seed=4)
    est = estimate(KalmanNet(), trs)
    kf = np.array([ConstantVelocityKF(q=100).run(t) for t in trs])
    truth = np.array([t.truth for t in trs])
    rmse = lambda e: np.sqrt(((e[:, 10:, :2] - truth[:, 10:, :2]) ** 2).sum(-1).mean())
    assert rmse(est) < 1.5 * rmse(kf)


def test_save_and_load_give_the_same_estimates(tmp_path):
    model = KalmanNet(hidden=12, layers=2)
    with torch.no_grad():
        model.head.weight.normal_(0, 0.3)
    trs = tracks(3)
    save_kalmannet(model, tmp_path / "kn.pt")
    loaded = load_kalmannet(tmp_path / "kn.pt")
    assert loaded.config == model.config
    np.testing.assert_allclose(estimate(loaded, trs), estimate(model, trs), atol=1e-4)


def test_every_gain_the_network_can_output_gives_a_stable_filter():
    F, _ = cv_matrices(100.0, 1.0)
    for a in np.linspace(0.001, 0.999, 25):
        for b in np.linspace(0.001, VMAX, 25):
            K = np.array([[a, 0], [0, a], [b, 0], [0, b]])
            assert max(abs(np.linalg.eigvals((np.eye(4) - K @ H_POS) @ F))) < 1.0
