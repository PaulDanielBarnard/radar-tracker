import numpy as np

from tracker.scene import simulate_scene, simulate_scene_batch
from tracker.simulate import RadarConfig, TargetConfig

T = TargetConfig().n_steps


def test_shapes_and_source_labels_are_consistent():
    scene = simulate_scene(np.random.default_rng(0), 3, RadarConfig(p_detect=0.8), TargetConfig(), clutter_rate=4.0)
    assert scene.truths.shape == (3, T, 4) and scene.alive.shape == (3, T)
    assert scene.n_targets == 3 and scene.n_scans == T == len(scene.scans)
    for plots in scene.scans:
        m = len(plots.source)
        assert plots.polar.shape == (m, 2) and plots.xy.shape == (m, 2)
        assert set(plots.source.tolist()) <= {-1, 0, 1, 2}
        np.testing.assert_allclose(plots.xy[:, 0], plots.polar[:, 0] * np.cos(plots.polar[:, 1]))
        np.testing.assert_allclose(plots.xy[:, 1], plots.polar[:, 0] * np.sin(plots.polar[:, 1]))


def test_the_same_seed_gives_the_same_scene_and_a_different_seed_does_not():
    kw = dict(n_targets=2, radar=RadarConfig(p_detect=0.9), target=TargetConfig(), clutter_rate=3.0)
    a = simulate_scene(np.random.default_rng(5), **kw)
    b = simulate_scene(np.random.default_rng(5), **kw)
    c = simulate_scene(np.random.default_rng(6), **kw)
    np.testing.assert_array_equal(a.truths, b.truths)
    for pa, pb in zip(a.scans, b.scans):
        np.testing.assert_array_equal(pa.polar, pb.polar)
        np.testing.assert_array_equal(pa.source, pb.source)
    assert not np.array_equal(a.truths, c.truths)


def test_a_scene_with_no_targets_and_no_clutter_is_all_empty_scans():
    scene = simulate_scene(np.random.default_rng(0), 0, clutter_rate=0.0)
    assert all(len(p.source) == 0 and p.xy.shape == (0, 2) for p in scene.scans)


def test_the_number_of_false_alarms_per_scan_follows_the_clutter_rate():
    scenes = simulate_scene_batch(3, 0, RadarConfig(), TargetConfig(), clutter_rate=10.0, seed=2)
    counts = [len(p.source) for s in scenes for p in s.scans]               # 360 scans of pure clutter
    assert abs(np.mean(counts) - 10.0) < 0.8                                 # sd of the mean is about 0.17
    assert abs(np.var(counts) - 10.0) < 3.0                                  # Poisson: variance equals the mean


def test_false_alarms_are_spread_uniformly_over_the_surveillance_disc():
    scenes = simulate_scene_batch(3, 0, RadarConfig(), TargetConfig(), clutter_rate=10.0, seed=3,
                                  region_radius=60_000.0)
    ranges = np.concatenate([p.polar[:, 0] for s in scenes for p in s.scans])
    assert ranges.max() <= 60_000.0
    assert abs(np.mean(ranges < 60_000.0 / np.sqrt(2)) - 0.5) < 0.04        # inner disc holds half the area


def test_the_detection_fraction_matches_p_detect():
    scenes = simulate_scene_batch(4, 4, RadarConfig(p_detect=0.7), TargetConfig(), seed=4)
    seen = sum(int((p.source >= 0).sum()) for s in scenes for p in s.scans)
    possible = sum(int(s.alive.sum()) for s in scenes)                        # 4 x 4 x 120 = 1920 chances
    assert abs(seen / possible - 0.7) < 0.05


def test_target_plots_carry_the_configured_range_and_bearing_noise():
    radar = RadarConfig(p_detect=1.0)
    scenes = simulate_scene_batch(5, 5, radar, TargetConfig(), seed=7)
    dr, dth = [], []
    for s in scenes:
        for k, p in enumerate(s.scans):
            for i in range(len(p.source)):
                truth = s.truths[p.source[i], k]
                dr.append(p.polar[i, 0] - np.hypot(truth[0], truth[1]))
                dth.append(p.polar[i, 1] - np.arctan2(truth[1], truth[0]))
    dr, dth = np.array(dr), np.array(dth)
    dth = (dth + np.pi) % (2 * np.pi) - np.pi                                 # wrap-around safe
    assert len(dr) == 5 * 5 * T
    assert abs(dr.mean()) < 0.5 * radar.sigma_r / 2 and abs(dr.std() / radar.sigma_r - 1) < 0.1
    assert abs(dth.std() / radar.sigma_theta - 1) < 0.1


def test_plots_are_shuffled_so_their_order_reveals_nothing():
    scene = simulate_scene(np.random.default_rng(8), 3, RadarConfig(), TargetConfig(), clutter_rate=5.0)
    # unshuffled, every false alarm would come after every aircraft plot; a false alarm directly
    # followed by an aircraft plot can only happen if the order was shuffled
    assert any(np.any((p.source[:-1] == -1) & (p.source[1:] >= 0)) for p in scene.scans)
    # and the aircraft plots themselves are not always in ascending aircraft order
    assert any(np.any(np.diff(p.source[p.source >= 0]) < 0) for p in scene.scans)


def test_birth_and_death_make_targets_exist_for_only_part_of_the_scene():
    scene = simulate_scene(np.random.default_rng(9), 6, RadarConfig(p_detect=1.0), TargetConfig(),
                           birth_death=True)
    assert not scene.alive.all()
    for i in range(scene.n_targets):
        idx = np.flatnonzero(scene.alive[i])
        assert len(idx) > 0 and np.all(np.diff(idx) == 1)                    # one unbroken lifetime
        assert np.isnan(scene.truths[i, ~scene.alive[i]]).all()
        assert np.isfinite(scene.truths[i, scene.alive[i]]).all()
    for k, p in enumerate(scene.scans):
        assert all(scene.alive[i, k] for i in p.source if i >= 0)            # dead targets give no plots
    ids, xy = scene.truth_at(T // 2)
    assert len(ids) == scene.alive[:, T // 2].sum() and xy.shape == (len(ids), 2)
