import numpy as np

from tracker.multitarget import MultiTargetTracker, run_scene, scan_covariances
from tracker.scene import ScanPlots, Scene, simulate_scene
from tracker.simulate import RadarConfig, TargetConfig

RADAR = RadarConfig(p_detect=1.0)
T = 60


def straight_scene(starts, velocities, seed=0, p_detect=1.0, clutter=0, vanish_at=None):
    """Hand-built scene: aircraft on straight lines (so no random crossings), noisy plots."""
    rng = np.random.default_rng(seed)
    radar = RadarConfig(p_detect=p_detect)
    n = len(starts)
    truths = np.zeros((n, T, 4))
    for i, (p, v) in enumerate(zip(starts, velocities)):
        for k in range(T):
            truths[i, k] = [p[0] + v[0] * k, p[1] + v[1] * k, v[0], v[1]]
    alive = np.ones((n, T), dtype=bool)
    if vanish_at is not None:
        alive[:, vanish_at:] = False
    scans = []
    for k in range(T):
        polar, source = [], []
        for i in np.flatnonzero(alive[:, k]):
            if rng.random() < p_detect:
                x, y = truths[i, k, :2]
                polar.append((np.hypot(x, y) + rng.normal(0, radar.sigma_r),
                              np.arctan2(y, x) + rng.normal(0, radar.sigma_theta)))
                source.append(i)
        for _ in range(int(rng.poisson(clutter))):
            polar.append((60_000 * np.sqrt(rng.random()), rng.uniform(-np.pi, np.pi)))
            source.append(-1)
        polar = np.array(polar).reshape(-1, 2)
        xy = np.stack([polar[:, 0] * np.cos(polar[:, 1]), polar[:, 0] * np.sin(polar[:, 1])], axis=1)
        scans.append(ScanPlots(polar, xy, np.array(source, dtype=int)))
    return Scene(truths, alive, scans, radar)


def test_one_target_without_clutter_gives_exactly_one_confirmed_track_that_follows_it():
    scene = straight_scene([(30_000.0, 10_000.0)], [(-200.0, 50.0)])
    out = run_scene(MultiTargetTracker(), scene)
    assert out[0] == [] and out[1] == []                      # not confirmed straight away
    assert all(len(c) == 1 for c in out[5:])
    assert len({c[0].id for c in out[5:]}) == 1               # one id for the whole flight
    err = np.linalg.norm(out[-1][0].x[:2] - scene.truths[0, -1, :2])
    assert err < 150.0
    assert np.linalg.norm(out[-1][0].x[2:] - scene.truths[0, -1, 2:]) < 60.0


def test_a_track_is_confirmed_after_three_hits_and_not_before():
    scene = straight_scene([(30_000.0, 10_000.0)], [(-200.0, 50.0)])
    out = run_scene(MultiTargetTracker(m_of_n=(3, 5)), scene)
    assert [len(c) for c in out[:4]] == [0, 0, 1, 1]          # scans 0,1,2 give three hits
    out = run_scene(MultiTargetTracker(m_of_n=(5, 5)), scene)
    assert [len(c) for c in out[:6]] == [0, 0, 0, 0, 1, 1]


def test_clutter_alone_gives_almost_no_confirmed_tracks():
    scene = simulate_scene(np.random.default_rng(1), 0, RADAR, TargetConfig(), clutter_rate=20.0)
    out = run_scene(MultiTargetTracker(), scene)
    assert np.mean([len(c) for c in out]) < 0.5                # 20 false plots per scan, ~0 tracks
    assert sum(len(c) for c in out) < 40


def test_two_well_separated_targets_give_two_tracks_that_each_follow_their_own_aircraft():
    scene = straight_scene([(30_000.0, 10_000.0), (-20_000.0, -30_000.0)],
                           [(-200.0, 50.0), (150.0, 100.0)], seed=3)
    out = run_scene(MultiTargetTracker(), scene)
    assert all(len(c) == 2 for c in out[6:])
    last = out[-1]
    for i in range(2):
        errs = [np.linalg.norm(t.x[:2] - scene.truths[i, -1, :2]) for t in last]
        assert min(errs) < 150.0
    assert len({t.id for c in out[6:] for t in c}) == 2        # ids never swap or restart


def test_a_target_that_disappears_loses_its_track_after_the_allowed_misses():
    scene = straight_scene([(30_000.0, 10_000.0)], [(-200.0, 50.0)], vanish_at=30)
    out = run_scene(MultiTargetTracker(max_misses=5), scene)
    assert len(out[29]) == 1 and len(out[33]) == 1             # 4 misses so far: still kept
    assert len(out[34]) == 0 and len(out[-1]) == 0             # the 5th miss deletes it


def test_a_track_coasts_through_missed_detections():
    scene = straight_scene([(30_000.0, 10_000.0)], [(-200.0, 50.0)], seed=5, p_detect=0.7)
    out = run_scene(MultiTargetTracker(), scene)
    assert len({t.id for c in out for t in c}) == 1            # one aircraft, one continuous track
    assert len(out[-1]) == 1
    assert np.linalg.norm(out[-1][0].x[:2] - scene.truths[0, -1, :2]) < 250.0


def test_a_target_among_clutter_is_still_tracked_with_few_false_tracks():
    scene = straight_scene([(30_000.0, 10_000.0)], [(-200.0, 50.0)], seed=6, clutter=10)
    out = run_scene(MultiTargetTracker(), scene)
    near = [min(np.linalg.norm(t.x[:2] - scene.truths[0, k, :2]) for t in c) if c else np.inf
            for k, c in enumerate(out)]
    assert all(d < 200.0 for d in near[10:])
    assert np.mean([len(c) for c in out[10:]]) < 1.5           # at most a stray false track now and then


def test_the_result_is_a_copy_and_the_plot_source_labels_are_not_needed():
    scene = straight_scene([(30_000.0, 10_000.0)], [(-200.0, 50.0)])
    trk = MultiTargetTracker()
    out = None
    for plots in scene.scans[:8]:
        out = trk.step(plots.xy, scan_covariances(plots, scene.radar))
    before = out[0].x.copy()
    out[0].x[:] = 0.0                                          # tampering must not reach the tracker
    np.testing.assert_array_equal(trk.confirmed()[0].x, before)


def test_empty_scans_are_fine_and_tracks_start_from_nothing():
    trk = MultiTargetTracker()
    assert trk.step(np.zeros((0, 2)), np.zeros((0, 2, 2))) == []
    assert trk.step([[1000.0, 2000.0]], [np.eye(2) * 2500.0]) == []
    assert len(trk.tracks) == 1 and trk.tracks[0].status == "tentative"


def test_a_duplicate_track_on_the_same_aircraft_is_removed_but_a_crossing_aircraft_is_not():
    trk = MultiTargetTracker()
    P = np.diag([100.0, 100.0, 50.0, 50.0])
    from tracker.multitarget import TrackState
    trk.tracks = [TrackState(0, np.array([1000.0, 0.0, 200.0, 0.0]), P, "confirmed", [True] * 5),
                  TrackState(1, np.array([1010.0, 5.0, 198.0, 3.0]), P, "tentative", [True, True]),
                  TrackState(2, np.array([1000.0, 0.0, -200.0, 0.0]), P, "confirmed", [True] * 5)]
    trk._remove_duplicates()
    assert sorted(t.id for t in trk.tracks) == [0, 2]          # the copy goes; the crossing aircraft stays


def test_a_tentative_track_that_cannot_reach_m_hits_is_dropped_early():
    trk = MultiTargetTracker(m_of_n=(3, 5))
    trk.step([[1000.0, 2000.0]], [np.eye(2) * 2500.0])
    for _ in range(2):                                         # hits F,F: 3 hits in 5 still possible
        trk.step(np.zeros((0, 2)), np.zeros((0, 2, 2)))
    assert len(trk.tracks) == 1
    trk.step(np.zeros((0, 2)), np.zeros((0, 2, 2)))            # a third miss makes it impossible
    assert trk.tracks == []


def test_a_second_plot_inside_a_tracks_gate_does_not_start_another_track():
    trk = MultiTargetTracker()
    trk.step([[1000.0, 2000.0]], [np.eye(2) * 2500.0])
    trk.step([[1005.0, 2003.0], [1010.0, 1995.0]], [np.eye(2) * 2500.0] * 2)   # both plausible
    assert len(trk.tracks) == 1
    trk.step([[1005.0, 2003.0], [30_000.0, 30_000.0]], [np.eye(2) * 2500.0] * 2)  # a far plot does
    assert len(trk.tracks) == 2
