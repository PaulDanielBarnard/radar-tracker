// Browser (and Node) port of the Python pieces the "Random lab" needs:
// the flight simulator, the radar, the constant-velocity Kalman filter, the network's input
// features. Every function mirrors its Python twin (named in the
// comment), and web/test_js.mjs checks the numbers against vectors exported from Python.
(function (root) {
  "use strict";

  // ---- constants (tracker/dataset.py, tracker/simulate.py) ----
  const POS_SCALE = 50000, VEL_SCALE = 300, OUT_SCALE = 1000;
  const HYBRID_QS = [0.1, 100], HYBRID_BASE_Q = 100;
  const TARGET = { nSteps: 120, speedRange: [150, 300], startRange: [20000, 50000], maxTurnRate: 0.05, maxAccel: 5, segmentSteps: [8, 25] };
  const RADAR = { sigmaR: 50, sigmaTheta: 0.005, pDetect: 1, dt: 1 };

  // ---- seedable random numbers ----
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function makeRng(seed) {
    const u = mulberry32(seed);
    let spare = null;
    return {
      uniform: (lo, hi) => lo + (hi - lo) * u(),
      random: u,
      integers: (lo, hi) => lo + Math.floor(u() * (hi - lo)),     // like numpy: lo <= n < hi
      normal(mean, sd) {                                          // Box-Muller
        if (spare !== null) { const v = spare; spare = null; return mean + sd * v; }
        let a = 0; while (a === 0) a = u();
        const b = u(), r = Math.sqrt(-2 * Math.log(a));
        spare = r * Math.sin(2 * Math.PI * b);
        return mean + sd * r * Math.cos(2 * Math.PI * b);
      },
    };
  }

  // ---- the flight (simulate._simulate_truth) ----
  // mode: "straight" | "turn" | "random". Returns an array of [x, y, vx, vy].
  function simulateTruth(rng, cfg, dt, mode) {
    const b0 = rng.uniform(-Math.PI, Math.PI), r0 = rng.uniform(cfg.startRange[0], cfg.startRange[1]);
    let px = r0 * Math.cos(b0), py = r0 * Math.sin(b0);
    let speed = rng.uniform(cfg.speedRange[0], cfg.speedRange[1]);
    let heading = rng.uniform(-Math.PI, Math.PI);
    let turn = 0, accel = 0, left = 0;
    if (mode === "turn") turn = (rng.random() < 0.5 ? -1 : 1) * rng.uniform(0.5, 1.0) * cfg.maxTurnRate;
    const truth = [];
    for (let k = 0; k < cfg.nSteps; k++) {
      truth.push([px, py, speed * Math.cos(heading), speed * Math.sin(heading)]);
      if (mode === "random" && left === 0) {
        left = rng.integers(cfg.segmentSteps[0], cfg.segmentSteps[1]);
        if (rng.random() < 0.3) { turn = 0; accel = 0; }
        else { turn = rng.uniform(-1, 1) * cfg.maxTurnRate; accel = rng.uniform(-1, 1) * cfg.maxAccel; }
      }
      left = Math.max(left - 1, 0);
      const ns = Math.min(Math.max(speed + accel * dt, 80), 400);
      const mh = heading + 0.5 * turn * dt, ms = 0.5 * (speed + ns);
      px += ms * dt * Math.cos(mh); py += ms * dt * Math.sin(mh);
      heading += turn * dt; speed = ns;
    }
    return truth;
  }

  // ---- the radar (simulate.track_from_truth) ----
  // A track: { truth, polar[k] = [range, bearing] or null, xy[k] = [x, y] or null, detected[], radar }
  function trackFromTruth(rng, truth, radar) {
    const T = truth.length, polar = [], xy = [], detected = [];
    for (let k = 0; k < T; k++) {
      const r = Math.hypot(truth[k][0], truth[k][1]) + rng.normal(0, radar.sigmaR);
      const th = Math.atan2(truth[k][1], truth[k][0]) + rng.normal(0, radar.sigmaTheta);
      const det = k === 0 ? (rng.random(), true) : rng.random() < radar.pDetect;   // scan 0 is always seen
      detected.push(det);
      polar.push(det ? [r, th] : null);
      xy.push(det ? [r * Math.cos(th), r * Math.sin(th)] : null);
    }
    return { truth, polar, xy, detected, radar };
  }

  function simulateTrack(rng, mode, radar, target) {
    radar = radar || RADAR; target = target || TARGET;
    return trackFromTruth(rng, simulateTruth(rng, target, radar.dt, mode), radar);
  }

  // forward_fill: the latest real plot at every scan
  function forwardFill(track) {
    let last = track.xy[0];
    return track.xy.map((p, k) => { if (track.detected[k]) last = p; return last; });
  }

  // ---- tiny matrix helpers (row-major arrays of arrays) ----
  const mm = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
  const tr = (A) => A[0].map((_, j) => A.map((row) => row[j]));
  const mv = (A, v) => A.map((row) => row.reduce((s, a, k) => s + a * v[k], 0));
  const add = (A, B) => A.map((row, i) => row.map((a, j) => a + B[i][j]));
  const eye = (n) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

  // kalman.polar_to_cart_cov
  function polarToCartCov(r, th, sr, st) {
    const c = Math.cos(th), s = Math.sin(th), J = [[c, -r * s], [s, r * c]];
    return mm(mm(J, [[sr * sr, 0], [0, st * st]]), tr(J));
  }

  // kalman.cv_matrices
  function cvMatrices(q, dt) {
    const F = [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]];
    const a = dt ** 3 / 3, b = dt ** 2 / 2, c = dt;
    const Q = [[a, 0, b, 0], [0, a, 0, b], [b, 0, c, 0], [0, b, 0, c]].map((r) => r.map((v) => v * q));
    return [F, Q];
  }

  // kalman.ConstantVelocityKF.run -> array of [x, y, vx, vy]
  function kalmanRun(track, q, initSpeedStd) {
    initSpeedStd = initSpeedStd === undefined ? 200 : initSpeedStd;
    const radar = track.radar, [F, Q] = cvMatrices(q, radar.dt);
    const [r0, th0] = track.polar[0], R0 = polarToCartCov(r0, th0, radar.sigmaR, radar.sigmaTheta);
    let x = [track.xy[0][0], track.xy[0][1], 0, 0];
    let P = [[R0[0][0], R0[0][1], 0, 0], [R0[1][0], R0[1][1], 0, 0], [0, 0, initSpeedStd ** 2, 0], [0, 0, 0, initSpeedStd ** 2]];
    const est = [x.slice()];
    for (let k = 1; k < track.detected.length; k++) {
      x = mv(F, x); P = add(mm(mm(F, P), tr(F)), Q);                        // predict
      if (track.detected[k]) {                                              // update (skipped on a miss)
        const [r, th] = track.polar[k], R = polarToCartCov(r, th, radar.sigmaR, radar.sigmaTheta);
        const S = [[P[0][0] + R[0][0], P[0][1] + R[0][1]], [P[1][0] + R[1][0], P[1][1] + R[1][1]]];
        const det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
        const Si = [[S[1][1] / det, -S[0][1] / det], [-S[1][0] / det, S[0][0] / det]];
        const PHt = P.map((row) => [row[0], row[1]]);                       // P H^T  (4x2)
        const K = mm(PHt, Si);
        const y = [track.xy[k][0] - x[0], track.xy[k][1] - x[1]];
        x = x.map((v, i) => v + K[i][0] * y[0] + K[i][1] * y[1]);
        const KH = K.map((row) => [row[0], row[1], 0, 0]);                  // K H (4x4)
        const IKH = eye(4).map((row, i) => row.map((v, j) => v - KH[i][j]));
        P = mm(IKH, P);
      }
      est.push(x.slice());
    }
    return est;
  }

  // ---- network inputs (dataset.track_to_arr). Returns { feats (T x n), base (T x 2) } ----
  function trackFeatures(track, hybrid) {
    const T = track.detected.length, filled = forwardFill(track), radar = track.radar;
    const feats = [];
    let since = 0;
    for (let k = 0; k < T; k++) {
      since = k === 0 ? 0 : (track.detected[k] ? 0 : since + 1);
      const step = k === 0 ? [0, 0] : [filled[k][0] - filled[k - 1][0], filled[k][1] - filled[k - 1][1]];
      const r = Math.hypot(filled[k][0], filled[k][1]), b = Math.atan2(filled[k][1], filled[k][0]);
      feats.push([filled[k][0] / POS_SCALE, filled[k][1] / POS_SCALE, step[0] / VEL_SCALE, step[1] / VEL_SCALE,
        track.detected[k] ? 1 : 0, since / 10, r / POS_SCALE, Math.sin(b), Math.cos(b),
        radar.sigmaR / 50, radar.sigmaTheta / 0.005]);
    }
    let base = filled;
    if (hybrid) {
      for (const q of HYBRID_QS) {
        const est = kalmanRun(track, q);
        est.forEach((e, k) => feats[k].push((e[0] - filled[k][0]) / OUT_SCALE, (e[1] - filled[k][1]) / OUT_SCALE, e[2] / VEL_SCALE, e[3] / VEL_SCALE));
        if (q === HYBRID_BASE_Q) base = est.map((e) => [e[0], e[1]]);
      }
    }
    return { feats, base };
  }

  const api = { POS_SCALE, VEL_SCALE, OUT_SCALE, TARGET, RADAR, makeRng, simulateTruth, trackFromTruth, simulateTrack,
    forwardFill, polarToCartCov, cvMatrices, kalmanRun, trackFeatures };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Lab = api;
})(typeof window !== "undefined" ? window : globalThis);
