// Browser (and Node) port of the Python pieces the "Random lab" needs:
// the flight simulator, the radar, the constant-velocity Kalman filter, the network's input
// features and the trained GRU itself. Every function mirrors its Python twin (named in the
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

  // ---- the IMM filter (tracker/imm.py): four motion models, a belief about which one fits ----
  const TUNED_MODELS = [["cv", 1], ["cv", 100], ["ct", 0.05, 10], ["ct", -0.05, 10]];
  const TUNED_STAY = 0.98, PROB_FLOOR = 1e-12;
  const scale = (A, s) => A.map((row) => row.map((v) => v * s));
  const outer = (a, b) => a.map((u) => b.map((v) => u * v));
  const sumMats = (list) => list.reduce((acc, M) => add(acc, M));
  const zeros4 = () => Array.from({ length: 4 }, () => [0, 0, 0, 0]);

  // kalman.ct_matrices: the velocity vector turns at omega rad/s, same noise as constant velocity
  function ctMatrices(omega, q, dt) {
    if (Math.abs(omega * dt) < 1e-9) return cvMatrices(q, dt);
    const s = Math.sin(omega * dt), c = Math.cos(omega * dt);
    const F = [[1, 0, s / omega, -(1 - c) / omega], [0, 1, (1 - c) / omega, s / omega], [0, 0, c, -s], [0, 0, s, c]];
    return [F, cvMatrices(q, dt)[1]];
  }

  function kalmanUpdate(x, P, z, R) {                       // kalman.update; also returns innovation y and S
    const S = [[P[0][0] + R[0][0], P[0][1] + R[0][1]], [P[1][0] + R[1][0], P[1][1] + R[1][1]]];
    const det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
    const Si = [[S[1][1] / det, -S[0][1] / det], [-S[1][0] / det, S[0][0] / det]];
    const K = mm(P.map((row) => [row[0], row[1]]), Si);
    const y = [z[0] - x[0], z[1] - x[1]];
    const xn = x.map((v, i) => v + K[i][0] * y[0] + K[i][1] * y[1]);
    const KH = K.map((row) => [row[0], row[1], 0, 0]);
    return { x: xn, P: mm(eye(4).map((row, i) => row.map((v, j) => v - KH[i][j])), P), y, S, det, Si };
  }

  // imm._combine, state only (the page does not need the combined covariance)
  const combineMean = (xs, w) => [0, 1, 2, 3].map((i) => xs.reduce((s, xj, j) => s + w[j] * xj[i], 0));

  // Returns { est: [[x, y, vx, vy]...], mu: [[p1..p4]...] }
  function immRun(track, models, stay, initSpeedStd) {
    models = models || TUNED_MODELS; stay = stay === undefined ? TUNED_STAY : stay;
    initSpeedStd = initSpeedStd === undefined ? 200 : initSpeedStd;
    const radar = track.radar, M = models.length;
    const mats = models.map((m) => (m[0] === "cv" ? cvMatrices(m[1], radar.dt) : ctMatrices(m[1], m[2], radar.dt)));
    const pi = Array.from({ length: M }, (_, i) => Array.from({ length: M }, (_, j) => (i === j ? stay : (1 - stay) / (M - 1))));
    const [r0, th0] = track.polar[0], R0 = polarToCartCov(r0, th0, radar.sigmaR, radar.sigmaTheta);
    const x0 = [track.xy[0][0], track.xy[0][1], 0, 0];
    const P0 = [[R0[0][0], R0[0][1], 0, 0], [R0[1][0], R0[1][1], 0, 0], [0, 0, initSpeedStd ** 2, 0], [0, 0, 0, initSpeedStd ** 2]];
    let xs = models.map(() => x0.slice()), Ps = models.map(() => P0.map((r) => r.slice()));
    let mu = new Array(M).fill(1 / M);
    const est = [x0.slice()], mus = [mu.slice()];
    for (let k = 1; k < track.detected.length; k++) {
      // 1. mix
      const c = Array.from({ length: M }, (_, j) => mu.reduce((s, m, i) => s + pi[i][j] * m, 0));
      const muIJ = (i, j) => pi[i][j] * mu[i] / c[j];
      const mx = [], mP = [];
      for (let j = 0; j < M; j++) {
        const xm = [0, 1, 2, 3].map((d) => xs.reduce((s, xi, i) => s + muIJ(i, j) * xi[d], 0));
        mx.push(xm);
        mP.push(sumMats(xs.map((xi, i) => scale(add(Ps[i], outer(xi.map((v, d) => v - xm[d]), xi.map((v, d) => v - xm[d]))), muIJ(i, j)))));
      }
      // 2. predict every model
      xs = mx.map((x, j) => mv(mats[j][0], x));
      Ps = mP.map((P, j) => add(mm(mm(mats[j][0], P), tr(mats[j][0])), mats[j][1]));
      if (track.detected[k]) {
        const [r, th] = track.polar[k], R = polarToCartCov(r, th, radar.sigmaR, radar.sigmaTheta), z = track.xy[k];
        const logw = [];
        for (let j = 0; j < M; j++) {                       // 2. update, 3. score
          const u = kalmanUpdate(xs[j], Ps[j], z, R);
          xs[j] = u.x; Ps[j] = u.P;
          const maha = u.y[0] * (u.Si[0][0] * u.y[0] + u.Si[0][1] * u.y[1]) + u.y[1] * (u.Si[1][0] * u.y[0] + u.Si[1][1] * u.y[1]);
          logw.push(Math.log(c[j]) + (-0.5 * (maha + Math.log(u.det)) - Math.log(2 * Math.PI)));
        }
        const mx_ = Math.max(...logw), w = logw.map((l) => Math.exp(l - mx_)), sw = w.reduce((a, b) => a + b, 0);
        mu = w.map((v) => v / sw);                          // 4. new mode probabilities
      } else mu = c.slice();                                // coast: no plot, no evidence
      mu = mu.map((v) => Math.max(v, PROB_FLOOR)); const s = mu.reduce((a, b) => a + b, 0); mu = mu.map((v) => v / s);
      est.push(combineMean(xs, mu)); mus.push(mu.slice());
    }
    return { est, mu: mus };
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
      if (hybrid === "imm") {                  // also show the network the IMM, and correct the IMM instead
        const { est, mu } = immRun(track);
        est.forEach((e, k) => feats[k].push((e[0] - filled[k][0]) / OUT_SCALE, (e[1] - filled[k][1]) / OUT_SCALE, e[2] / VEL_SCALE, e[3] / VEL_SCALE, ...mu[k]));
        base = est.map((e) => [e[0], e[1]]);
      }
    }
    return { feats, base };
  }

  // ---- the trained GRU (gru_numpy.forward). model = window.GRU_MODEL ----
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  // one scan of one GRU layer: x = input vector, h = previous memory. L = {w_ih, w_hh, b_ih, b_hh}
  function gruStep(L, x, h) {
    const H = h.length;
    const gi = L.w_ih.map((row, i) => row.reduce((s, w, j) => s + w * x[j], L.b_ih[i]));
    const gh = L.w_hh.map((row, i) => row.reduce((s, w, j) => s + w * h[j], L.b_hh[i]));
    const nh = new Array(H);
    for (let i = 0; i < H; i++) {
      const r = sigmoid(gi[i] + gh[i]), z = sigmoid(gi[H + i] + gh[H + i]);
      const n = Math.tanh(gi[2 * H + i] + r * gh[2 * H + i]);
      nh[i] = (1 - z) * n + z * h[i];
    }
    return nh;
  }

  function gruForward(model, feats, base) {
    let seq = feats;
    for (const L of model.layers) {
      let h = new Array(L.w_hh[0].length).fill(0);
      const out = [];
      for (const x of seq) { h = gruStep(L, x, h); out.push(h); }
      seq = out;
    }
    const W = model.head.w, B = model.head.b;
    return seq.map((h, k) => [0, 1].map((o) => base[k][o] + (h.reduce((s, v, j) => s + v * W[o][j], B[o])) * OUT_SCALE));
  }

  function gruRun(model, track) {
    const hybrid = model.config.hybrid, { feats, base } = trackFeatures(track, hybrid);
    return gruForward(model, feats, base);
  }

  // ---- the learned-gain filter (tracker/kalmannet.py). model = window.KNET_MODEL. Returns [[x, y, vx, vy]...] ----
  function knetRun(model, track) {
    const T = track.detected.length, filled = forwardFill(track), { feats } = trackFeatures(track, false);
    let x = [filled[0][0], filled[0][1], 0, 0], prev = [0, 0, 0, 0];
    let hs = model.cells.map((c) => new Array(c.w_hh[0].length).fill(0));
    const est = [x.slice()];
    for (let k = 1; k < T; k++) {
      const xp = mv(model.F, x), det = track.detected[k] ? 1 : 0;
      const y = [(filled[k][0] - xp[0]) * det, (filled[k][1] - xp[1]) * det];   // no plot -> no innovation
      let inp = [y[0] / OUT_SCALE, y[1] / OUT_SCALE, prev[0] / OUT_SCALE, prev[1] / OUT_SCALE, prev[2] / VEL_SCALE, prev[3] / VEL_SCALE,
        feats[k][4], feats[k][5], feats[k][6], feats[k][9], feats[k][10]];       // detected, staleness, range, radar noise x2
      model.cells.forEach((c, i) => { hs[i] = gruStep(c, inp, hs[i]); inp = hs[i]; });
      const g = model.head.w.map((row, i) => sigmoid(row.reduce((s, w, j) => s + w * inp[j], model.head.b[i])));
      const corr = [g[0] * y[0], g[1] * y[1], g[2] * model.vmax * y[0], g[3] * model.vmax * y[1]];
      x = xp.map((v, i) => v + corr[i]); prev = corr;
      est.push(x.slice());
    }
    return est;
  }

  const api = { POS_SCALE, VEL_SCALE, OUT_SCALE, TARGET, RADAR, makeRng, simulateTruth, trackFromTruth, simulateTrack,
    forwardFill, polarToCartCov, cvMatrices, kalmanRun, immRun, trackFeatures, gruForward, gruRun, knetRun };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Lab = api;
})(typeof window !== "undefined" ? window : globalThis);
