// Checks the JavaScript port in web/lab.js against fixed answers from Python (web/testvectors.json).
// Run:  node web/test_js.mjs      (no dependencies; exit code 1 if anything fails)
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const Lab = createRequire(import.meta.url)(join(here, "lab.js"));
const V = JSON.parse(readFileSync(join(here, "testvectors.json"), "utf8"));

let failed = 0, passed = 0;
function check(name, ok, detail = "") {
  if (ok) { passed++; console.log("ok   ", name); } else { failed++; console.log("FAIL ", name, detail); }
}
const maxDiff = (a, b) => Math.max(...a.flatMap((row, i) => row.map((v, j) => Math.abs(v - b[i][j]))));
const mean = (xs) => xs.reduce((s, v) => s + v, 0) / xs.length;
const sd = (xs) => Math.sqrt(mean(xs.map((v) => (v - mean(xs)) ** 2)));

// the Python track, as the JS functions expect it
const track = { truth: V.truth, polar: V.polar, xy: V.xy, detected: V.detected, radar: V.radar };
track.radar = { ...V.radar };

// ---- Kalman filter and features: must match Python closely ----
for (const q of ["0.1", "100"]) {
  const d = maxDiff(Lab.kalmanRun(track, +q), V.kf[q]);
  check(`Kalman filter q=${q} matches Python (max diff ${d.toExponential(2)} m)`, d < 1e-2, d);
}
const { feats, base } = Lab.trackFeatures(track, true);
const df = maxDiff(feats, V.features), db = maxDiff(base, V.base);
check(`19 network features match Python (max diff ${df.toExponential(2)})`, feats[0].length === 19 && df < 1e-4, df);
check(`hybrid base position matches Python (max diff ${db.toExponential(2)} m)`, db < 1e-2, db);
check("plain features are the first 11 columns", Lab.trackFeatures(track, false).feats[5].every((v, j) => Math.abs(v - V.features[5][j]) < 1e-4));
check("the track really has missed scans (the test covers them)", V.detected.some((d) => !d));

// ---- the trained GRU: only if lab.js has the GRU (added in a later step) ----
if (Lab.gruForward) {
  const out = Lab.gruForward(V.model, feats, base);
  const dg = maxDiff(out, V.gru);
  check(`GRU output matches Python (max diff ${dg.toExponential(2)} m)`, dg < 0.05, dg);
  const full = Lab.gruRun(V.model, track), dr = maxDiff(full, V.gru);
  check(`gruRun (features + Kalman base + GRU in one call) matches Python (max diff ${dr.toExponential(2)} m)`, dr < 0.05, dr);
  const moved =Math.max(...out.map((p, k) => Math.hypot(p[0] - base[k][0], p[1] - base[k][1])));
  check("the GRU test output differs from the base (not vacuous)", moved > 5, moved);
}

// ---- the simulator: statistical checks (the random streams differ from NumPy's, the distributions must not) ----
const rng = Lab.makeRng(123);
const g = Array.from({ length: 20000 }, () => rng.normal(0, 1));
check(`normal(0,1): mean ${mean(g).toFixed(3)}, sd ${sd(g).toFixed(3)}`, Math.abs(mean(g)) < 0.03 && Math.abs(sd(g) - 1) < 0.03);
const a = Lab.makeRng(5), b = Lab.makeRng(5);
check("the same seed gives the same numbers", [a.random(), a.normal(0, 1)].join() === [b.random(), b.normal(0, 1)].join());

const radar = { sigmaR: 50, sigmaTheta: 0.005, pDetect: 0.7, dt: 1 };
const errR = [], errT = [], det = [], speeds = [], ranges0 = [], turns = [];
const r2 = Lab.makeRng(2024);
for (let i = 0; i < 300; i++) {
  const t = Lab.simulateTrack(r2, "turn", radar);
  speeds.push(Math.hypot(t.truth[0][2], t.truth[0][3]));
  ranges0.push(Math.hypot(t.truth[0][0], t.truth[0][1]));
  const h = (k) => Math.atan2(t.truth[k][3], t.truth[k][2]);
  let dh = h(10) - h(9); dh = Math.atan2(Math.sin(dh), Math.cos(dh)); turns.push(Math.abs(dh));
  t.truth.forEach((p, k) => {
    det.push(t.detected[k] ? 1 : 0);
    if (!t.polar[k]) return;
    errR.push(t.polar[k][0] - Math.hypot(p[0], p[1]));
    let e = t.polar[k][1] - Math.atan2(p[1], p[0]); e = Math.atan2(Math.sin(e), Math.cos(e)); errT.push(e);
  });
}
check(`range noise sd ${sd(errR).toFixed(1)} m (expect 50)`, Math.abs(sd(errR) - 50) < 2);
check(`bearing noise sd ${sd(errT).toFixed(4)} rad (expect 0.005)`, Math.abs(sd(errT) - 0.005) < 0.0003);
check(`detection fraction ${mean(det).toFixed(3)} (expect about 0.70, scan 0 always seen)`, Math.abs(mean(det) - 0.7) < 0.02);
check(`start speed mean ${mean(speeds).toFixed(0)} m/s (expect 225)`, Math.abs(mean(speeds) - 225) < 8);
check(`start range mean ${mean(ranges0).toFixed(0)} m (expect 35000)`, Math.abs(mean(ranges0) - 35000) < 1500);
check(`turn mode: turn rate per scan between 0.025 and 0.05 rad (min ${Math.min(...turns).toFixed(4)}, max ${Math.max(...turns).toFixed(4)})`,
  Math.min(...turns) > 0.0245 && Math.max(...turns) < 0.0505);
const straight = Lab.simulateTrack(Lab.makeRng(9), "straight", radar);
check("straight mode: velocity never changes", straight.truth.every((p) => Math.abs(p[2] - straight.truth[0][2]) < 1e-9 && Math.abs(p[3] - straight.truth[0][3]) < 1e-9));
const rnd = Lab.simulateTrack(Lab.makeRng(9), "random", radar);
check("random mode: velocity changes", rnd.truth.some((p) => Math.abs(p[2] - rnd.truth[0][2]) > 1));
check("tracks have 120 scans", rnd.truth.length === 120 && rnd.xy.length === 120);

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
