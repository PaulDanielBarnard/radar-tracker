# Radar Tracker

A radar sees an aircraft only as a scatter of noisy dots. This project tracks a
maneuvering target from those dots and asks one question: **when does a classic Kalman
filter beat a small learned neural network (a GRU), and when does it lose?**

Everything runs on a laptop with simulated data. No downloads, no hardware.

**Live demo:** https://pauldanielbarnard.github.io/radar-tracker/
(an animated radar screen where you can watch both trackers clean up the same noisy
plots, scenario by scenario)

## Contents

1. [How to read this README](#1-how-to-read-this-readme)
2. [Quick start](#2-quick-start)
3. [Concepts for beginners](#3-concepts-for-beginners)
4. [Project map](#4-project-map)
5. [The five scenarios](#5-the-five-scenarios)
6. [Evaluating fairly](#6-evaluating-fairly)
7. [Results](#7-results)
8. [Lessons learned](#8-lessons-learned)
9. [Glossary](#9-glossary)
10. [Exam self-check](#10-exam-self-check)
11. [Roadmap and further reading](#11-roadmap-and-further-reading)
12. [Status](#12-status)

---

## 1. How to read this README

- **New to tracking or machine learning?** Read section 3 top to bottom. Each idea builds
  on the one before it, and every concept is tied to the file that implements it.
- **Just want to run it?** Jump to section 2.
- **Want to find a file?** Section 4 lists every file with one line on what it does.
- **Studying for an exam?** Read sections 3, 6 and 7, look up unfamiliar words in the
  glossary (section 9), then test yourself with the questions in section 10.

## 2. Quick start

You need Python 3.10 or newer. Commands below are for Windows PowerShell; on macOS or
Linux use `source .venv/bin/activate` instead of the activate line.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt        # numpy, matplotlib, pytest, torch, scipy
python -m pytest                       # all tests should pass
```

| Command | What it does | Takes |
|---|---|---|
| `python run_baseline.py` | Kalman filters (4 settings) vs raw radar on every scenario; saves plots to `outputs/` | seconds |
| `python train_neural.py --hybrid --randomize --hidden 128 --layers 2 --epochs 50` | trains the GRU and saves `outputs/gru_tracker.pt` | about 5 minutes on a laptop CPU |
| `python run_compare.py` | one table: raw radar vs Kalman vs IMM vs GRU (needs the trained model) | a few minutes |
| `python run_eval.py` | the same comparison over 10 random seeds with confidence intervals and a filter-consistency check | about 30 minutes (`--seeds 2 --tracks 30` for a quick look) |
| `python export_demo.py` | bakes example tracks and results into `web/data.js` | a minute or two |
| `python -m http.server 8000 -d web` | serves the demo page at http://localhost:8000 | instant |

Notes:
- The trained model (`outputs/*.pt`) and everything else in `outputs/` is **not committed**
  (see `.gitignore`), so a fresh clone must run `train_neural.py` before `run_compare.py`,
  `run_eval.py` or `export_demo.py`.
- Run commands from the project root, the folder that contains `run_baseline.py`.
- `train_neural.py --help` lists every option (epochs, model size, learning rate, ...).
  Use `--name` to save several models side by side and `--model` on the other scripts
  to pick one.

## 3. Concepts for beginners

### 3.1 The problem: hidden truth, noisy measurements

An aircraft flies along some true path. You never see that path. You only get a **plot**
(one radar measurement) once per **scan**, and each plot is off by some random amount.
Sometimes the radar misses the aircraft entirely and gives no plot at all.

A **tracker** is an algorithm that turns the stream of noisy plots into the best possible
guess of where the aircraft really is. This is called **state estimation**.

```
truth (hidden):    ----o----o----o----o----o---->
measurements:      -----x---x------x------x---x     (noisy, sometimes missing)
tracker output:    ----o'---o'---o'---o'---o'--->   (should land close to the truth)
```

The simulator gives us the truth, so we can measure exactly how wrong each tracker is.

### 3.2 The radar model (`tracker/simulate.py`)

- The radar sits at the **origin** (0, 0). All units are SI: metres, seconds, radians.
- Each scan (one per second) it measures the target's **range** (distance) and **bearing**
  (angle). Both get Gaussian noise:
  - range noise: standard deviation `sigma_r` = 50 m
  - bearing noise: `sigma_theta` = 0.005 rad (about 0.29 degrees)
- A bearing error becomes a sideways position error that **grows with distance**:
  at 35 km, 0.005 rad is 35,000 x 0.005 = **175 m** sideways, but only 50 m along the
  line of sight. So the noise blob is long and thin, not round.
- With probability `1 - p_detect` the radar **misses** the target and returns nothing.
  The first scan of every track is always a detection, so trackers can start.
- Plots are converted to x, y: `x = r cos(theta)`, `y = r sin(theta)`.
- Each track is 120 scans long. The target starts 20-50 km away at 150-300 m/s.

### 3.3 The state

A tracker keeps a **state**: its current belief about the target.

```
state = [ x, y, vx, vy ]      position (m) and velocity (m/s) in two dimensions
```

The radar only measures position, yet the filter also estimates velocity. It infers speed
and direction from how the plots move over time.

### 3.4 The Kalman filter (`tracker/kalman.py`)

The Kalman filter is the classic answer. It repeats two steps every scan.

**1. Predict.** Assume the target keeps its velocity (a *constant-velocity model*) and
move the estimate forward one scan. Uncertainty grows, because the target might have
turned or sped up.

```
x_pred = F x              F moves position forward by velocity x dt
P_pred = F P F^T + Q      P is the filter's uncertainty (a covariance matrix)
                          Q is the extra uncertainty added by possible maneuvers
```

**2. Update.** A plot z arrives. Compare it with the prediction and blend the two:

```
y = z - H x_pred          innovation: how far the plot is from where we expected it
S = H P_pred H^T + R      how big an innovation we expect (R = radar noise)
K = P_pred H^T S^-1       the gain: how much to trust the plot vs the prediction
x = x_pred + K y          move the estimate toward the plot by a fraction K of y
P = (I - K H) P_pred      uncertainty shrinks, because we learned something
```

The gain **K** is the heart of it. If the prediction is very uncertain (large `P_pred`),
K is large and the filter trusts the plot. If the plot is very noisy (large `R`), K is
small and the filter trusts its prediction.

**A worked example** (one axis only, so the numbers stay small; this is the same maths
as `ConstantVelocityKF`, and the numbers below were checked by running it). Time step 1 s,
`q = 10`. The filter believes the target is at 1000 m moving at 200 m/s, with
uncertainty 100 m and 50 m/s. Radar noise is 50 m.

| Step | Result |
|---|---|
| Predict | position 1000 + 200 = **1200 m**, velocity 200 m/s. Position variance grows from 10,000 to **12,503** (a spread of about 112 m) |
| A plot arrives at 1260 m | innovation y = 1260 - 1200 = **60 m** |
| Innovation covariance | S = 12,503 + 50^2 = **15,003** |
| Gain | K = [**0.833**, **0.167**] (position, velocity) |
| Update | position 1200 + 0.833 x 60 = **1250 m**, velocity 200 + 0.167 x 60 = **210 m/s** |
| New uncertainty | position spread falls from 100 m to **45.6 m** |

So the filter moved 83% of the way toward the plot, nudged its speed estimate up, and
became more certain than either the prediction or the radar alone.

**Turning the radar noise into R.** Because noise is polar, the filter converts it to an
x, y covariance for each plot with the Jacobian `J` of the polar-to-Cartesian conversion:

```
R = J diag(sigma_r^2, sigma_theta^2) J^T       (_polar_to_cart_cov in kalman.py)
```

**Missed scans.** With no plot there is nothing to update with, so the filter just
predicts and its uncertainty `P` keeps growing until the next plot. This is called
*coasting*.

**Starting up.** The filter starts at the first plot with velocity 0 and a large velocity
uncertainty of 200 m/s, so the first few scans are poor. That is why the metrics ignore
the first 10 scans (the *burn-in*).

**The one knob, `q`.** `Q` is built from `q`, the *process noise intensity*: how much
unplanned acceleration the filter expects.

| | small `q` (0.1) | large `q` (100) |
|---|---|---|
| Believes | the target flies steadily | the target may maneuver at any time |
| Gain | small: leans on its smooth prediction | large: follows the plots closely |
| Straight flight | excellent: noise is averaged away | worse: it passes plot noise through |
| Turns | **lags far behind** | follows the turn |

No single `q` is best everywhere. That tradeoff is exactly the gap a learned tracker is
trying to close.

### 3.5 Measuring error (`tracker/metrics.py`)

- **Position error** at one scan is the straight-line distance between the estimate and
  the truth, in metres.
- **RMSE** (root mean square error) combines all scans of all tracks into one number: square
  every error, average, take the square root. Lower is better. It punishes large errors
  more than small ones.
- The first **10 scans are skipped** (burn-in), because every tracker is still settling.
- The **raw radar baseline** is the dumbest possible tracker: just reuse the latest plot
  (`forward_fill` in `tracker/simulate.py`). Everything should beat it.

### 3.6 The neural tracker (`tracker/neural.py`, `tracker/dataset.py`, `train_neural.py`)

**Why try it?** A Kalman filter has to be *told* how targets move and how noisy the radar
is. A neural network can *learn* those patterns from thousands of simulated tracks.

**A GRU reads the scans in order.** A GRU (Gated Recurrent Unit) is a recurrent network.
At each scan it reads that scan's numbers and updates a **memory vector** (64 or 128
numbers) that summarises everything seen so far: roughly how fast the target is moving,
whether it seems to be turning, how noisy the plots look. Built-in gates decide what to
keep and what to forget.

**What it reads.** Each scan becomes a row of numbers (`track_to_arr` in `tracker/dataset.py`),
scaled so they are near size 1, which networks learn from far better than raw metres:

| Columns | Meaning |
|---|---|
| 2 | latest plot position (the last real plot, reused when a scan is missed) |
| 2 | how far that plot moved since the last scan |
| 1 | was this scan a real detection? |
| 1 | scans since the last real plot |
| 1 | range from the radar |
| 2 | bearing as sin and cos (avoids the jump at +/- 180 degrees) |
| 2 | the radar's noise levels (a Kalman filter is told these too, so it is fair) |

That is **11 columns**. In *hybrid* mode (below) 8 more are added, for **19**.

**Residual learning: output a correction, not a position.** A network asked to output a
position directly would have to be accurate to a tiny fraction of tens of kilometres.
Instead it outputs a small **correction** that is added to a base position:

```
estimate = base position + correction        (correction = network output x 1000 m)
```

The output layer starts at **exactly zero**, so an untrained model returns the base
position unchanged (a test checks this). Training can then only improve on the base.

**Hybrid mode (`--hybrid`).** Normally the base is the latest raw plot. In hybrid mode the
base is a Kalman filter's position (q = 100), and the network also sees the estimates of
two Kalman filters (q = 0.1 and q = 100) as extra inputs. It starts at Kalman quality and
learns where to deviate. This is the best-performing configuration.

**How it learns** (`train_neural.py`):
- *Training data:* 2000 simulated tracks. Normally 400 per scenario; with `--randomize`
  every track gets its own random noise level and miss rate, so the model learns the
  general skill instead of five fixed cases.
- *Loss:* the average squared distance error, but each track's error is divided by that
  track's **expected raw radar error** (about 182 m for the normal radar, 546 m for the
  noisy one). Without this the noisy tracks, which have the largest errors, would drown out
  the quiet ones.
- *Each epoch:* the network sees all training tracks once, in small batches, and the
  AdamW optimiser nudges the weights to reduce the loss.
- *Validation:* after every epoch the model is scored on **separate tracks it never
  trains on** (different random seeds). The best epoch is saved. Scoring only on training
  data would reward memorising instead of learning.
- *Training aids:* the learning rate halves when validation stalls, training stops early
  after 8 epochs without improvement, and a small weight penalty discourages extreme weights.

**Fairness.** The GRU reads scans strictly left to right, so at each moment it knows only
the past, exactly like a Kalman filter. Letting it peek at later scans would be cheating.

**Seeds keep the tests honest.** Training uses seeds 1000 and up, validation 2000 and up,
and the test tracks use seed 42 (`run_baseline.py`, `run_compare.py`, `export_demo.py`)
or seeds 10,000 and up (`run_eval.py`). None overlap, so the models are always judged on
tracks they have never seen.

### 3.7 The IMM filter (`tracker/imm.py`)

**The idea.** One Kalman filter must pick one `q`. An **IMM** (Interacting Multiple Model)
filter runs **several filters at once**, each built for a different kind of flight, and keeps
a running belief about which one fits right now. That belief is the **mode probabilities**
(they always add up to 1).

The four models used (`TUNED_MODELS` in `tracker/imm.py`):

| model | what it expects |
|---|---|
| calm straight (`q` = 1) | steady flight |
| agile straight (`q` = 100) | straight flight with surprises |
| left turn (turn rate +0.05 rad/s, `q` = 10) | the target is circling left |
| right turn (turn rate -0.05 rad/s, `q` = 10) | the target is circling right |

A **coordinated-turn** model predicts that the velocity vector *rotates* at a fixed **turn
rate**, so the target flies a circle of radius speed / turn rate (200 m/s at 0.05 rad/s is a
circle of 4,000 m). At turn rate 0 it is exactly the constant-velocity model.

**Each scan has four steps.** `Pi[i][j]` is the chance of switching from model i to model j
between scans, and `mu_i` is the current probability of model i.

```
1. MIX      c_j     = sum_i Pi[i][j] * mu_i                 chance the target is in model j now
            mu_i|j  = Pi[i][j] * mu_i / c_j                 chance it was in model i, given model j now
            x0_j    = sum_i mu_i|j * x_i                    filter j restarts from this blend
            P0_j    = sum_i mu_i|j * (P_i + (x_i - x0_j)(x_i - x0_j)^T)
                      (the second term is the spread: disagreement between filters adds uncertainty)
2. FILTER   every model runs its own Kalman predict and update
3. SCORE    L_j     = exp(-0.5 y^T S^-1 y) / sqrt(det(2 pi S))      how well model j predicted the plot
4. COMBINE  mu_j    = L_j * c_j / sum_k (L_k * c_k)         new mode probabilities
            x       = sum_j mu_j * x_j                      output: probability-weighted average
            P       = sum_j mu_j * (P_j + (x_j - x)(x_j - x)^T)
```

With no plot (a missed scan) there is no evidence, so the probabilities simply follow `Pi`
(`mu = c`) while every filter coasts.

**A worked example** (two models, a calm one A and an agile one B; stay probability 0.95;
the numbers were checked by running the code):

| Step | Result |
|---|---|
| Starting belief | `mu` = [0.8, 0.2]: probably calm |
| Predicted probabilities | c_A = 0.95 x 0.8 + 0.05 x 0.2 = **0.77**, c_B = 0.05 x 0.8 + 0.95 x 0.2 = **0.23** |
| Where did B's start come from? | 0.05 x 0.8 / 0.23 = **17%** from A, **83%** from B |
| A turn begins: the calm model is surprised | likelihoods L_A = 0.02, L_B = 0.08 |
| New belief | A: 0.77 x 0.02 = 0.0154, B: 0.23 x 0.08 = 0.0184, normalised: `mu` = **[0.456, 0.544]** |

In one scan the belief flipped from 80/20 for the calm model to 46/54, and over the next
few scans it moves further as the evidence keeps adding up.

**How it was set up honestly.** The model set and the stay probability were chosen by
sweeps on *validation* seeds (3000 and up), never on the test seeds, trying three turn rates
(0.02, 0.03, 0.05 rad/s) and three stay probabilities (0.90, 0.95, 0.98; the chosen 0.98
scored best). With a single model the IMM is bit-for-bit the plain Kalman filter, which a
test checks.

**Limits.** The turn rate (0.05 rad/s) is the simulator's maximum, so the IMM is tuned to this
simulator. Real trackers also know their aircraft's turn limits, but the comparison with the
GRU, which had to learn everything from examples, is not tilt-free. And the IMM still loses to
a very calm Kalman filter on straight flight (section 7.1).

## 4. Project map

| Path | Purpose |
|---|---|
| `tracker/simulate.py` | the radar and target simulator; `Track` data; `simulate_batch`; `forward_fill` |
| `tracker/kalman.py` | the constant-velocity Kalman filter, reusable `predict` and `update` steps, the coordinated-turn model, optional covariance and NIS output |
| `tracker/imm.py` | the IMM filter (several Kalman filters blended by mode probability) and its tuned settings |
| `tracker/metrics.py` | position error, RMSE (with burn-in), and the shared evaluation helpers |
| `tracker/scenarios.py` | the five test scenarios, the Kalman `q` values, and `random_radar` for training |
| `tracker/dataset.py` | turns tracks into tensors for the network; the input features; hybrid features |
| `tracker/neural.py` | the GRU tracker, plus save and load of models together with their settings |
| `tracker/stats.py` | confidence intervals, paired differences, NEES, chi-squared bounds |
| `tracker/paths.py` | where `outputs/` and the model files live |
| `run_baseline.py` | table of Kalman settings vs raw radar, plus two plots in `outputs/` |
| `train_neural.py` | trains the GRU and prints a per-scenario report |
| `run_compare.py` | one table: raw radar, Kalman, IMM, GRU on seed-42 test tracks |
| `run_eval.py` | multi-seed evaluation of raw radar, Kalman, IMM and GRU: confidence intervals, paired verdicts, filter consistency |
| `export_demo.py` | writes `web/data.js` (example tracks and results) for the demo page |
| `web/index.html` | the demo page: radar (PPI) view, close-up view, results bars |
| `web/data.js` | generated by `export_demo.py`; committed so the site works without training |
| `tests/test_tracker.py` | simulator and Kalman filter tests |
| `tests/test_neural.py` | dataset, model save/load, zero-start and hybrid tests |
| `tests/test_stats.py` | tests for the statistics helpers and the Kalman diagnostics |
| `tests/test_kalman_refactor.py` | proves the reusable Kalman steps give bit-identical results to the original loop |
| `tests/test_imm.py` | IMM filter, mixing, combining, turn-model and mode-probability tests |
| `.github/workflows/pages.yml` | publishes `web/` to GitHub Pages when it changes on `main` |
| `requirements.txt` | Python dependencies (numpy, matplotlib, pytest, torch, scipy) |
| `.gitignore` | keeps the virtual environment, caches and `outputs/` out of git |

## 5. The five scenarios

Every script tests on the same five scenarios (`tracker/scenarios.py`):

| name | target motion | radar |
|---|---|---|
| `straight` | constant velocity | normal noise |
| `turning` | one constant turn, 50-100% of the maximum rate (0.05 rad/s, about 2.9 degrees per second), random direction | normal noise |
| `maneuvers` | random segments of 8-25 scans: 30% straight, otherwise a random turn and a random acceleration up to 5 m/s^2 | normal noise |
| `noisy` | random maneuvers | 3x range and bearing noise (150 m, 0.015 rad) |
| `missed_30` | random maneuvers | 30% of scans have no plot |

Speed is kept between 80 and 400 m/s.

## 6. Evaluating fairly

### 6.1 Why one test set is not enough

Every track is random, so any single table of results carries luck. A gap of a few metres
between two trackers might be real, or might just be this batch of tracks. To tell the
difference you need many independent test batches and a way to measure the doubt.
(`run_eval.py` does this.)

### 6.2 Many seeds and confidence intervals

One **seed** gives one fresh batch of test tracks (`run_eval.py` uses seeds 10,000, 10,001, ...
with 200 tracks each). Score every method on every batch to get one RMSE per method per seed,
then summarise those numbers as a mean with a **95% confidence interval**:

```
mean +/- t * s / sqrt(n)

  n = number of seeds       s = standard deviation of the n RMSEs (divide by n-1)
  t = 2.26 for n = 10       (it is 12.7 for n = 2: with few seeds the interval is very wide)
```

Read it as: if you repeated the whole experiment many times, about 95% of intervals built
this way would contain the true average. A narrow interval means the number is trustworthy.
Implemented in `mean_ci` (`tracker/stats.py`).

### 6.3 Paired comparison

Within one seed, every method sees **exactly the same tracks**. So instead of comparing two
separate intervals, compute the difference per seed (GRU minus best Kalman) and put an
interval around the average difference (`paired_diff_ci`).

- If the interval **excludes 0**, one method is clearly better.
- If it **includes 0**, the data cannot separate them ("no clear difference").

Pairing is more sensitive because hard tracks are hard for everyone. That shared difficulty
cancels in the difference. For the maneuvers scenario the paired interval is +/-0.8 m,
less than half the width of either method's own interval (+/-1.9 and +/-1.8 m).

### 6.4 Is the filter honest about its own uncertainty?

A Kalman filter reports its own uncertainty `P`. We can check whether that claim is true.

```
NEES = (truth - estimate)^T  P^-1  (truth - estimate)     uses the true state (all 4 values)
NIS  = y^T  S^-1  y                                        uses only the innovation y, so it
                                                           needs no ground truth
```

NEES (normalized estimation error squared) divides the real error by the filter's claimed
error. If the filter is honest, NEES averages **4** (the state has 4 numbers) and NIS
averages **2** (a plot has 2 numbers). For an average over `n` independent tracks, the
95% band comes from the chi-squared distribution (`chi2_bounds` in `tracker/stats.py`);
for 200 tracks it is **3.62 to 4.40**.

| Reading | Meaning | Typical cause | Fix |
|---|---|---|---|
| **above** the ideal | **overconfident**: errors are larger than the filter believes | `q` too small, so it trusts its prediction and lags | raise `q`, or model maneuvers |
| **below** the ideal | **too cautious**: errors are smaller than the filter believes | `q` too large, so it throws accuracy away | lower `q` |
| close to the ideal | **consistent** | `q` suits the motion | none needed |

Caveats: consecutive scans are not independent, so the exact band is only a guide. The
tables below call a value "consistent" if it is within 15% of the ideal.
NIS has a practical advantage: it can be computed on real data, where the truth is unknown.

## 7. Results

All GRU numbers are for the recommended model, trained with
`python train_neural.py --hybrid --randomize --hidden 128 --layers 2 --epochs 50`.

### 7.1 Position error over 10 seeds (`python run_eval.py`)

Position RMSE in metres, mean +/- 95% interval over 10 seeds of 200 tracks each. Lower is
better; the best mean in each row is bold.

| scenario | raw radar | KF q=0.1 | KF q=1 | KF q=10 | KF q=100 | IMM | GRU |
|---|---|---|---|---|---|---|---|
| straight | 205.6 ± 2.0 | **57.6 ± 0.8** | 63.3 ± 0.7 | 76.1 ± 0.7 | 96.0 ± 0.7 | 70.2 ± 0.7 | 81.7 ± 0.7 |
| turning | 192.7 ± 2.6 | 2081.3 ± 46.4 | 920.0 ± 20.9 | 327.8 ± 7.6 | 134.6 ± 2.4 | **95.6 ± 1.4** | 104.9 ± 1.6 |
| maneuvers | 202.1 ± 3.5 | 1039.7 ± 43.7 | 505.1 ± 18.0 | 216.9 ± 5.2 | 118.7 ± 1.8 | **111.1 ± 1.4** | 113.6 ± 1.9 |
| noisy | 606.3 ± 10.5 | 1687.3 ± 74.2 | 1021.6 ± 41.9 | 516.2 ± 16.6 | 297.8 ± 4.7 | **275.4 ± 3.6** | 286.2 ± 4.3 |
| missed_30 | 296.7 ± 4.2 | 1153.2 ± 46.9 | 585.7 ± 21.0 | 262.1 ± 6.0 | 146.4 ± 1.9 | **136.5 ± 1.3** | 141.6 ± 2.0 |

Paired differences (negative means the first method has less error):

| scenario | best single Kalman | GRU minus Kalman (m) | verdict |
|---|---|---|---|
| straight | q=0.1 | +24.2 ± 0.7 | Kalman better |
| turning | q=100 | -29.7 ± 1.3 | GRU better |
| maneuvers | q=100 | -5.1 ± 0.8 | GRU better |
| noisy | q=100 | -11.6 ± 3.3 | GRU better |
| missed_30 | q=100 | -4.8 ± 1.2 | GRU better |

| scenario | IMM minus best Kalman (m) | verdict | GRU minus IMM (m) | verdict |
|---|---|---|---|---|
| straight | +12.6 ± 0.4 (q=0.1) | Kalman better | +11.6 ± 0.5 | IMM better |
| turning | -39.1 ± 2.0 (q=100) | IMM better | +9.4 ± 1.2 | IMM better |
| maneuvers | -7.6 ± 0.7 (q=100) | IMM better | +2.4 ± 0.9 | IMM better |
| noisy | -22.4 ± 2.9 (q=100) | IMM better | +10.8 ± 2.2 | IMM better |
| missed_30 | -9.8 ± 1.0 (q=100) | IMM better | +5.0 ± 1.3 | IMM better |

What to take from it:
- **Against single Kalman filters the GRU wins four of five scenarios.** Straight flight goes to a
  very steady filter (q=0.1), which averages the noise away almost perfectly (57.6 m against 81.7 m).
- **A well-tuned IMM does better still. It beats the GRU in all five scenarios and the best
  single Kalman filter in four of five.** Every interval excludes zero. A classical filter
  whose models match the problem (calm, agile, left turn, right turn) beat a network that had to
  learn the motion from examples. Only straight flight stays with q=0.1 (57.6 m against 70.2 m
  for the IMM), because the IMM's extra agile models cost it some smoothing there.
- **The comparison is not tilt-free.** The IMM was given the simulator's maximum turn rate; the
  GRU had to discover it. Real trackers do know their aircraft's limits, but against targets whose
  dynamics are unknown the balance could shift.
- **The turning scenario shows the q tradeoff at its most extreme.** The q=0.1 filter is
  off by about 2 km (it keeps flying straight while the target turns); q=100 gets 134.6 m.
  The GRU, which can adapt, reaches 104.9 m, and the IMM, which switches to a turn model, 95.6 m.
- **Everything beats raw radar**, except the badly tuned filters in turns and noise.
- **The small wins are real.** On maneuvers the GRU beats the best Kalman by 5.1 m and the
  IMM beats the GRU by 2.4 m; both intervals exclude zero, so neither edge is luck.

### 7.2 Is each filter honest? (one batch of 200 tracks per scenario)

Mean **NEES** after the 10-scan burn-in (ideal 4; 95% band for a single scan 3.62 to 4.40):

| scenario | q=0.1 | q=1 | q=10 | q=100 | IMM | reading |
|---|---|---|---|---|---|---|
| straight | 2.5 | 2.2 | 2.1 | 2.1 | 1.8 | every filter is **too cautious** |
| turning | 14,098 | 1,088 | 65.9 | 5.5 | 2.8 | every Kalman q is **overconfident**; the IMM is slightly too cautious |
| maneuvers | 3,652 | 357 | 29.5 | 3.8 | 4.0 | q=100 and the IMM are **consistent** |
| noisy | 2,938 | 370 | 37.6 | 4.8 | 4.4 | q=100 slightly overconfident (20% above ideal); the IMM is consistent |
| missed_30 | 3,609 | 368 | 31.5 | 4.0 | 4.0 | q=100 and the IMM are **consistent** |

Mean **NIS** (ideal 2; the IMM's is computed from its combined prediction):

| scenario | q=0.1 | q=1 | q=10 | q=100 | IMM |
|---|---|---|---|---|---|
| straight | 1.96 | 1.92 | 1.86 | 1.74 | 1.82 |
| turning | 368.0 | 62.4 | 9.28 | 2.56 | 1.94 |
| maneuvers | 83.3 | 18.8 | 4.54 | 2.10 | 2.04 |
| noisy | 31.1 | 10.4 | 3.67 | 2.14 | 2.04 |
| missed_30 | 102.6 | 23.7 | 5.40 | 2.16 | 2.05 |

What to take from it:
- **q=100 is nearly honest in every scenario with maneuvers.** Its claimed uncertainty matches
  its real errors, so large `q` is the right model for aircraft that turn.
- **q=0.1 is wildly overconfident in turns** (NEES about 14,000 against an ideal of 4). It is
  certain of its path while being 2 km wrong. This is the lag problem, now measured.
- **The IMM is the most honest filter overall.** It is consistent on maneuvers, noisy radar and
  missed scans (NEES 4.0 to 4.4), and its NIS is within 3% of the ideal 2 in four scenarios and
  within 9% on straight flight (1.82). In turns it is even a little cautious (NEES 2.8) where
  every single Kalman filter is overconfident.
- **On straight flight every filter reads below 4.** Even q=0.1 is too cautious, because the
  simulated target really never accelerates. An even smaller `q` would be the honest one.
- **No single `q` is honest everywhere.** That is the formal version of "one knob cannot fit
  all scenarios", and the IMM is the fix: it switches model instead of compromising.
- **The GRU has no NEES.** It outputs a position, not an uncertainty, so this check cannot be
  applied to it. That is a real limitation compared with a Kalman filter or an IMM.

### 7.3 Beyond the training range

During development a plain GRU (no Kalman inputs) was tested on a very noisy radar, far
beyond anything it trained on (300 m range noise, 0.03 rad bearing noise). Both the
fixed-scenario and the randomised plain GRU did badly there (about 670 m), losing to a
Kalman filter with q=100 (about 523 m). The **hybrid** GRU, which starts from a Kalman
estimate, got 503.9 m. (That experiment used a throwaway script that is not part of the
repository.) Lesson: a learned tracker is only as good as the conditions it has seen, and
building a physical model into it helps it fail more gracefully.

### 7.4 Why two seeds were not enough

Running `run_eval.py` with only 2 seeds gave "no clear difference" for maneuvers (-4.0 +/- 15.3 m),
noisy and missed scans. The averages were nearly the same as with 10 seeds, but with two
data points the interval is enormous (t = 12.7). The same small gap became clearly real
once there were ten seeds (-5.1 +/- 0.8 m). More data does not change the answer, it
lets you be **sure** of it.

## 8. Lessons learned

- **A units bug hid in plain sight.** The training loss averaged the squared error over the
  x and y axes instead of adding them into a distance, so every RMSE printed during training
  was 1/sqrt(2) (about 0.71 times) too small, and did not match the evaluation tables. Training
  itself was unaffected (an identical rerun gave identical results), but the printed numbers
  misled. Lesson: when two numbers that should agree don't, find out why.
- **Read train versus validation loss.** When training and validation error were almost the
  same, the model was not memorising (overfitting); it was limited by capacity and training
  (underfitting). If validation error had risen while training error fell, that would have
  been overfitting.
- **More epochs did not help.** A run allowed 150 epochs stopped itself at epoch 64 and gave
  results identical to the 50-epoch model, because the best epoch had already been reached.
  What looked like "still improving" was a plateau.
- **Richer input features barely helped** (about 1 m) at the default model size. The network
  was not short of information. A bigger model (hidden 128, two layers) helped most: for
  example turning dropped from 112.6 m to 99.4 m.
- **The noise-balanced loss helped modestly** (turning 119.2 to 112.6 m, missed scans
  167.8 to 154.5 m), by stopping noisy tracks from drowning out quiet ones.
- **Randomising the radar during training helped on unseen conditions** (50% missed scans:
  212.3 to 195.1 m), though not on the five fixed scenarios it was never specifically shown.
- **A well-tuned classical filter beat the learned one.** The IMM beat the GRU in every
  scenario, with every interval excluding zero. Before reaching for a neural network, build the
  strongest classical baseline you can; the network's value is that it needs no motion model
  spelled out, not that it beats a model matched to the problem.
- **Tune on validation data, test on fresh seeds.** The IMM's models and stay probability came
  from sweeps on seeds 3000 and up and never touched the test seeds, so the final numbers are not
  flattered by the tuning.
- **Deliberately breaking code finds weak tests.** Mutation checks on the IMM exposed two
  behaviours nothing tested (coasting through missed scans, and adding the spread between models
  when combining), and one test that rested on a wrong premise: with switching forbidden,
  evidence still accumulates, so the agile model eventually takes over a maneuvering track.
- **A negative result is still a result.** "Training longer does not help" saved time and
  pointed at the real limits.
- **Pairing, many seeds and consistency checks changed what we could claim.** The small wins
  only became believable with them.

## 9. Glossary

| Term | Meaning |
|---|---|
| plot | one radar measurement of the target (range and bearing) |
| scan | one sweep of the radar; here one per second, so one possible plot per second |
| track | one simulated flight: 120 scans of truth and measurements |
| state | the tracker's belief about the target: `[x, y, vx, vy]` |
| covariance (`P`) | a matrix describing how uncertain the state is; its diagonal holds the variances |
| innovation (`y`) | the plot minus the prediction: how surprised the filter is |
| gain (`K`) | how far to move the estimate toward the plot; large when the prediction is uncertain |
| process noise (`q`, `Q`) | how much unplanned acceleration the filter allows for; the one tuning knob |
| burn-in | the first scans of a track, ignored in scoring while trackers settle (10 scans) |
| RMSE | root mean square error: square the errors, average, take the root; lower is better |
| recurrent network | a network that reads a sequence one step at a time and keeps a memory |
| GRU | Gated Recurrent Unit: a recurrent network whose gates decide what to remember or forget |
| residual learning | learning a correction to a base estimate instead of the estimate itself |
| hybrid | the GRU starts from a Kalman estimate and sees Kalman outputs as inputs (`--hybrid`) |
| causal | using only past and present data, never the future |
| validation | scoring on held-out tracks the model never trains on, to choose the best epoch |
| overfitting | learning the training data's quirks; training error low, validation error high |
| underfitting | not learning enough; both training and validation error stay high |
| early stopping | ending training once validation stops improving |
| seed | the number that fixes a random generator, so the same tracks can be recreated |
| confidence interval | a range around an average that expresses how sure we are of it |
| paired comparison | comparing methods on the same tracks, using the per-track or per-seed differences |
| NEES | normalized estimation error squared: real error measured against the filter's claim; ideal = 4 |
| NIS | normalized innovation squared: the same idea using only the innovation; ideal = 2 |
| chi-squared | the distribution that the squared errors of an honest filter follow; gives the bands |
| consistent | the filter's claimed uncertainty matches its real errors |
| overconfident | real errors are larger than the filter claims (NEES above the ideal) |
| too cautious | real errors are smaller than the filter claims (NEES below the ideal) |
| IMM | Interacting Multiple Model: several filters run at once and are blended by how well each fits |
| mode probability | the IMM's current belief about which motion model is right (always sums to 1) |
| transition matrix (`Pi`) | the chance of switching from motion model i to j between scans; its diagonal is the "stay" probability |
| mixing | the IMM step where every filter restarts from a blend of all filters' estimates |
| likelihood | how probable the new plot was under a model's prediction; high means that model predicted well |
| coordinated turn | a motion model where the velocity rotates at a fixed turn rate, so the target flies a circle |
| turn rate | how fast the heading changes, in radians per second (0.05 rad/s is about 2.9 degrees per second) |

## 10. Exam self-check

Try each question before opening the answer.

<details><summary><b>1.</b> What problem does a tracker solve?</summary>

It estimates the hidden true state of a target (position and velocity) from measurements that
are noisy and sometimes missing.
</details>

<details><summary><b>2.</b> The radar only measures position. Why does the state also contain velocity?</summary>

To predict where the target will be next you need its velocity. The filter infers it from how the
plots move over successive scans.
</details>

<details><summary><b>3.</b> Describe the two steps of the Kalman filter.</summary>

Predict: move the estimate forward with the motion model and grow the uncertainty (`P = F P F^T + Q`).
Update: compare the new plot with the prediction (the innovation), and move the estimate toward the
plot by the gain `K`, which also shrinks the uncertainty.
</details>

<details><summary><b>4.</b> What does the gain K control, and when is it large?</summary>

It sets how much to trust the plot versus the prediction. K is large when the prediction is uncertain
(large `P`) or the plot is precise (small `R`), and small in the opposite cases.
</details>

<details><summary><b>5.</b> Predicted position 1200 m, plot at 1260 m, position gain 0.833. What is the new position?</summary>

1200 + 0.833 x (1260 - 1200) = 1200 + 50 = 1250 m.
</details>

<details><summary><b>6.</b> What does q control, and why does a small q lag in turns?</summary>

`q` is how much unplanned acceleration the filter expects. With small `q` it believes the target flies
steadily, so its gain is small and it leans on its straight-line prediction, which keeps flying straight
while the target turns.
</details>

<details><summary><b>7.</b> Why can no single q win in every scenario?</summary>

Small `q` averages noise well on straight flight but lags turns; large `q` follows turns but passes plot
noise through. Straight flight and turns want opposite settings (q=0.1 wins straight, q=100 wins the rest).
</details>

<details><summary><b>8.</b> What does the filter do when a scan is missed?</summary>

It only predicts (no update), so its uncertainty keeps growing until the next plot arrives. This is called coasting.
</details>

<details><summary><b>9.</b> Why are the first 10 scans ignored when scoring?</summary>

The filter starts from one plot with zero velocity and large uncertainty, so every tracker is poor until it
has seen a few scans. Including them would measure start-up, not tracking quality.
</details>

<details><summary><b>10.</b> What is residual learning, and why does the output layer start at zero?</summary>

The network outputs a correction that is added to a base position instead of the position itself, which
would need tiny relative accuracy. With a zero-initialised output layer, the untrained model returns the base
exactly, so training can only improve on it.
</details>

<details><summary><b>11.</b> What does hybrid mode change?</summary>

The base position becomes a Kalman filter's estimate (q=100), and the network also receives the Kalman
estimates (q=0.1 and q=100) as inputs. It starts at Kalman quality and learns where to deviate. It also
degrades more gracefully on conditions outside the training range.
</details>

<details><summary><b>12.</b> Why is each track's loss divided by its expected radar noise?</summary>

Noisy tracks have far larger errors and would dominate the loss. Dividing by the expected raw error makes
quiet and noisy tracks count about equally.
</details>

<details><summary><b>13.</b> Why score on separate validation tracks? What do the loss curves tell you?</summary>

Scoring on training data rewards memorising. If training error falls while validation error rises, the model is
overfitting. If both stay high and similar, it is underfitting (needs more capacity or better training).
</details>

<details><summary><b>14.</b> Why must the GRU be causal?</summary>

A Kalman filter only knows the past and present. A network that saw future scans would have extra information,
so the comparison would be unfair.
</details>

<details><summary><b>15.</b> Why evaluate over many seeds, and what does a 95% confidence interval mean?</summary>

One batch of random tracks carries luck, so small gaps could be chance. Many seeds measure that doubt. A 95%
interval means that if the whole experiment were repeated many times, about 95% of such intervals would
contain the true average.
</details>

<details><summary><b>16.</b> Why compare methods in a paired way?</summary>

Both methods see the same tracks in each seed, so shared difficulty cancels in the per-seed difference. The
interval on the difference is narrower, which exposes small real gaps. If it excludes 0, the gap is clear.
</details>

<details><summary><b>17.</b> What is the ideal NEES here, and what does a NEES of 14,000 mean?</summary>

The ideal is 4, the number of state components. 14,000 is vastly above it: the filter (q=0.1 in the turning
scenario) is overconfident, claiming far smaller errors than it really has.
</details>

<details><summary><b>18.</b> NEES is about 2.1 on straight flight. What does that say, and what would you change?</summary>

It is below 4, so the filter is too cautious: its real errors are smaller than it claims. A smaller `q` would
match the motion better, since the target never accelerates.
</details>

<details><summary><b>19.</b> Why can NEES not be computed for the GRU?</summary>

NEES needs the filter's own covariance. The GRU outputs only a position, with no uncertainty estimate.
</details>

<details><summary><b>20.</b> Where does the Kalman filter beat the GRU here, and why?</summary>

On straight flight (57.6 m against 81.7 m). With nothing to adapt to, a very steady filter (q=0.1) averages
the noise almost perfectly, while the GRU, trained on all kinds of motion, does not smooth that hard.
</details>

<details><summary><b>21.</b> What are the four steps the IMM filter repeats every scan?</summary>

Mix (each filter restarts from a blend of all filters' estimates), filter (each model runs its own
predict and update), score (how well each model predicted the new plot), and combine (update the mode
probabilities and output the probability-weighted average of the states and covariances).
</details>

<details><summary><b>22.</b> An IMM of two straight-flight models that differ only in q never beat the best single filter. What fixed that?</summary>

It was a compromise between two models that were each wrong during a turn. Adding coordinated-turn models
gave it one that actually matches a turn, so in turns it could switch to a model that fits instead of
averaging two that do not.
</details>

<details><summary><b>23.</b> Stay probability 0.95, belief [0.8, 0.2], then likelihoods 0.02 and 0.08. What is the new belief?</summary>

Predicted probabilities: c = [0.77, 0.23]. Multiply by the likelihoods: 0.77 x 0.02 = 0.0154 and
0.23 x 0.08 = 0.0184. Normalise: [0.456, 0.544]. The belief flipped from 80/20 to 46/54 in one scan.
</details>

<details><summary><b>24.</b> Why does the IMM's combined covariance include the spread between the filters?</summary>

If the filters disagree, the true error is larger than any one filter's own variance suggests. The spread
term (x_j - x)(x_j - x)^T adds that disagreement, so the combined uncertainty stays honest.
</details>

<details><summary><b>25.</b> What should the mode probabilities do during a run of missed scans?</summary>

Nothing is learned without a plot, so they only follow the transition matrix (mu becomes Pi^T mu) and drift
toward their long-run values while every filter coasts.
</details>

## 11. Roadmap and further reading

**Where the project could go next** (the IMM filter is done, see sections 3.7 and 7)
- **Several targets and false alarms:** gating (only consider plots near a prediction), the
  Hungarian algorithm for assigning plots to tracks, clutter as random false plots, and
  track management (confirm after M hits in N scans, drop after misses). This is the biggest
  step up in realism.
- **A learned gain (KalmanNet style):** keep the Kalman predict-and-update structure but let a
  small GRU output the gain, so the physics stays fixed and only the gain is learned. It
  should hold up better outside the training range than a free-form GRU.
- **Real flight data:** ADS-B tracks (for example from the OpenSky Network) with simulated radar
  noise added, in place of invented maneuvers.
- **Notebooks only for exploring:** keep the code in the `tracker/` package and its tests, and use
  notebooks that import it for plots and experiments.

**Further reading**
- Kalman filter tutorials: search for "Kalman filter from scratch" or Roger Labbe's free book
  *Kalman and Bayesian Filters in Python*.
- Recurrent networks and GRUs: Chris Olah, *Understanding LSTM Networks* (the GRU is a close
  relative), and the PyTorch `nn.GRU` documentation.
- Tracking theory: Bar-Shalom, Li and Kirubarajan, *Estimation with Applications to Tracking and
  Navigation* (covers NEES, NIS, IMM and data association).
- Statistics: any introduction to confidence intervals and paired t-tests.

## 12. Status

- [x] Step 1: simulator, Kalman baseline, tests
- [x] Step 2: neural tracker (GRU in PyTorch) trained on simulated tracks
- [x] Step 3: comparison by scenario (`run_compare.py`, `run_eval.py`)
- [x] Step 4: browser demo with a traditional radar (PPI) view (`web/`)
- [x] Rigorous evaluation: many seeds, confidence intervals, paired comparison, NEES and NIS
- [x] IMM filter as a stronger classical baseline (it beat the GRU in every scenario)
- [ ] Multi-target tracking with clutter
- [ ] KalmanNet-style learned gain
- [ ] Real ADS-B flight data
