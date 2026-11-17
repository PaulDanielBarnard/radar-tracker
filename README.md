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
3. [Concepts for beginners](#3-concepts-for-beginners) (3.11 covers the web demo)
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
| `python train_neural.py` | trains the recommended GRU (hybrid, randomised radar, hidden 128, 2 layers, 50 epochs) and saves `outputs/gru_tracker.pt` | about 5 minutes on a laptop CPU |
| `python run_compare.py` | one table: raw radar vs Kalman vs IMM vs GRU (needs the trained model) | a few minutes |
| `python run_eval.py` | the same comparison over 10 random seeds with confidence intervals and a filter-consistency check | about 30 minutes (`--seeds 2 --tracks 30` for a quick look) |
| `python train_kalmannet.py` | trains the learned-gain filter (section 3.9) and saves `outputs/kalmannet.pt`; `run_compare.py` and `run_eval.py` include it automatically once it exists | about 10 minutes |
| `python run_eval.py --stress` | only the very noisy radar (300 m, 0.03 rad) that is beyond the training range | about 10 minutes |
| `python run_adsb.py` | the trackers on a real recorded flight seen through a simulated radar; needs a CSV in `data/` (section 3.10) | about 1 minute |
| `python run_multitarget.py` | the multi-target tracker on 1, 3 and 6 aircraft among 0, 5 and 20 false plots per scan, scored with OSPA | about 6 minutes (`--scenes 2` for a quick look) |
| `python export_demo.py` | bakes example tracks and results into `web/data.js` | a minute or two |
| `python train_neural.py --record outputs/training_run.json` then `python export_demo.py --training outputs/training_run.json --only-training` | records a training run and bakes it into `web/training.js` for the page's "How the GRU learns" panel (the page works without that file) | about 6 minutes |
| `python export_model.py` | writes `web/model.js` (the trained GRU for the browser's Random lab) | a second |
| `python export_testvectors.py` then `node web/test_js.mjs` | checks the JavaScript port against Python | a few seconds |
| `python -m http.server 8000 -d web` | serves the demo page at http://localhost:8000 | instant |

Notes:
- The trained model (`outputs/*.pt`) and everything else in `outputs/` is **not committed**
  (see `.gitignore`), so a fresh clone must run `train_neural.py` before `run_compare.py`,
  `run_eval.py` or `export_demo.py`.
- Run commands from the project root, the folder that contains `run_baseline.py`.
- `train_neural.py --help` lists every option (epochs, model size, learning rate, ...). The bare
  command trains the recommended model; `--no-hybrid --no-randomize --hidden 64 --layers 1
  --epochs 30` gives the small plain model used in the early experiments.
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

**Hybrid mode (on by default; `--no-hybrid` turns it off).** Without it the base is the latest raw plot. In hybrid mode the
base is a Kalman filter's position (q = 100), and the network also sees the estimates of
two Kalman filters (q = 0.1 and q = 100) as extra inputs. It starts at Kalman quality and
learns where to deviate. This is the best-performing configuration.

**How it learns** (`train_neural.py`):
- *Training data:* 2000 simulated tracks. Normally 400 per scenario; with randomisation
  (the default; `--no-randomize` turns it off) every track gets its own random noise level and miss rate, so the model learns the
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

### 3.8 Several aircraft and false alarms (`tracker/scene.py`, `association.py`, `multitarget.py`, `ospa.py`)

**The new problem.** Until now the radar saw one aircraft and every plot was that aircraft. With
several aircraft the radar returns a **pile of plots in no particular order**: some come from
aircraft, some are **false alarms** (also called **clutter**: noise, birds, weather). The tracker
must work out which plot belongs to which aircraft *before* it can filter anything. This is
**data association**. (`tracker/scene.py` simulates it: each scan has the detected aircraft plus a
random number of false plots, on average `clutter_rate`, spread evenly over the radar's disc. Each
plot secretly carries its true source, used only for scoring, never by the tracker.)

**The pipeline.** Every scan, `MultiTargetTracker` (`tracker/multitarget.py`) does five things:

```
1. PREDICT    every existing track moves forward one scan (the normal Kalman predict)
2. ASSOCIATE  gating + Hungarian assignment decide which plot belongs to which track
3. UPDATE     a matched track is corrected by its plot; an unmatched track coasts (a miss)
4. INITIATE   a plot nobody claimed starts a new TENTATIVE track
5. MANAGE     tentative -> confirmed after M hits in N scans; drop weak or lost tracks
```

**Gating.** A plot is only a candidate for a track if it is statistically close to where the
track expects it. "Close" is measured in standard deviations by the **squared Mahalanobis
distance** `d^2 = y^T S^-1 y` (the same quantity as NIS: the innovation `y` against its
covariance `S`). If the plot really belongs to the track, `d^2` follows a chi-squared
distribution with 2 degrees of freedom, so 99% of true plots have `d^2` under **9.21**. That is
the gate (`GATE_99`). Example: a track predicts (0, 0) with sigma 20 m in each direction
(`S` = 400 on the diagonal). A plot 30 m away has `d^2` = 2.25, inside the gate. A plot 70 m
away has `d^2` = 12.25, outside it.

**Assignment.** Within the gates, each track gets at most one plot and each plot at most one
track. Letting every track grab its nearest plot (**greedy**) can force a terrible pairing
later. The **Hungarian algorithm** instead finds the pairing with the smallest *total* cost.
A tiny example (cost = `d^2`, rows are tracks, columns are plots; checked by running the code):

| | plot 0 | plot 1 |
|---|---|---|
| track 0 | 1 | 2 |
| track 1 | 2 | 8 |

Greedy takes the cheapest pair first (track 0 with plot 0, cost 1) and is left with track 1 and
plot 1 at cost 8: **total 9**. Hungarian pairs track 0 with plot 1 and track 1 with plot 0:
2 + 2 = **total 4**.

**Track management.** One false plot is enough to start a tentative track, so a track is
reported only after it is **confirmed**: M hits within N scans (default 3 of 5). A tentative
track that can no longer reach 3 hits in 5 is dropped. Clutter almost never hits three scans
in five in a row, so it stays out of the answer. A confirmed track that misses 5 scans in a row
is dropped, which is how a target that leaves is forgotten. Two details the tests exposed:
a plot already inside an existing track's gate is never used to start a new track, and two
tracks whose position *and* velocity agree are merged (an unlucky plot just outside the gate
used to spawn a lasting duplicate of the same aircraft; comparing velocity too keeps two
different aircraft that merely cross apart).

**Scoring several tracks: OSPA (`tracker/ospa.py`).** RMSE needs one track per aircraft in a
known order. With sets of different sizes and no order you need a distance between *sets*.
**OSPA** pairs truths and tracks one-to-one as well as possible, caps each pair's distance at a
cutoff `c` (1000 m here), and charges `c` for every aircraft the tracker missed and every ghost
track it invented. It averages over the larger set. It splits into a **localisation** part (how
far the paired tracks are off) and a **cardinality** part (the cost of the wrong count), with
`total^2 = localisation^2 + cardinality^2`. Example (checked by running the code): two true
aircraft at 0 m and 5,000 m, and one track 100 m from the first:

| Part | Value |
|---|---|
| localisation | sqrt(100^2 / 2) = **70.7 m** (the one pair, averaged over 2) |
| cardinality | sqrt(1000^2 / 2) = **707.1 m** (one aircraft missed) |
| total OSPA | sqrt((100^2 + 1000^2) / 2) = **710.6 m** |

A perfect tracker scores 0, and one that finds nothing scores 1000.

### 3.9 A learned gain (`tracker/kalmannet.py`, `train_kalmannet.py`)

**The idea.** The hybrid GRU (section 3.6) learns a free-form correction to a Kalman estimate.
A **learned-gain filter** (KalmanNet style) keeps the Kalman filter's skeleton and learns only
the **gain** `K`, the number that decides how far to move toward each plot:

```
predict    x_pred = F x                           fixed physics: constant velocity, as in section 3.4
innovate   y      = plot - position of x_pred     how surprised are we?
correct    x      = x_pred + K y                  K is chosen by a small GRU, every scan
```

The motion model stays fixed, so only one thing is learned. The hope was that this would hold up
better on conditions the network never saw than a network that outputs positions freely.

**What the network reads each scan:** the innovation `y`, its own previous correction, whether a
plot arrived, how many scans ago the last one was, the range, and the radar's noise levels (the
things a Kalman filter also knows). On a missed scan there is no innovation, so the filter coasts.

**A bounded gain, and a lesson.** The plan was a free 4x2 gain matrix. With random weights it
diverged (the error reached 10^27): a gain that mixes the x and y axes, or whose velocity part has
the wrong sign, can make the filter unstable even when every number stays bounded. The final
design is the classic **alpha-beta filter**, one **position gain** `alpha` in (0, 1) and one
**velocity gain** `beta` in (0, 0.5) per axis (both from a sigmoid):

```
K = [[alpha_x, 0], [0, alpha_y], [beta_x, 0], [0, beta_y]]
```

For every `alpha` and `beta` in that range the filter is stable, and a test checks the whole
range numerically. The price is that the network cannot ask for a gain outside it.

**It starts as a real filter.** The output layer starts at zero with its bias set to the steady-state
gain of an ordinary `q = 100` Kalman filter: `alpha` = 0.282 and `beta` = 0.0466 (checked by running
the code). So before any training it is a working fixed-gain filter. Worked example: predicted
position 1,200 m, plot at 1,260 m, so `y` = 60 m. New position = 1,200 + 0.282 x 60 = **1,216.9 m**,
and the velocity estimate rises by 0.0466 x 60 = **2.8 m/s**.

**Training** (`train_kalmannet.py`) mirrors the GRU's: random radar settings, separate validation
tracks, the noise-balanced position loss, early stopping, and gradient clipping (the filter loop is
120 steps long, so gradients can blow up). It has 15,044 parameters, about a tenth of the 156,546 in
the hybrid GRU.

### 3.10 Real flights (`tracker/adsb.py`, `run_adsb.py`)

**Why.** Every flight so far was invented by a simulator, with maneuvers the author chose. Real
aircraft fly differently. **ADS-B** is a signal most aircraft broadcast with their own GPS
position, picked up by volunteer receivers (the **OpenSky Network** collects it). A recorded
flight is therefore a trustworthy **truth**, with real, human-flown maneuvers. The radar is still
simulated: we add the usual range and bearing noise and missed plots, so only the *flying* is real.

**How to get a sample.** Put a CSV of positions in `data/opensky_sample.csv` (the `data/` folder is
ignored by git, so nothing is ever committed). It needs these columns:

| column | meaning |
|---|---|
| `time` | seconds since 1970 (epoch seconds) |
| `icao24` | the aircraft's address, to tell aircraft apart |
| `lat`, `lon` | position in degrees |
| `velocity`, `heading` | optional (not needed for tracking) |
| `onground` | optional; rows marked true are skipped |

OpenSky offers small public samples at https://opensky-network.org/datasets/ (the folder
`incident-data/` holds extracts that already have these columns in a `state_vectors` file), and a
free registration gives access to historical state vectors. Other header spellings such as `latitude`,
`longitude`, `groundspeed` or `true_track` are accepted too. Please read and respect the licence of
whatever you download (OpenSky's data is CC BY 4.0 and asks you to cite Schaefer et al., 2014).

**From file to radar track** (`tracker/adsb.py`):

```
1. read      one flight per aircraft; skip rows on the ground or without a position
2. project   latitude/longitude -> metres east and north of the flight's own origin
3. split     cut where reports are more than 5 s apart, or imply more than 400 m/s (a glitch);
             drop pieces shorter than 120 s or flying slower than 50 m/s (taxiing)
4. resample  linear interpolation onto an exact 1-second grid (the radar scans once per second)
5. window    cut 120-scan windows, the same length as the simulated tracks
6. place     move each window so its centre is 20 to 60 km from a radar at the origin
7. measure   truth = the real flight (velocity by finite differences); add radar noise and misses
```

The projection is **equirectangular**: one degree of latitude is about 111,195 m everywhere, and one
degree of longitude is shorter by the cosine of the latitude (about 96,300 m at 30 degrees north).
That is accurate to well under a metre per kilometre over the tens of kilometres of one window.

**What to look for.** Does the ranking of trackers from the simulation survive contact with real
flying? And is the best `q` still 100, or does a calmer real flight prefer less process noise?

### 3.11 The web demo (`web/`)

Open the page with `python -m http.server 8000 -d web`. It has four parts, top to bottom:

1. **Live replay.** One simulated flight, replayed as a rotating radar screen (PPI) or as a zoomed
   close-up. Orange dots are the noisy plots, black is the truth, blue a Kalman filter (pick its `q`),
   green the GRU. The cards show the error now and the average so far.
2. **Which one wins?** The average error per scenario over 500 unseen tracks (`web/data.js`). Click a
   group to replay that scenario.
3. **How the GRU learns.** A recorded 50-epoch training run (`web/training.js`) replayed epoch by epoch.
   Drag the slider (or press play): the green line moves from the network's *starting guess* (epoch 0,
   the Kalman base position) toward the true path, the loss chart gets a cursor, and the bars show the
   GRU's validation error against raw radar and the best Kalman filter. Real numbers from that run
   (validation tracks, not the test seeds of section 7): average error over the five scenarios fell from
   177.1 m to 163.2 m at the saved epoch 49. If `training.js` is missing the panel hides itself.
4. **Random lab.** Choose the radar noise, the missed scans, the motion and how hard it manoeuvres, or press
   *Surprise me*, and the page simulates a brand-new flight and runs raw radar, three Kalman filters (q=0.1,
   10, 100) and the trained GRU **live in JavaScript**. *Run 20 random flights* averages over many flights
   (shown with a 95% interval over flights). Settings beyond the training range (noise above 200 m or 0.02 rad,
   manoeuvres above 1x) show a warning, because the GRU has never seen them.

**How the browser version was checked against Python.** The simulator, the Kalman filter, the 19 network
input features and the GRU were re-written in `web/lab.js`. `export_testvectors.py` writes one fixed flight
(with missed scans) and the answers Python gives for it to `web/testvectors.json`, and `node web/test_js.mjs`
(no dependencies; also run by pytest when Node is installed) compares them. Measured differences: Kalman
filter 5e-5 m, features 5e-7, a small test GRU 5e-5 m, and the real trained model (`web/model.js`) against
PyTorch on a full flight 0.008 m. JavaScript's random numbers differ from NumPy's, so the simulator is
checked *statistically* instead (range and bearing noise, detection rate, speeds, turn rates, straight
flights really straight). I also broke `lab.js` on purpose 23 ways (wrong noise, a swapped gate, a dropped bias, ...): 22 were caught and 1 was a change with no effect.

**Sizes.** `web/model.js` is the 128 x 2 hybrid GRU rounded to 5 significant digits (about 1.4 MB; the
export warns above 4 MB); `web/training.js` is about 580 KB; one random flight with all five methods
takes about 0.2 s in the browser.

**Honest limits.** In one 20-flight click with the default radar (50 m, 0.005 rad, 10% missed, random
manoeuvres) the averages were raw radar 214 m, Kalman q=0.1 806 m, q=10 188 m, q=100 114 +/- 17 m and
GRU 112 +/- 13 m: the GRU and the best Kalman filter are *level*, as in section 7. Every click gives
different flights, so do not read one run as a result; section 7 is the careful one. The checkpoint
exported to `web/model.js` and replayed in `web/training.js` was trained for this demo with the default
recipe; the section 7 tables were measured with an earlier checkpoint of the same recipe, so its numbers
can differ slightly.

## 4. Project map

| Path | Purpose |
|---|---|
| `tracker/simulate.py` | the radar and target simulator; `Track` data; `simulate_batch`; `forward_fill` |
| `tracker/kalman.py` | the constant-velocity Kalman filter, reusable `predict` and `update` steps, the coordinated-turn model, optional covariance and NIS output |
| `tracker/imm.py` | the IMM filter (several Kalman filters blended by mode probability) and its tuned settings |
| `tracker/scene.py` | multi-aircraft scenes: several targets plus Poisson false alarms per scan, with birth and death |
| `tracker/association.py` | gating (Mahalanobis distance, 99% chi-squared gate) and Hungarian assignment, plus a greedy version for comparison |
| `tracker/multitarget.py` | the multi-target track manager: tentative and confirmed tracks, M-of-N confirmation, deletion, duplicate removal |
| `tracker/ospa.py` | the OSPA distance between the true aircraft and the tracks, with localisation and cardinality parts |
| `tracker/metrics.py` | position error, RMSE (with burn-in), and the shared evaluation helpers |
| `tracker/scenarios.py` | the five test scenarios, the Kalman `q` values, and `random_radar` for training |
| `tracker/dataset.py` | turns tracks into tensors for the network; the input features; hybrid features |
| `tracker/adsb.py` | loads ADS-B flights (csv module), projects, splits at gaps, resamples to 1 Hz, windows, places them near the radar, builds `Track`s |
| `tracker/neural.py` | the GRU tracker, plus save and load of models together with their settings |
| `tracker/kalmannet.py` | the learned-gain Kalman filter (a GRU outputs a bounded alpha-beta gain), its steady-state starting gain, save and load |
| `tracker/stats.py` | confidence intervals, paired differences, NEES, chi-squared bounds |
| `tracker/paths.py` | where `outputs/` and the model files live |
| `run_baseline.py` | table of Kalman settings vs raw radar, plus two plots in `outputs/` |
| `train_neural.py` | trains the GRU and prints a per-scenario report |
| `train_kalmannet.py` | trains the learned-gain filter and prints a per-scenario report |
| `run_compare.py` | one table: raw radar, Kalman, IMM, GRU (and the learned-gain filter if trained) on seed-42 test tracks |
| `run_eval.py` | multi-seed evaluation of raw radar, Kalman, IMM, GRU and the learned-gain filter: confidence intervals, paired verdicts, filter consistency; `--stress` for the very noisy radar |
| `run_adsb.py` | evaluates raw radar, Kalman, IMM, GRU and the learned-gain filter on real flights in `data/opensky_sample.csv`, over several radar placements |
| `run_multitarget.py` | multi-target evaluation: aircraft counts x clutter rates, scored with OSPA and the confirmed-track count |
| `export_demo.py` | writes `web/data.js` (example tracks and results) for the demo page |
| `web/index.html` | the demo page: radar (PPI) view, close-up view, results bars |
| `web/data.js` | generated by `export_demo.py`; committed so the site works without training |
| `web/training.js` | generated by `export_demo.py --training`: a recorded 50-epoch training run (about 580 KB) replayed by the "How the GRU learns" panel. Validation RMSE over the five scenarios fell from 177.1 m (untrained) to 163.2 m at the saved epoch 49: straight 95.0 to 79.3, turning 130.4 to 100.4, maneuvers 120.8 to 115.6, noisy 304.0 to 286.5, missed_30 154.1 to 146.0 (validation tracks, not the test seeds of section 7) |
| `tracker/gru_numpy.py` | the trained GRU re-implemented in plain NumPy (the maths the browser runs), checked against PyTorch |
| `export_model.py` | writes `web/model.js`, the trained GRU's weights as numbers for the browser |
| `export_testvectors.py` | writes `web/testvectors.json`: fixed Python answers the JavaScript port is tested against |
| `web/lab.js` | JavaScript port of the simulator, radar, Kalman filter, network features and GRU (used by the Random lab and by Node) |
| `web/model.js` | generated by `export_model.py` (about 1.4 MB) |
| `web/test_js.mjs` | `node web/test_js.mjs`: checks `lab.js` against Python's answers and the simulator statistically |
| `tests/test_gru_numpy.py` | NumPy GRU equals PyTorch (plain/hybrid, 1/2 layers), hand-computed gate cases |
| `tests/test_export_model.py` | the exported weights rebuild a model with the same outputs; rounding, shapes, size warning |
| `tests/test_js_port.py` | the test vectors are what Python computes; runs the Node test when Node is installed |
| `tests/test_tracker.py` | simulator and Kalman filter tests |
| `tests/test_neural.py` | dataset, model save/load, zero-start and hybrid tests |
| `tests/test_stats.py` | tests for the statistics helpers and the Kalman diagnostics |
| `tests/test_kalman_refactor.py` | proves the reusable Kalman steps give bit-identical results to the original loop |
| `tests/test_imm.py` | IMM filter, mixing, combining, turn-model and mode-probability tests |
| `tests/test_scene.py` | scene simulation: clutter rate, uniform spread, detection fraction, noise, shuffling, birth and death |
| `tests/test_association.py` | gating, Hungarian assignment checked against brute force, greedy comparison |
| `tests/test_multitarget.py` | track manager: one target, clutter only, two targets, a vanishing target, coasting, duplicates |
| `tests/test_ospa.py` | OSPA hand cases: equal sets, empty sets, a missing aircraft, a ghost, best pairing |
| `tests/test_run_multitarget.py` | the multi-target scoring and one evaluation cell |
| `tests/test_adsb.py` | the ADS-B loader on synthetic CSV files: projection, header names, gaps, glitches, slow and short pieces, windows, placement, noise statistics |
| `tests/test_run_adsb.py` | the ADS-B evaluation: scores per method, paired verdict signs, missing-file message |
| `.github/workflows/pages.yml` | publishes `web/` to GitHub Pages when it changes on `main` |
| `requirements.txt` | Python dependencies (numpy, matplotlib, pytest, torch, scipy) |
| `.gitignore` | keeps the virtual environment, caches, `outputs/` and `data/` out of git |

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
`python train_neural.py` (its defaults are hybrid, randomised radar, hidden 128, two layers and
50 epochs; the older long form `--hybrid --randomize --hidden 128 --layers 2 --epochs 50` is identical).

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
estimate, got 503.9 m. (That first experiment used a throwaway script that is not part of the
repository.) The same test is now repeatable with `python run_eval.py --stress` (10 seeds of
200 tracks, random maneuvers; training never went above 200 m and 0.02 rad):

| method | position error (m) |
|---|---|
| raw radar | 1212.4 ± 21.0 |
| KF q=0.1 / 1 / 10 | 2024.0 ± 89.5 / 1437.1 ± 57.6 / 840.5 ± 29.4 |
| KF q=100 | 522.8 ± 9.0 |
| IMM | **482.2 ± 7.2** |
| hybrid GRU | 502.6 ± 7.8 |
| learned-gain filter (section 3.9) | 559.0 ± 8.4 |

Lesson: a learned tracker is only as good as the conditions it has seen, and building a physical
model into it helps it fail more gracefully (the hybrid GRU still beats the best single Kalman
filter here). The learned-gain filter was expected to be the most robust and was **not**: see 7.6.

### 7.4 Why two seeds were not enough

Running `run_eval.py` with only 2 seeds gave "no clear difference" for maneuvers (-4.0 +/- 15.3 m),
noisy and missed scans. The averages were nearly the same as with 10 seeds, but with two
data points the interval is enormous (t = 12.7). The same small gap became clearly real
once there were ten seeds (-5.1 +/- 0.8 m). More data does not change the answer, it
lets you be **sure** of it.

### 7.5 Several aircraft among false alarms (`python run_multitarget.py`)

The track manager (`q` = 100, 3 hits in 5 scans to confirm, dropped after 5 misses) was run on
10 freshly simulated scenes per cell (seeds 10,000 and up), 120 scans each, aircraft flying random
maneuvers, each detected with probability 0.9 per scan. The first 10 scans are skipped. OSPA in metres
(lower is better, 0 is perfect, 1000 means nothing found), mean +/- 95% interval over the scenes:

| aircraft | false plots per scan | OSPA | localisation | cardinality | tracks found (true count) |
|---|---|---|---|---|---|
| 1 | 0 | 116.9 +/- 33.2 | 116.9 | 0.0 | 1.00 (1) |
| 1 | 5 | 99.8 +/- 24.6 | 92.7 | 8.4 | 1.01 (1) |
| 1 | 20 | 128.5 +/- 42.1 | 116.8 | 13.0 | 1.01 (1) |
| 3 | 0 | 118.9 +/- 19.4 | 111.2 | 9.4 | 3.01 (3) |
| 3 | 5 | 119.0 +/- 17.9 | 108.7 | 12.9 | 3.02 (3) |
| 3 | 20 | 130.8 +/- 20.1 | 117.6 | 16.4 | 3.03 (3) |
| 6 | 0 | 137.7 +/- 13.2 | 124.1 | 18.1 | 6.02 (6) |
| 6 | 5 | 148.5 +/- 19.7 | 132.7 | 21.4 | 6.05 (6) |
| 6 | 20 | 157.3 +/- 23.6 | 137.8 | 26.9 | 6.04 (6) |

**Reading it.**
- **The count is almost always right.** Even with 20 false plots per scan the tracker reports
  within 0.05 of the true number of aircraft on average, so the cardinality part is small (0 to 27 m).
- **The error is mostly position error.** The localisation part (about 90 to 140 m) is close to what a
  single `q` = 100 Kalman filter achieves on one aircraft alone (section 7.1), so tracking among clutter
  costs little extra accuracy.
- **More aircraft and more clutter cost a little.** From 1 to 6 aircraft with no clutter, OSPA rose from
  about 117 to 138 m, and 20 false plots per scan add roughly 10 to 20 m on top.
- **Honest limits.** Only 10 scenes per cell, so the intervals are wide (the one-aircraft rows overlap
  each other, so their order means nothing). The aircraft are placed at random, so there is no dedicated
  test of crossing or very close targets, where association is hardest. The tracker uses one Kalman
  `q` for everything (the IMM could be plugged in per track, but is not).

### 7.6 The learned-gain filter (`python run_eval.py`, with `outputs/kalmannet.pt` trained)

Same 10 seeds of 200 tracks as 7.1. Position RMSE in metres (the filter trained for 40 epochs):

| scenario | best single Kalman | IMM | hybrid GRU | learned-gain filter |
|---|---|---|---|---|
| straight | **57.6** (q=0.1) | 70.2 | 81.7 | 107.7 ± 0.9 |
| turning | 134.6 (q=100) | **95.6** | 104.9 | 120.8 ± 1.5 |
| maneuvers | 118.7 (q=100) | **111.1** | 113.6 | 116.7 ± 1.8 |
| noisy | 297.8 (q=100) | **275.4** | 286.2 | 288.6 ± 4.3 |
| missed_30 | 146.4 (q=100) | **136.5** | 141.6 | 144.3 ± 1.7 |

Paired differences for the learned-gain filter (negative = it has less error):

| scenario | minus IMM (m) | minus GRU (m) | minus best Kalman (m) |
|---|---|---|---|
| straight | +37.5 ± 0.7 | +26.0 ± 0.6 | +50.1 ± 0.8 |
| turning | +25.3 ± 1.2 | +15.9 ± 0.8 | **-13.8 ± 1.0** |
| maneuvers | +5.6 ± 0.6 | +3.2 ± 0.6 | **-2.0 ± 0.6** |
| noisy | +13.2 ± 1.2 | +2.4 ± 1.9 | **-9.2 ± 3.4** |
| missed_30 | +7.8 ± 0.8 | +2.7 ± 1.0 | **-2.1 ± 1.0** |

**Reading it, honestly.**
- **It does not beat the IMM or the GRU anywhere.** Every one of those intervals excludes zero, in
  the wrong direction for the learned-gain filter.
- **It does beat the best single Kalman filter in four of five scenarios** (13.8 m better in turns), because
  it can raise its gain when the target turns. On straight flight it loses clearly: the training
  mixes all kinds of motion and radar, and the trained gain smooths less than q=0.1.
- **The hoped-for robustness did not appear.** On the very noisy radar (section 7.3) it scored 559.0 m,
  worse than the q=100 Kalman filter (522.8 m), the GRU (502.6 m) and the IMM (482.2 m), each by a clear
  margin. Likely reasons (**guesses, not tested**): its gain is capped at the range needed for the
  training radars, training noise stopped at 200 m, and its inputs are scaled by fixed constants chosen for
  normal conditions.
- **It has no covariance, so no NEES.** Like the GRU, it cannot say how sure it is.
- **What it is good for.** It is small (15,044 parameters), always stable, and starts as a working filter,
  but here a good classical IMM beat it. A negative result: keep the idea, but do not expect it to win
  without more work (a wider training range, or learning the gain *and* the process noise).

### 7.7 A real flight (`python run_adsb.py`)

The sample is **one aircraft** (a Learjet from an OpenSky incident extract): 4,156 seconds of flying
in 12 gap-free segments, which give **28 non-overlapping 120-scan windows**. Each repeat places the
windows at new random spots near the radar and draws new radar noise; there are 10 repeats. Position
RMSE in metres, mean +/- 95% interval over the 10 placements (the GRU and the learned-gain filter are
the models trained on **simulated** data only, nothing was retrained on this flight):

| method | normal radar (50 m, 0.005 rad) | noisy radar (150 m, 0.015 rad) |
|---|---|---|
| raw radar | 216.0 ± 10.9 | 648.0 ± 32.7 |
| KF q=0.1 | 204.3 ± 10.4 | 366.4 ± 28.9 |
| KF q=1 | 120.9 ± 6.7 | 270.2 ± 10.3 |
| KF q=10 | 95.0 ± 2.9 | 229.6 ± 8.4 |
| KF q=100 | 105.1 ± 4.1 | 250.5 ± 10.6 |
| IMM | **90.8 ± 2.7** | **223.7 ± 9.3** |
| GRU | 96.8 ± 3.7 | 244.1 ± 10.9 |
| learned-gain filter | 116.6 ± 5.1 | 283.8 ± 13.9 |

Paired differences (negative = the first has less error), normal radar then noisy radar:

| comparison | normal | noisy |
|---|---|---|
| IMM minus best Kalman (q=10) | -4.3 ± 2.3 (IMM better) | -5.9 ± 5.5 (IMM better, barely) |
| GRU minus best Kalman (q=10) | +1.8 ± 2.2 (no clear difference) | +14.5 ± 7.1 (Kalman better) |
| GRU minus IMM | +6.1 ± 1.5 (IMM better) | +20.4 ± 3.9 (IMM better) |
| learned-gain minus IMM | +25.8 ± 2.8 (IMM better) | +60.2 ± 6.3 (IMM better) |

**Reading it, honestly.**
- **The simulated ranking mostly holds.** The IMM is best again, the GRU is level with the best single
  Kalman filter on a normal radar and a little behind it on a noisy one, and the learned-gain filter is the
  weakest of the serious contenders. Real flying did not overturn the simulation's conclusions.
- **The best single `q` moved from 100 to 10.** This flight (climb, cruise, gentle turns) is smoother than the
  simulator's random maneuvers, so a calmer filter fits it better. That is a likely explanation, not a tested one.
- **The IMM's lead over the best Kalman filter is small** (4 to 6 m) and on the noisy radar its interval almost
  touches zero.
- **The models were never trained on this flight**, so the GRU result is a fair test of how well simulated
  training carries over to real motion: it carries over reasonably, without beating the classical filters.
- **Limits.** One aircraft and about 70 minutes of flying is a sanity check, not a benchmark. The flight is the
  same in every repeat, so the intervals show **radar luck only**, not how other flights would differ. The
  file comes from an incident extract: it is raw sensor data and says nothing about the cause of the accident,
  and nothing here should be read as saying anything about it. A busier, multi-aircraft file would be the next step.

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
- **A one-in-a-hundred event is not rare over many scans.** The 99% gate lets about 1% of true plots
  fall outside their own track. That plot started a second track on the same aircraft, which then
  shared its plots for the rest of the flight, so one aircraft showed up as two tracks. Only a test
  with two aircraft and no clutter exposed it. The fixes were to never start a track from a plot
  inside an existing gate and to merge tracks whose position and velocity agree.
- **Bounded outputs are not the same as a stable system.** The first learned-gain design kept every
  output inside sensible limits and still diverged to 10^27, because a feedback loop (the gain acts on
  its own past errors) can be unstable for gains that each look reasonable. The fix was to use a gain
  structure that is provably stable (alpha-beta) and to test the whole range numerically.
- **A plausible idea can fail its own test.** "Keep the physics and learn only the gain, so it
  generalises better" is a sensible hypothesis. The stress test said no, so the README says no.
- **Real data answers a different question than simulation.** The ranking of trackers survived, but the
  best `q` changed from 100 to 10 because the real flight is gentler than the invented maneuvers. A tuning
  chosen on a simulator is a hypothesis about the real world, not a fact about it.
- **Say what the interval does not cover.** The real-flight intervals are narrow, but they only cover
  radar randomness, because the flight itself never changes. A narrow interval is not the same as a general result.
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
| hybrid | the GRU starts from a Kalman estimate and sees Kalman outputs as inputs (the default; `--no-hybrid` turns it off) |
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
| clutter | false alarms: plots that come from no aircraft |
| data association | deciding which plot belongs to which track |
| gating | only considering plots statistically close to a track's prediction |
| Mahalanobis distance | distance measured in standard deviations, using the covariance (squared: `y^T S^-1 y`) |
| Hungarian algorithm | finds the one-to-one pairing of tracks and plots with the smallest total cost |
| greedy (nearest-neighbour) | each track takes its nearest plot in turn; simple, but can force bad pairings |
| tentative track | a new track not yet trusted; started from one unclaimed plot |
| confirmed track | a track that earned M hits in N scans and is reported |
| M-of-N confirmation | the rule "M hits within N scans" that keeps false alarms out of the answer |
| coasting | predicting without a plot while a track misses scans |
| ghost track | a reported track that matches no real aircraft |
| learned gain (KalmanNet style) | a Kalman filter whose gain `K` is chosen by a small network each scan, with the motion model fixed |
| alpha-beta filter | a filter with one position gain (alpha) and one velocity gain (beta) per axis; stable for alpha in (0, 1) and small positive beta |
| steady-state gain | the gain a Kalman filter settles to after many scans with the same noise |
| generalisation | how well a method works on conditions it was not trained or tuned on |
| ADS-B | a signal aircraft broadcast with their own GPS position, received by ground stations |
| OpenSky Network | a volunteer network that records ADS-B and shares the data for research |
| icao24 | the unique 24-bit address that identifies one aircraft in the signal |
| equirectangular projection | a simple map projection: metres north = latitude difference, metres east = longitude difference times the cosine of the latitude |
| resampling | computing values on a regular time grid (here 1 s) from irregular reports, by interpolation |
| segment | an unbroken stretch of a flight, with no gap longer than 5 seconds |
| OSPA | a distance between a set of true positions and a set of tracks, counting wrong positions and the wrong count |
| localisation / cardinality | the two parts of OSPA: how far paired tracks are off, and the cost of the wrong number of tracks |

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

<details><summary><b>26.</b> Why is a plot not simply given to the nearest track? What do gating and the Hungarian algorithm add?</summary>

Gating throws out pairs that are statistically implausible (squared Mahalanobis distance above 9.21 for
99% coverage), so false alarms and far-away plots are not forced onto a track. The Hungarian algorithm
then picks the one-to-one pairing with the smallest total cost, where nearest-first (greedy) can make an
early cheap choice that forces a terrible one later (total cost 9 against 4 in the section 3.8 example).
</details>

<details><summary><b>27.</b> Why must a new track be "confirmed" before it is reported?</summary>

A single false plot is enough to start a track. Real aircraft keep producing plots near the prediction,
so they collect 3 hits in 5 scans; false alarms almost never do. Tentative tracks that cannot get there
are dropped, so clutter stays out of the answer.
</details>

<details><summary><b>28.</b> A track predicts (0, 0) with S = 400 on the diagonal. Is a plot at (70, 0) inside the 99% gate?</summary>

d^2 = 70^2 / 400 = 12.25, above 9.21, so it is outside the gate. A plot at (30, 0) gives 2.25 and is inside.
</details>

<details><summary><b>29.</b> Two true aircraft, one track 100 m from the first, none for the second. What is the OSPA with c = 1000 m, p = 2?</summary>

Averaged over the larger set (2): total = sqrt((100^2 + 1000^2) / 2) = 710.6 m. Localisation is
sqrt(100^2 / 2) = 70.7 m and cardinality is sqrt(1000^2 / 2) = 707.1 m, and 70.7^2 + 707.1^2 = 710.6^2.
</details>

<details><summary><b>30.</b> Why is a plain RMSE not enough to score a multi-target tracker?</summary>

RMSE needs one estimate per truth in a known order. A multi-target tracker returns a set of tracks of
unknown size and order, and it can miss aircraft or invent ghosts. OSPA pairs the sets optimally and
also charges for the wrong count.
</details>

<details><summary><b>31.</b> Why did the first tracker show one aircraft as two tracks, and what fixed it?</summary>

About 1% of true plots fall outside their own track's 99% gate. That plot started a new tentative track
on the same aircraft, and the two tracks then shared its plots. Fixes: no new track from a plot inside any
track's gate, and merge tracks whose position and velocity agree.
</details>

<details><summary><b>32.</b> How does the learned-gain filter differ from the hybrid GRU?</summary>

The hybrid GRU outputs a free-form correction to a Kalman position. The learned-gain filter keeps the
Kalman structure (predict, then add `K` times the innovation) and the network only chooses the gain `K`.
The physics is fixed and only one thing is learned.
</details>

<details><summary><b>33.</b> Why was the gain limited to an alpha-beta form with alpha in (0, 1) and beta in (0, 0.5)?</summary>

A freely chosen 4x2 gain diverged with random weights, even though every number was bounded: the gain feeds
back on the filter's own errors, so some gains make it unstable. The alpha-beta form is stable for every
gain in that range (checked numerically on a grid), so training cannot push the filter into instability.
</details>

<details><summary><b>34.</b> Predicted position 1,200 m, plot 1,260 m, alpha = 0.282, beta = 0.0466. What are the new position and the velocity change?</summary>

Position: 1,200 + 0.282 x 60 = 1,216.9 m. Velocity estimate: + 0.0466 x 60 = + 2.8 m/s.
</details>

<details><summary><b>35.</b> The learned-gain filter beat the best single Kalman filter in four scenarios, yet it is not a success. Why?</summary>

It lost to the IMM and the GRU in every scenario, and on a radar noisier than any it trained on it was worse than
the plain q=100 Kalman filter. The hope that learning only the gain would generalise better was not supported.
</details>

<details><summary><b>36.</b> The aircraft's own ADS-B position is already very accurate. Why add radar noise to it?</summary>

We want to test trackers on what a radar sees, not on GPS. The real flight is the truth (real, human-flown
maneuvers) and the simulated radar adds range and bearing noise and missed plots. Only the flying is real.
</details>

<details><summary><b>37.</b> Why is a flight cut wherever two reports are more than 5 seconds apart?</summary>

Interpolating across a long gap would draw a straight line the aircraft may not have flown, inventing
truth. Cutting there keeps every resampled point within a few seconds of a real report. Pieces that are too
short (under 120 s) or too slow (taxiing) are dropped for the same reason of keeping real flying only.
</details>

<details><summary><b>38.</b> The intervals in section 7.7 are narrow. Does that mean the result holds for other flights?</summary>

No. The flight is identical in every repeat, so the interval only measures radar luck (placement and noise).
It says nothing about how a different aircraft would behave. That needs more flights.
</details>

<details><summary><b>39.</b> Why might q=10 beat q=100 on the real flight when q=100 won in the simulation?</summary>

The simulator's random maneuvers include sudden accelerations and turns, so a large q (an agile filter) fits.
The real flight climbs, cruises and turns gently, so a calmer filter averages the noise better. This is a
likely reason, not a tested one.
</details>

<details><summary><b>40.</b> What is the distance in metres of one degree of longitude at 30 degrees north, using a mean Earth radius of 6,371 km?</summary>

One degree of latitude is 111,195 m. One degree of longitude is that times cos(30 degrees) = 0.866, about 96,300 m.
</details>

## 11. Roadmap and further reading

**Where the project could go next** (the IMM filter, multi-target tracking and the learned gain are done, see sections 3.7, 3.8, 3.9 and 7)
- **Harder multi-target cases:** crossing and very close aircraft, an IMM inside every track, and
  track-to-track swaps scored explicitly.
- **A better learned gain:** the first version (section 3.9) did not generalise (section 7.6). Ideas:
  train on a wider noise range, also learn the process noise, or give it the IMM's mode probabilities.
- **More real data:** many aircraft at once (to test the multi-target tracker on real traffic), and real radar
  plots instead of ADS-B plus simulated noise. The single-flight result in section 7.7 is only a first check.
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
- Data: Schaefer, Strohmeier, Lenders, Martinovic and Wilhelm (2014), *Bringing up OpenSky: a large-scale
  ADS-B sensor network for research*, and https://opensky-network.org (data under CC BY 4.0).

## 12. Status

- [x] Step 1: simulator, Kalman baseline, tests
- [x] Step 2: neural tracker (GRU in PyTorch) trained on simulated tracks
- [x] Step 3: comparison by scenario (`run_compare.py`, `run_eval.py`)
- [x] Step 4: browser demo with a traditional radar (PPI) view (`web/`)
- [x] Rigorous evaluation: many seeds, confidence intervals, paired comparison, NEES and NIS
- [x] IMM filter as a stronger classical baseline (it beat the GRU in every scenario)
- [x] Multi-target tracking with clutter (gating, Hungarian assignment, M-of-N track management, OSPA)
- [x] KalmanNet-style learned gain (works, beats a single Kalman filter in 4 of 5 scenarios, but loses to the IMM and the GRU and did not generalise better)
- [x] Real ADS-B flight data (loader, evaluation and a first result on one real flight; the IMM is still best)
- [x] Web demo: a replay of a recorded training run, and a Random lab that runs the Kalman filters and the trained GRU live in the browser (checked against Python, section 3.11)
