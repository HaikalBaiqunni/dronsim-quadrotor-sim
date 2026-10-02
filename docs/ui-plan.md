# UI v3 plan

The Tkinter GUI in `drone_sim/gui/` works and passes a clipping audit, but it is limited: matplotlib 3D is slow, and
styling is hand-built. The plan is a web-style UI on top of the unchanged `drone_sim/core/`.

## Candidate stack

| Option | Notes |
|---|---|
| **NiceGUI** (preferred) | Quasar components, built-in three.js scene (`ui.scene`), pure Python, runs in a browser or a native window |
| Rerun | Robotics viewer for replay and analysis, could be an extra tab |
| Dear PyGui | Very fast plots, no built-in 3D |
| Flet | Material look, weak 3D |

## Screens

1. **Guided setup**: stepper (Vehicle, Mission, Controller, World, Review). Controllers are picked from cards with tags and
   ratings for robustness, precision and CPU cost, plus a step-response preview. Ratings must be measured from simulation runs,
   not hand-written.
2. **Studio**: large 3D viewport, collapsible property cards (Controller, World, Events, Vehicle), timeline dock with event markers.
3. **Analysis**: run chips, error curves with event markers, side-by-side mini viewers, leaderboard, auto-generated insight cards,
   replay scrubber.

## Open questions

* Is `ui.scene` good enough for a drone model with spinning rotors and trails? Prototype the Studio screen first.
* Insight-card rules (for example "controller X error jumps after the fault") still need to be defined.
* Keep the Tkinter GUI until the new one is at parity.

## Status

* **Studio prototype** (`python -m drone_sim.web.studio`, needs `pip install nicegui`): answers the `ui.scene` question. It is good enough for a
  drone with spinning rotors, a point-cloud trail, obstacles, waypoints and a setpoint marker. Checked in headless Chromium at 1240x700
  with no clipping and no console errors. The drone model is drawn 2.5x real size so it stays readable.
  `drone_sim/web/session.py` holds the UI-independent run logic (no NiceGUI import).
* **Guided setup** (`/setup`): Goal, Controller and Review steps. Ratings are measured by `drone_sim/web/ratings.py` via `run_comparison`:
  a circle in calm air (tracking RMSE after a 5 s settle) and a hover in moderate wind with a 50% rotor loss at 8 s (crash or RMSE).
  CPU is wall time relative to the fastest. Robustness stars are absolute, tracking stars are relative to the best controller
  (the circle task is lag-dominated, so absolute limits gave everyone one star). Results are cached in `ratings_cache.json`
  (gitignored, keyed by a hash of `core/`). Measuring takes about a minute. The experimental RL controller is shown as not rated.
  Checked in headless Chromium at 1240x700: measure from the UI, sort by goal, pick a card, open in Studio with the controller carried over.
* **Analysis** (`/analysis`): fly several controllers on one scenario (trajectory, wind, turbulence, optional rotor fault and gust), then run chips,
  error curves with event markers, leaderboard (crashed runs last), insight cards and a replay scrubber that moves one marker per run in a 3D view.
  Insights are rules over the recorded data (`analysis.py`, thresholds are named constants): crash (time from the core's 80 deg tilt criterion),
  error jump after a fault or gust with recovery time, steady offset in the last 5 s, tightest tracker, lowest control effort.
  `drone_sim/web/test_analysis.py` checks the rules on synthetic runs and is part of CI. Units: tilt is in degrees (the logger stores degrees);
  "effort" is sum of rotor speed squared over time, a power proxy and not joules. Not built: side-by-side mini viewers per run (one shared 3D view instead).
* Not built yet: migrating off Tkinter. The Tkinter GUI is untouched. Config is shared between pages through a module-level object, so the
  app is single-user and local.
