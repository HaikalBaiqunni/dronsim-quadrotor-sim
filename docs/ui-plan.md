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
* Not built yet: guided setup (measured ratings) and analysis screens. The Tkinter GUI is untouched.
