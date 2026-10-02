#!/usr/bin/env python3
"""
main.py — DRONSIM Launcher
===========================
Usage:
    python main.py            # Launch the GUI
    python main.py --batch    # Compare controllers headless (no GUI)
    python main.py --test     # Self-test: dynamics, controllers, estimator, safety
"""

import argparse
import os
import sys
import time

# The `drone_sim` package is this folder, so its parent must be importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows consoles default to cp1252; avoid UnicodeEncodeError on symbols.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def check_dependencies(need_gui: bool = False):
    """Fail early with an actionable message instead of a bare traceback."""
    missing = []
    for mod in ("numpy", "scipy", "matplotlib"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if need_gui:
        try:
            import tkinter  # noqa: F401
        except ImportError:
            missing.append("tkinter (reinstall Python with the 'tcl/tk' option)")
    if missing:
        req = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
        print("DRONSIM cannot start: missing " + ", ".join(missing) + ".\n")
        print(f"This Python is:  {sys.executable}\n")
        print("Install into THIS interpreter with:")
        print(f'    "{sys.executable}" -m pip install -r "{req}"\n')
        print("If you have several Pythons installed (for example the Microsoft Store one and\n"
              "python.org), make sure your editor uses the one you installed the packages into.")
        sys.exit(1)


# ── GUI ──────────────────────────────────────────────────────────────────────

def launch_gui():
    check_dependencies(need_gui=True)
    from drone_sim.gui.app import DroneSimApp
    DroneSimApp().run()


# ── Batch comparison ─────────────────────────────────────────────────────────

def run_batch_test():
    """Fly every controller on hover and circle missions and print a table."""
    check_dependencies()
    from drone_sim.core.catalog import CATALOG, default_values
    from drone_sim.core.compare import run_comparison
    from drone_sim.core.dynamics import DroneParams
    from drone_sim.core.environment import EnvironmentParams, WindCondition
    from drone_sim.core.mission import MissionSpec

    keys = [k for k in CATALOG if k != "RL"]
    env = EnvironmentParams(wind_condition=WindCondition.MODERATE, turbulence_enabled=True,
                            turbulence_intensity=0.2, noise_enabled=False)
    values = {k: default_values(k) for k in keys}
    print("=" * 66)
    print("  DRONSIM — batch controller comparison")
    print("  moderate wind, turbulence, sensor fusion, 15 s")
    print("=" * 66)
    for preset in ("hover", "circle"):
        spec = MissionSpec(kind="preset", preset=preset, duration=15.0)
        print(f"\nMission: {preset}")
        print(f"{'Controller':<18}{'RMSE (m)':>10}{'Max (m)':>10}{'Tilt (deg)':>12}{'Result':>10}")
        print("-" * 60)
        res = run_comparison(spec, keys, values, DroneParams(), env, True, [], 15.0)
        for r in sorted(res, key=lambda r: (r.crashed, r.rmse)):
            print(f"{r.label:<18}{r.rmse:>10.3f}{r.max_error:>10.3f}{r.max_tilt:>12.0f}"
                  f"{'crashed' if r.crashed else 'flew':>10}")


# ── Self-test ────────────────────────────────────────────────────────────────

def run_tests():
    """Smoke tests that exercise the physics, controllers, estimator and safety layers."""
    check_dependencies()
    import numpy as np
    from drone_sim.core.catalog import build_controller
    from drone_sim.core.dynamics import DroneParams, QuadrotorDynamics
    from drone_sim.core.environment import EnvironmentParams, WindCondition
    from drone_sim.core.mission import MissionSpec, Waypoint, build_simulation
    from drone_sim.core.obstacles import BoxObstacle
    from drone_sim.core.simulation import ScheduledEvent

    dp = DroneParams()
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))

    def fly(key, spec, env, fused=True, events=None):
        b = build_simulation(spec, build_controller(key, dp), dp, env, fused, events or [], real_time=False)
        b.sim.start()
        while b.sim.is_running:
            time.sleep(0.01)
        return b.sim, b

    print("Dynamics")
    dyn = QuadrotorDynamics(dp)
    hover = np.sqrt(dp.mass * dp.g / (4 * dp.kT))
    state = np.zeros(12)
    state[2] = 2.0
    dx = dyn.derivatives(0, state, np.ones(4) * hover)
    check("hover thrust cancels gravity", abs(dx[5]) < 0.1, f"z-acc {dx[5]:.4f} m/s²")
    for _ in range(10):
        state = dyn.step(state, np.ones(4) * hover, 0.01)
    check("hover holds altitude for 0.1 s", abs(state[2] - 2.0) < 0.01, f"z {state[2]:.4f} m")

    calm = EnvironmentParams(wind_condition=WindCondition.CALM, noise_enabled=False)
    windy = EnvironmentParams(wind_condition=WindCondition.MODERATE, turbulence_enabled=True,
                              noise_enabled=True)
    hold = MissionSpec(kind="preset", preset="hover", duration=10.0)

    print("Controllers")
    np.random.seed(1234)       # same noise draws on every run and every CI image
    for key in ("PID", "SMC", "PID+SMC", "ADRC", "GEO", "MPC"):
        sim, _ = fly(key, hold, calm, fused=False)
        d = sim.logger.get_all()
        err = np.linalg.norm([d["x"][-1], d["y"][-1], d["z"][-1] - 2.0])
        check(f"{key} reaches the hover point in calm air", err < 0.35 and not sim.metrics.is_crashed,
              f"final error {err:.2f} m")
    sim, _ = fly("SMC", hold, windy, fused=False)
    check("SMC survives raw sensor noise", not sim.metrics.is_crashed)

    print("Sensing")
    np.random.seed(1234)       # same noise draws on every run and every CI image
    sim, _ = fly("ADRC", hold, windy, fused=True)
    check("Kalman filter beats raw GPS noise", sim.metrics.est_error < 0.2,
          f"position error {sim.metrics.est_error:.3f} m")

    print("Disturbances")
    np.random.seed(1234)       # same noise draws on every run and every CI image
    ev = [ScheduledEvent(t=4.0, kind="rotor_fault", rotor=1, effectiveness=0.5)]
    sim, _ = fly("ADRC", hold, windy, fused=True, events=ev)
    d = sim.logger.get_all()
    check("ADRC keeps flying after a 50% rotor loss", not sim.metrics.is_crashed and d["z"][-1] > 1.5,
          f"final altitude {d['z'][-1]:.2f} m")
    check("rotor fault is logged as an event", len(sim.event_log) == 1)

    print("Obstacle avoidance")
    np.random.seed(1234)       # same noise draws on every run and every CI image
    obs = [BoxObstacle(cx=1.5, cy=0, cz=1.5, lx=1, ly=1.5, lz=3)]
    wps = [Waypoint(0, 0, 0.1), Waypoint(0, 0, 2), Waypoint(3, 0, 2), Waypoint(3, 3, 2)]
    for algo, must_clear in (("None", False), ("RRT*+CBF", True)):
        spec = MissionSpec(kind="waypoints", waypoints=wps, obstacles=obs, duration=25.0)
        spec.avoid.algorithm = algo
        sim, _ = fly("ADRC", spec, calm, fused=True)
        d = sim.logger.get_all()
        goal_err = np.linalg.norm([d["x"][-1] - 3, d["y"][-1] - 3, d["z"][-1] - 2])
        clear = d["clearance"].min()
        if must_clear:
            check("RRT*+CBF reaches the goal without touching the obstacle",
                  clear > 0.2 and goal_err < 0.6, f"clearance {clear:.2f} m, goal error {goal_err:.2f} m")
        else:
            check("without avoidance the drone hits the obstacle (sanity check)", clear < 0.05,
                  f"clearance {clear:.2f} m")

    print("Swarm")
    np.random.seed(1234)       # same noise draws on every run and every CI image
    from drone_sim.core.safety import CBFParams, CBFSafetyFilter
    from drone_sim.core.swarm import SingleDroneSim, WaypointSpec
    starts, goals = [(-2, 0), (2, 0), (0, 2)], [(2, 0), (-2, 0), (0, -2)]
    for sep in (False, True):
        drones = []
        for i, (a, g) in enumerate(zip(starts, goals)):
            wp = [WaypointSpec(a[0], a[1], 2.0, 0, 0), WaypointSpec(g[0], g[1], 2.0, 0, 0)]
            d = SingleDroneSim(i, wp, dp, calm, "ADRC", 0.01,
                               controller=build_controller("ADRC", dp), use_estimator=True)
            d._state[0:3] = (a[0], a[1], 2.0)
            drones.append(d)
        for d in drones:
            d.peers = [q for q in drones if q is not d]
            if sep:
                kp, kd = d._controller.position_gains
                d.separation_filter = CBFSafetyFilter([], CBFParams(influence=2.0, nominal_kp=kp, nominal_kd=kd))
        closest = 9.0
        for _ in range(1500):
            for d in drones:
                d.step()
            closest = min(closest, min(np.linalg.norm(p.state[:3] - q.state[:3])
                                       for p in drones for q in drones if p is not q))
        goal_err = max(np.linalg.norm(d.state[:2] - np.array(g)) for d, g in zip(drones, goals))
        if sep:
            check("swarm separation keeps head-on drones apart and on mission",
                  closest > 0.25 and goal_err < 0.6, f"closest {closest:.2f} m, goal error {goal_err:.2f} m (per drone "
                  + ", ".join(f"{np.linalg.norm(d.state[:2] - np.array(g)):.2f}" for d, g in zip(drones, goals)) + ")")
        else:
            check("without separation head-on drones nearly collide (sanity check)", closest < 0.3,
                  f"closest {closest:.2f} m")

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRONSIM — quadrotor simulation platform")
    parser.add_argument("--batch", action="store_true", help="compare controllers without the GUI")
    parser.add_argument("--test", action="store_true", help="run the self-test")
    args = parser.parse_args()
    if args.batch:
        run_batch_test()
    elif args.test:
        run_tests()
    else:
        launch_gui()
