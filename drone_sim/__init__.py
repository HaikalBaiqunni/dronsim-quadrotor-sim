# drone_sim/__init__.py
"""
DRONSIM — Quadrotor Drone Simulation Platform
==============================================
Packages:
    core.dynamics             6-DOF rigid body dynamics (RK4)
    core.environment          Wind, Dryden turbulence, ground effect, gusts
    core.controllers          PID, SMC, PID+SMC
    core.advanced_controllers MPC, IFT, RL
    core.research_controllers ADRC, geometric SE(3)
    core.estimation           IMU/GPS/baro simulation + Kalman filter
    core.safety               Control barrier function safety filter
    core.obstacles            Obstacles, APF, RRT*
    core.mission              Waypoints, planning, simulation factory
    core.compare              Headless controller comparison
    core.catalog              Controller catalogue + parameter specs
    core.simulation / swarm   Simulation engines
    gui                       Tkinter application
"""
__version__ = "2.0.0"
__author__ = "Haikal Hakim Baiqunni"
