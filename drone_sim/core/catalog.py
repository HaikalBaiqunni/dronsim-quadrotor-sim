"""
drone_sim/core/catalog.py
==========================
Controller catalogue: one place that knows every controller, its family,
its tunable parameters (with ranges, so the GUI can build sliders) and how to
build it from a plain ``{key: value}`` dictionary.

The GUI, the batch runner and the comparison tab all go through
``build_controller`` so they stay in sync.

Author : Haikal Hakim Baiqunni
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .dynamics import DroneParams
from .controllers import (PIDController, SMController, HybridPIDSMC,
                          PIDParams, PIDGains, SMCParams, BaseController)
from .advanced_controllers import (MPCController, MPCParams, IFTController,
                                   IFTParams, RLController, RLParams)
from .research_controllers import (ADRCController, ADRCParams,
                                   GeometricController, GeoParams)


@dataclass
class Param:
    key: str
    label: str
    default: float
    lo: float = 0.0
    hi: float = 10.0
    step: float = 0.1
    level: str = "basic"          # 'basic' | 'adv'
    group: str = ""
    unit: str = ""
    hint: str = ""
    kind: str = "float"           # 'float' | 'int' | 'choice' | 'bool'
    options: tuple = ()


@dataclass
class ControllerSpec:
    key: str
    label: str
    family: str
    blurb: str
    params: List[Param] = field(default_factory=list)
    new: bool = False             # added in v2.0 (research methods)
    experimental: bool = False    # works, but not expected to fly well


FAMILIES = ["Classic", "Predictive", "Adaptive", "Geometric", "Learning"]


def _pid_params(prefix_defaults: bool = True) -> List[Param]:
    g1, g2, g3 = "Horizontal position (X, Y)", "Altitude (Z)", "Roll and pitch"
    return [
        Param("xy_kp", "Kp", 1.2, 0, 8, 0.05, "basic", g1, hint="Stiffness of horizontal position loop"),
        Param("xy_kd", "Kd", 0.8, 0, 5, 0.05, "basic", g1, hint="Damping of horizontal position loop"),
        Param("xy_ki", "Ki", 0.0, 0, 2, 0.02, "adv", g1),
        Param("z_kp", "Kp", 2.0, 0, 10, 0.05, "basic", g2),
        Param("z_kd", "Kd", 1.0, 0, 6, 0.05, "basic", g2),
        Param("z_ki", "Ki", 0.5, 0, 3, 0.02, "adv", g2),
        Param("att_kp", "Kp", 6.0, 0, 30, 0.1, "basic", g3),
        Param("att_kd", "Kd", 0.8, 0, 5, 0.05, "basic", g3),
        Param("att_ki", "Ki", 0.2, 0, 2, 0.02, "adv", g3),
        Param("yaw_kp", "Kp", 3.0, 0, 20, 0.1, "adv", "Yaw"),
        Param("yaw_kd", "Kd", 0.3, 0, 3, 0.05, "adv", "Yaw"),
        Param("yaw_ki", "Ki", 0.1, 0, 2, 0.02, "adv", "Yaw"),
    ]


def _smc_params(attitude_only: bool = False) -> List[Param]:
    out = []
    if not attitude_only:
        out += [
            Param("lam_pos", "λ position", 1.5, 0.1, 6, 0.1, "basic", "Position surface", hint="Sliding-surface slope"),
            Param("eta_pos", "η position", 2.0, 0.1, 10, 0.1, "basic", "Position surface", hint="Reaching gain"),
            Param("phi_pos", "Boundary layer", 0.5, 0.05, 2, 0.05, "adv", "Position surface", "m"),
            Param("dmax_pos", "Disturbance bound", 1.0, 0, 5, 0.1, "adv", "Position surface", "m/s²"),
        ]
    out += [
        Param("lam_att", "λ attitude", 3.0, 0.1, 12, 0.1, "basic", "Attitude surface", hint="Sliding-surface slope"),
        Param("eta_att", "η attitude", 4.0, 0.1, 15, 0.1, "basic", "Attitude surface", hint="Reaching gain"),
        Param("phi_att", "Boundary layer", 0.3, 0.05, 2, 0.05, "adv", "Attitude surface", "rad/s"),
        Param("dmax_att", "Disturbance bound", 0.5, 0, 5, 0.1, "adv", "Attitude surface"),
    ]
    return out


CATALOG: Dict[str, ControllerSpec] = {}


def _add(spec: ControllerSpec):
    CATALOG[spec.key] = spec


_add(ControllerSpec("PID", "PID", "Classic",
                    "Cascaded position and attitude PID. Simple and predictable.",
                    _pid_params()))
_add(ControllerSpec("SMC", "Sliding mode", "Classic",
                    "Robust to disturbances, boundary layer avoids chattering.",
                    _smc_params()))
_add(ControllerSpec("PID+SMC", "PID + SMC hybrid", "Classic",
                    "PID position loop with a sliding-mode attitude loop.",
                    _pid_params() + _smc_params(attitude_only=True)))
_add(ControllerSpec("MPC", "MPC", "Predictive",
                    "Linear model predictive control over a receding horizon.", [
    Param("horizon", "Horizon steps", 15, 5, 40, 1, "basic", "Prediction", kind="int",
          hint="Longer looks further ahead but costs more CPU"),
    Param("dt_mpc", "MPC step", 0.05, 0.02, 0.2, 0.01, "basic", "Prediction", "s"),
    Param("q_pos", "Position weight", 10.0, 1, 60, 1, "basic", "Weights", hint="Higher tracks position harder"),
    Param("q_att", "Attitude weight", 5.0, 0.5, 30, 0.5, "adv", "Weights"),
    Param("r_scale", "Input cost scale", 1.0, 0.1, 10, 0.1, "adv", "Weights"),
    Param("t_max", "Max thrust", 20.0, 5, 40, 1, "adv", "Limits", "N"),
    Param("tau_max", "Max torque", 2.0, 0.2, 5, 0.1, "adv", "Limits", "N·m"),
]))
_add(ControllerSpec("ADRC", "ADRC", "Adaptive",
                    "Observers estimate and cancel wind, drag and actuator loss on every axis.", [
    Param("wc_pos", "Controller bandwidth", 2.2, 0.5, 6, 0.1, "basic", "Position", "rad/s", "Higher reacts faster"),
    Param("wo_pos", "Observer bandwidth", 9.0, 3, 25, 0.5, "basic", "Position", "rad/s", "Keep 3 to 5 times the controller"),
    Param("wc_att", "Controller bandwidth", 14.0, 4, 30, 0.5, "basic", "Attitude", "rad/s"),
    Param("wo_att", "Observer bandwidth", 40.0, 10, 70, 1, "basic", "Attitude", "rad/s"),
    Param("wc_yaw", "Yaw controller", 3.0, 1, 10, 0.5, "adv", "Yaw", "rad/s"),
    Param("wo_yaw", "Yaw observer", 12.0, 3, 30, 1, "adv", "Yaw", "rad/s"),
    Param("max_tilt", "Max tilt", 35.0, 10, 60, 1, "adv", "Limits", "deg"),
    Param("dist_limit", "Disturbance clamp", 12.0, 2, 30, 1, "adv", "Limits", "m/s²"),
], new=True))
_add(ControllerSpec("IFT", "IFT auto-tuning", "Adaptive",
                    "PID whose gains are optimised online from closed-loop runs.",
                    [Param("axis", "Axis to tune", 2, kind="choice", options=("X", "Y", "Z"),
                           group="Tuning", level="basic"),
                     Param("update_interval", "Update every", 3.0, 1, 10, 0.5, "basic", "Tuning", "s"),
                     Param("gamma_p", "Kp learning rate", 0.005, 0.0005, 0.02, 0.0005, "adv", "Learning rates"),
                     Param("gamma_i", "Ki learning rate", 0.001, 0.0001, 0.01, 0.0001, "adv", "Learning rates"),
                     Param("gamma_d", "Kd learning rate", 0.003, 0.0003, 0.02, 0.0003, "adv", "Learning rates")]
                    + _pid_params()))
_add(ControllerSpec("GEO", "Geometric SE(3)", "Geometric",
                    "Attitude error on the rotation group. No Euler singularity, handles large tilts.", [
    Param("wn_pos", "Position frequency", 2.2, 0.5, 6, 0.1, "basic", "Position", "rad/s"),
    Param("zeta_pos", "Position damping", 0.9, 0.3, 2, 0.05, "basic", "Position"),
    Param("wn_att", "Attitude frequency", 14.0, 4, 30, 0.5, "basic", "Attitude", "rad/s"),
    Param("zeta_att", "Attitude damping", 1.0, 0.3, 2, 0.05, "basic", "Attitude"),
    Param("ki_pos", "Integral gain", 0.0, 0, 3, 0.05, "adv", "Position", "1/s²"),
    Param("max_tilt", "Max tilt", 40.0, 10, 70, 1, "adv", "Limits", "deg"),
], new=True))
_add(ControllerSpec("RL", "RL (PPO)", "Learning",
                    "Experimental. The built-in actor-critic starts untrained, so expect crashes: "
                    "it needs far more flight time than one run to learn anything useful.", [
    Param("algorithm", "Algorithm", 0, kind="choice", options=("PPO", "SAC"), group="Policy"),
    Param("training", "Keep learning", 1, kind="bool", group="Policy",
          hint="Off freezes the policy (inference only)"),
    Param("r_pos", "Position reward", 2.0, 0, 10, 0.1, "adv", "Reward shaping"),
    Param("r_att", "Attitude reward", 0.5, 0, 5, 0.1, "adv", "Reward shaping"),
    Param("r_vel", "Velocity reward", 0.3, 0, 5, 0.1, "adv", "Reward shaping"),
    Param("r_energy", "Energy penalty", 0.01, 0, 0.5, 0.01, "adv", "Reward shaping"),
], experimental=True))


def default_values(key: str) -> Dict[str, float]:
    return {p.key: p.default for p in CATALOG[key].params}


def _pid_from(v: dict) -> PIDParams:
    def g(pfx, **fallback):
        return PIDGains(kp=v.get(f"{pfx}_kp", fallback["kp"]),
                        ki=v.get(f"{pfx}_ki", fallback["ki"]),
                        kd=v.get(f"{pfx}_kd", fallback["kd"]))
    xy = g("xy", kp=1.2, ki=0.0, kd=0.8)
    z = g("z", kp=2.0, ki=0.5, kd=1.0)
    att = g("att", kp=6.0, ki=0.2, kd=0.8)
    yaw = g("yaw", kp=3.0, ki=0.1, kd=0.3)
    return PIDParams(pos_x=xy, pos_y=PIDGains(xy.kp, xy.ki, xy.kd), pos_z=z,
                     roll=att, pitch=PIDGains(att.kp, att.ki, att.kd), yaw=yaw)


def _smc_from(v: dict) -> SMCParams:
    return SMCParams(lambda_pos=v.get("lam_pos", 1.5), eta_pos=v.get("eta_pos", 2.0),
                     phi_pos=v.get("phi_pos", 0.5), Dmax_pos=v.get("dmax_pos", 1.0),
                     lambda_att=v.get("lam_att", 3.0), eta_att=v.get("eta_att", 4.0),
                     phi_att=v.get("phi_att", 0.3), Dmax_att=v.get("dmax_att", 0.5))


def build_controller(key: str, dp: DroneParams, values: Optional[dict] = None) -> BaseController:
    """Instantiate controller ``key`` using ``values`` (missing keys use defaults)."""
    v = default_values(key)
    v.update(values or {})
    if key == "PID":
        return PIDController(dp, _pid_from(v))
    if key == "SMC":
        return SMController(dp, _smc_from(v))
    if key == "PID+SMC":
        return HybridPIDSMC(dp, _pid_from(v), _smc_from(v))
    if key == "MPC":
        q = float(v["q_pos"])
        qa = float(v["q_att"])
        rs = float(v["r_scale"])
        mp = MPCParams(horizon=int(v["horizon"]), dt_mpc=float(v["dt_mpc"]),
                       Q_diag=[q, q, 1.5 * q, 2, 2, 3, qa, qa, 2, 0.5, 0.5, 0.5],
                       R_diag=[0.1 * rs, 0.5 * rs, 0.5 * rs, 1.0 * rs],
                       T_max=float(v["t_max"]), tau_max=float(v["tau_max"]))
        return MPCController(dp, mp)
    if key == "ADRC":
        return ADRCController(dp, ADRCParams(
            wc_pos=v["wc_pos"], wo_pos=v["wo_pos"], wc_att=v["wc_att"], wo_att=v["wo_att"],
            wc_yaw=v["wc_yaw"], wo_yaw=v["wo_yaw"], max_tilt_deg=v["max_tilt"],
            dist_limit=v["dist_limit"]))
    if key == "IFT":
        axis = ("X", "Y", "Z")[int(v["axis"])]
        return IFTController(dp, _pid_from(v), IFTParams(
            gamma_p=v["gamma_p"], gamma_i=v["gamma_i"], gamma_d=v["gamma_d"],
            update_interval=v["update_interval"], axis=axis))
    if key == "GEO":
        return GeometricController(dp, GeoParams(
            wn_pos=v["wn_pos"], zeta_pos=v["zeta_pos"], wn_att=v["wn_att"],
            zeta_att=v["zeta_att"], ki_pos=v["ki_pos"], max_tilt_deg=v["max_tilt"]))
    if key == "RL":
        return RLController(dp, RLParams(
            algorithm=("PPO", "SAC")[int(v["algorithm"])], training_mode=bool(v["training"]),
            r_pos_coeff=v["r_pos"], r_att_coeff=v["r_att"], r_vel_coeff=v["r_vel"],
            r_energy_coeff=v["r_energy"]))
    raise ValueError(f"Unknown controller: {key}")


def by_family() -> Dict[str, List[ControllerSpec]]:
    out: Dict[str, List[ControllerSpec]] = {f: [] for f in FAMILIES}
    for s in CATALOG.values():
        out[s.family].append(s)
    return out
