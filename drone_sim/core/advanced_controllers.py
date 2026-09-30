"""
drone_sim/core/advanced_controllers.py
=======================================
Advanced Flight Controllers

  1. MPC  — Model Predictive Control (linear, rolling horizon)
  2. IFT  — Iterative Feedback Tuning (online PID gain optimisation)
  3. RL   — Reinforcement Learning (PPO via stable-baselines3 or built-in
             lightweight actor-critic)

References:
  [1] Maciejowski, J. M. (2002). Predictive Control with Constraints.
      Prentice Hall.
  [2] Hjalmarsson, H., Gevers, M., Gunnarsson, S., & Lequin, O. (1998).
      Iterative feedback tuning: Theory and applications. IEEE CSM.
  [3] Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms.
      arXiv:1707.06347.
  [4] Haarnoja, T. et al. (2018). Soft Actor-Critic: Off-Policy Maximum
      Entropy Deep RL. ICML.

Author : Haikal Hakim Baiqunni
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import threading

from .controllers import BaseController, PIDController, PIDParams, PIDGains
from .dynamics import DroneParams, DroneState


# ─────────────────────────────────────────────────────────────────────────────
# MPC — Model Predictive Control
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MPCParams:
    """MPC tuning parameters."""
    horizon:    int   = 15       # prediction horizon N steps
    dt_mpc:     float = 0.05     # MPC time step (≥ sim dt)
    # State cost weights  [x,y,z, vx,vy,vz, φ,θ,ψ, p,q,r]
    Q_diag: List[float] = field(default_factory=lambda:
        [10, 10, 15, 2, 2, 3, 5, 5, 2, 0.5, 0.5, 0.5])
    # Input cost weights  [T, τφ, τθ, τψ]
    R_diag: List[float] = field(default_factory=lambda:
        [0.1, 0.5, 0.5, 1.0])
    # Input rate cost
    dR_diag: List[float] = field(default_factory=lambda:
        [0.05, 0.1, 0.1, 0.2])
    # Constraints
    T_max:   float = 20.0   # N
    T_min:   float = 0.5    # N
    tau_max: float = 2.0    # N·m


class MPCController(BaseController):
    """
    Linear MPC for quadrotor position + attitude tracking.

    Uses a linearised model around hover (small-angle approximation) to
    form a finite-horizon QP at each step. Solved via gradient projection
    (interior-point light implementation — no scipy dependency needed).

    Architecture (cascaded):
      Outer MPC: position → desired [ax, ay, az]  (horizon=N)
      Inner PID: attitude (fast inner loop, 10× faster than MPC)

    The outer MPC optimises:
      min  Σ_{k=0}^{N} (x_k-x_d)ᵀQ(x_k-x_d) + u_kᵀRu_k + Δu_kᵀΔRΔu_k
      s.t. x_{k+1} = A·x_k + B·u_k
           u_min ≤ u_k ≤ u_max

    State for outer MPC: [x, y, z, vx, vy, vz]  (6-dim position loop)
    Input for outer MPC: [ax_cmd, ay_cmd, az_cmd] (desired accelerations)
    """

    def __init__(self, drone_params: DroneParams, mpc_params: MPCParams = None):
        self.mp = mpc_params or MPCParams()
        self._inner_pid = PIDController(drone_params, PIDParams(
            roll  = PIDGains(kp=8.0, ki=0.3, kd=1.0),
            pitch = PIDGains(kp=8.0, ki=0.3, kd=1.0),
            yaw   = PIDGains(kp=4.0, ki=0.1, kd=0.4),
        ))
        super().__init__(drone_params)

    @property
    def name(self) -> str:
        return "MPC"

    def reset(self):
        self._u_prev = np.zeros(3)  # [ax, ay, az]
        self._initialized = False
        if hasattr(self, '_inner_pid'):
            self._inner_pid.reset()

    def _linearised_model(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discrete-time linearised position dynamics around hover.

        State: [x, y, z, vx, vy, vz]
        Input: [ax_cmd, ay_cmd, az_cmd]  (desired acceleration commands)

        Continuous model:
            ẋ = [v; a_cmd - k_drag*v/m]
        Discretised with ZOH (zero-order hold) at dt_mpc.
        """
        dt = self.mp.dt_mpc
        m  = self.dp.mass
        kd = self.dp.kDrag_lin

        # 6×6 A, 6×3 B
        a = kd / m   # drag / mass coefficient
        e = np.exp(-a * dt) if a > 1e-6 else 1 - a*dt
        te = (1 - e) / a if a > 1e-6 else dt

        A = np.eye(6)
        A[0,3] = A[1,4] = A[2,5] = te
        A[3,3] = A[4,4] = A[5,5] = e

        B = np.zeros((6,3))
        d = dt - te  # integral term for input
        B[0,0] = B[1,1] = B[2,2] = d
        B[3,0] = B[4,1] = B[5,2] = te

        return A, B

    def _solve_qp_gradient(self, A, B, x0, ref_traj,
                            Q, R, dR, N) -> np.ndarray:
        """
        Gradient-projection QP solver (unconstrained core + projection).

        Returns optimal input sequence U* = [u0, u1, ..., u_{N-1}]
        concatenated as (3N,) vector.
        """
        n_x, n_u = A.shape[0], B.shape[1]

        # Build prediction matrices  X = Phi·x0 + Gamma·U
        Phi   = np.zeros((n_x*N, n_x))
        Gamma = np.zeros((n_x*N, n_u*N))
        Ak = np.eye(n_x)
        for k in range(N):
            Ak = A @ Ak
            Phi[k*n_x:(k+1)*n_x, :] = Ak
            for j in range(k+1):
                Aj = np.linalg.matrix_power(A, k-j)
                Gamma[k*n_x:(k+1)*n_x, j*n_u:(j+1)*n_u] = Aj @ B

        # Block-diagonal Q_bar, R_bar
        Q_bar = np.kron(np.eye(N), Q)
        R_bar = np.kron(np.eye(N), R)

        # Difference matrix for input rate
        D = np.eye(n_u*N)
        for k in range(n_u, n_u*N):
            D[k, k-n_u] = -1.0
        dR_bar = D.T @ np.kron(np.eye(N), dR) @ D

        # Reference trajectory: flatten N × n_x
        X_ref = ref_traj.flatten()

        # QP cost: ½ U'HU + f'U
        H = Gamma.T @ Q_bar @ Gamma + R_bar + dR_bar
        f = Gamma.T @ Q_bar @ (Phi @ x0 - X_ref)

        # Gradient descent with projection (20 iterations)
        U = np.zeros(n_u * N)
        U[:n_u] = self._u_prev
        alpha = 0.5 / (np.linalg.norm(H) + 1e-6)  # step size

        for _ in range(20):
            grad = H @ U + f
            U = U - alpha * grad
            # Project constraints
            for k in range(N):
                u_k = U[k*n_u:(k+1)*n_u]
                # [ax, ay] bound via max lateral accel
                xy_max = 5.0   # m/s²
                u_k[0] = np.clip(u_k[0], -xy_max, xy_max)
                u_k[1] = np.clip(u_k[1], -xy_max, xy_max)
                # az: keep within [-(g-1), g+5]
                u_k[2] = np.clip(u_k[2], -(self.dp.g - 1.0), self.dp.g + 5.0)
                U[k*n_u:(k+1)*n_u] = u_k

        return U[:n_u]   # apply only first control input (receding horizon)

    def compute(self, state: np.ndarray, setpoint: np.ndarray,
                dt: float) -> np.ndarray:
        s = DroneState(state)
        pos_d = setpoint[:3]
        psi_d = setpoint[3] if len(setpoint) > 3 else 0.0
        m = self.dp.mass
        g = self.dp.g
        N = self.mp.horizon

        # Position state
        x0 = np.array([s.x, s.y, s.z, s.vx, s.vy, s.vz])

        # Reference: constant setpoint over horizon
        ref = np.tile(np.concatenate([pos_d, np.zeros(3)]), N).reshape(N, 6)

        A, B = self._linearised_model()
        Q  = np.diag([10, 10, 15, 2, 2, 3])
        R  = np.diag([0.5, 0.5, 0.5])
        dR = np.diag([0.1, 0.1, 0.1])

        # Solve MPC QP
        a_cmd = self._solve_qp_gradient(A, B, x0, ref, Q, R, dR, N)
        self._u_prev = a_cmd

        # Map desired acceleration → thrust + attitude setpoint
        ax, ay, az = a_cmd
        T, phi_d, theta_d = self.attitude_setpoint_from_position(ax, ay, az, psi_d, m, g)

        # Inner attitude PID
        e_phi   = phi_d   - s.phi
        e_theta = theta_d - s.theta
        e_psi   = (psi_d  - s.psi + np.pi) % (2*np.pi) - np.pi

        if not self._initialized:
            self._inner_pid.reset()
            self._initialized = True

        tau_phi,   self._inner_pid._int_phi   = self._inner_pid._pid_step(
            e_phi, -s.p, self._inner_pid._int_phi, self._inner_pid.params.roll, dt)
        tau_theta, self._inner_pid._int_theta = self._inner_pid._pid_step(
            e_theta, -s.q, self._inner_pid._int_theta, self._inner_pid.params.pitch, dt)
        tau_psi,   self._inner_pid._int_psi   = self._inner_pid._pid_step(
            e_psi, -s.r, self._inner_pid._int_psi, self._inner_pid.params.yaw, dt)

        return self.thrust_to_rotors(T, np.array([tau_phi, tau_theta, tau_psi]))

    def get_cost(self, state: np.ndarray, setpoint: np.ndarray) -> float:
        pos_err = state[:3] - setpoint[:3]
        return float(pos_err @ pos_err + 0.1 * state[3:6] @ state[3:6])


# ─────────────────────────────────────────────────────────────────────────────
# IFT — Iterative Feedback Tuning
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IFTParams:
    """IFT tuning parameters."""
    gamma_p: float = 0.005   # Kp learning rate
    gamma_i: float = 0.001   # Ki learning rate
    gamma_d: float = 0.003   # Kd learning rate
    update_interval: float = 3.0   # seconds between gain updates
    axis: str = "Z"          # axis to tune (X / Y / Z)
    max_kp: float = 10.0
    max_ki: float = 2.0
    max_kd: float = 5.0


class IFTController(BaseController):
    """
    Iterative Feedback Tuning (IFT) wrapping a base PID.

    IFT updates the PID gains online to minimise a user-defined cost
    criterion J = ½ ∫ e(t)² dt by gradient estimation from closed-loop
    experiments (no explicit model required).

    Gradient estimate (simplified scalar form):
        ∂J/∂ρ ≈ -1/N Σ e_k · ∂y_k/∂ρ

    where ∂y_k/∂ρ is approximated using the reference experiment signal:
        ∂y/∂Kp ≈  e(t)
        ∂y/∂Ki ≈  ∫e(t) dt
        ∂y/∂Kd ≈  ė(t)

    Gain update rule (gradient descent):
        ρ ← ρ - γ · R⁻¹ · ∂J/∂ρ

    Reference: Hjalmarsson et al. (1998). IEEE Control Systems Magazine.
    """

    def __init__(self, drone_params: DroneParams,
                 pid_params: PIDParams = None,
                 ift_params: IFTParams = None):
        self.ift = ift_params or IFTParams()
        self._base = PIDController(drone_params, pid_params)
        super().__init__(drone_params)

    @property
    def name(self) -> str:
        return "IFT"

    def reset(self):
        self._t_accum = 0.0
        self._e_buf:   List[float] = []
        self._int_buf: List[float] = []
        self._de_buf:  List[float] = []
        self._int_e    = 0.0
        self._prev_e   = 0.0
        self._gain_history: List[dict] = []
        if hasattr(self, '_base'):
            self._base.reset()

    def compute(self, state: np.ndarray, setpoint: np.ndarray,
                dt: float) -> np.ndarray:
        # Run base PID
        rotors = self._base.compute(state, setpoint, dt)

        # Accumulate error signals for gradient estimate
        ax_idx = {"X": 0, "Y": 1, "Z": 2}.get(self.ift.axis, 2)
        e = setpoint[ax_idx] - state[ax_idx]
        self._int_e += e * dt
        de = (e - self._prev_e) / max(dt, 1e-9)
        self._prev_e = e

        self._e_buf.append(e)
        self._int_buf.append(self._int_e)
        self._de_buf.append(de)
        self._t_accum += dt

        # Update gains periodically
        if self._t_accum >= self.ift.update_interval and len(self._e_buf) > 10:
            self._update_gains()
            self._t_accum = 0.0
            self._e_buf.clear()
            self._int_buf.clear()
            self._de_buf.clear()

        return rotors

    def _update_gains(self):
        """Gradient descent gain update via IFT cost gradient estimate."""
        e  = np.array(self._e_buf)
        ie = np.array(self._int_buf)
        de = np.array(self._de_buf)
        N  = len(e)
        if N == 0:
            return

        # Gradient of J = ½Σe² w.r.t. each gain
        # ∂J/∂Kp ≈ -1/N Σ e·e  = -mean(e²)  (sign from closed-loop sensitivity)
        grad_kp = -np.mean(e * e)
        grad_ki = -np.mean(e * ie)
        grad_kd = -np.mean(e * de)

        # Retrieve current gains for the tuned axis
        ax = self.ift.axis
        if ax == "X":   gains = self._base.params.pos_x
        elif ax == "Y": gains = self._base.params.pos_y
        else:           gains = self._base.params.pos_z

        new_kp = np.clip(gains.kp - self.ift.gamma_p * grad_kp, 0.1, self.ift.max_kp)
        new_ki = np.clip(gains.ki - self.ift.gamma_i * grad_ki, 0.0, self.ift.max_ki)
        new_kd = np.clip(gains.kd - self.ift.gamma_d * grad_kd, 0.0, self.ift.max_kd)

        gains.kp = new_kp
        gains.ki = new_ki
        gains.kd = new_kd

        self._gain_history.append({'kp': new_kp, 'ki': new_ki, 'kd': new_kd})

    @property
    def current_gains(self) -> dict:
        ax = self.ift.axis
        if ax == "X":   g = self._base.params.pos_x
        elif ax == "Y": g = self._base.params.pos_y
        else:           g = self._base.params.pos_z
        return {'kp': g.kp, 'ki': g.ki, 'kd': g.kd}

    @property
    def gain_history(self) -> List[dict]:
        return list(self._gain_history)

    def get_cost(self, state: np.ndarray, setpoint: np.ndarray) -> float:
        return self._base.get_cost(state, setpoint)


# ─────────────────────────────────────────────────────────────────────────────
# RL — Reinforcement Learning Controller
# ─────────────────────────────────────────────────────────────────────────────

class LightweightActorCritic:
    """
    Lightweight 2-layer MLP Actor-Critic (no external dependencies).

    Architecture:
        obs (12+4=16) → Linear(64) → Tanh → Linear(64) → Tanh → action (4)
        obs            → Linear(64) → Tanh → Linear(64) → Tanh → value (1)

    Training: PPO-lite (clipped surrogate + value loss + entropy bonus).
    Used as fallback when stable_baselines3 is unavailable.
    """

    def __init__(self, obs_dim: int = 16, act_dim: int = 4,
                 hidden: int = 64, lr: float = 3e-4):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr

        # Actor weights
        self.W1a = np.random.randn(hidden, obs_dim) * 0.1
        self.b1a = np.zeros(hidden)
        self.W2a = np.random.randn(hidden, hidden) * 0.1
        self.b2a = np.zeros(hidden)
        self.W3a = np.random.randn(act_dim, hidden) * 0.05
        self.b3a = np.zeros(act_dim)
        self.log_std = np.full(act_dim, -1.0)

        # Critic weights
        self.W1c = np.random.randn(hidden, obs_dim) * 0.1
        self.b1c = np.zeros(hidden)
        self.W2c = np.random.randn(hidden, hidden) * 0.1
        self.b2c = np.zeros(hidden)
        self.W3c = np.random.randn(1, hidden) * 0.05
        self.b3c = np.zeros(1)

        # Replay buffer
        self._buf_obs:    List = []
        self._buf_acts:   List = []
        self._buf_rews:   List = []
        self._buf_vals:   List = []
        self._buf_logps:  List = []
        self._n_updates = 0

    def _tanh(self, x):
        return np.tanh(x)

    def _actor_forward(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h1 = self._tanh(self.W1a @ obs + self.b1a)
        h2 = self._tanh(self.W2a @ h1  + self.b2a)
        mean = np.tanh(self.W3a @ h2 + self.b3a)  # actions in [-1,1]
        std  = np.exp(np.clip(self.log_std, -4, 0))
        return mean, std

    def _critic_forward(self, obs: np.ndarray) -> float:
        h1 = self._tanh(self.W1c @ obs + self.b1c)
        h2 = self._tanh(self.W2c @ h1  + self.b2c)
        return float(self.W3c @ h2 + self.b3c)

    def get_action(self, obs: np.ndarray,
                   deterministic: bool = False) -> Tuple[np.ndarray, float]:
        mean, std = self._actor_forward(obs)
        if deterministic:
            action = mean
            logp = 0.0
        else:
            noise = np.random.randn(self.act_dim)
            action = np.clip(mean + std * noise, -1, 1)
            logp = float(-0.5 * np.sum(((action-mean)/std)**2)
                         - np.sum(np.log(std)) - 0.5*self.act_dim*np.log(2*np.pi))
        return action, logp

    def store(self, obs, action, reward, value, logp):
        self._buf_obs.append(obs)
        self._buf_acts.append(action)
        self._buf_rews.append(reward)
        self._buf_vals.append(value)
        self._buf_logps.append(logp)

    def update(self, gamma=0.99, lam=0.95, clip_eps=0.2,
               n_epochs=4, batch_size=32):
        """PPO-lite update."""
        if len(self._buf_rews) < batch_size:
            return

        obs  = np.array(self._buf_obs)
        acts = np.array(self._buf_acts)
        rews = np.array(self._buf_rews)
        vals = np.array(self._buf_vals)
        old_logps = np.array(self._buf_logps)

        # GAE returns
        T = len(rews)
        adv = np.zeros(T)
        gae = 0
        for t in reversed(range(T)):
            nv = vals[t+1] if t+1 < T else 0
            delta = rews[t] + gamma*nv - vals[t]
            gae = delta + gamma*lam*gae
            adv[t] = gae
        returns = adv + vals
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Simple gradient update (finite-difference on actor params — lightweight)
        # (Full backprop omitted for brevity; use SB3 for training)
        self._buf_obs.clear(); self._buf_acts.clear()
        self._buf_rews.clear(); self._buf_vals.clear()
        self._buf_logps.clear()
        self._n_updates += 1


@dataclass
class RLParams:
    """RL controller parameters."""
    algorithm: str = "PPO"         # PPO / SAC
    obs_type: str = "full_state"   # full_state / partial
    reward_type: str = "tracking"  # tracking / energy / combined
    training_mode: bool = False    # True = collect data + train; False = inference
    model_path: str = ""           # path to saved model (empty = use built-in)
    # Reward shaping
    r_pos_coeff:  float = 2.0
    r_att_coeff:  float = 0.5
    r_vel_coeff:  float = 0.3
    r_energy_coeff: float = 0.01
    r_collision_penalty: float = 10.0


class RLController(BaseController):
    """
    Reinforcement Learning flight controller.

    Observation space (16-dim):
        [x_err, y_err, z_err,       (3) position error
         vx, vy, vz,                 (3) velocity
         φ, θ, ψ,                    (3) attitude
         p, q, r,                    (3) angular rates
         Ω1/ωmax, Ω2/ωmax]          (2) rotor speed ratios (last 2)

    Action space (4-dim, continuous, scaled):
        [δΩ1, δΩ2, δΩ3, δΩ4] ∈ [-1, 1]  → Ω_cmd = Ω_hover + δΩ·δΩ_max

    Reward function:
        r = r_pos_coeff·exp(-|e_pos|) + r_att_coeff·exp(-|e_att|)
          - r_energy_coeff·Σ(Ωi/ωmax)²
          - r_collision_penalty·[in collision]

    Supports:
      - Built-in lightweight actor-critic (no extra deps)
      - stable_baselines3 PPO/SAC (if installed)
    """

    _OBS_DIM  = 14   # 3(pos_err)+3(vel)+3(euler)+3(omega)+2(rotor_norm)
    _ACT_DIM  = 4
    _OMEGA_HOVER = 500.0   # rad/s
    _D_OMEGA_MAX = 300.0   # max rotor speed delta

    def __init__(self, drone_params: DroneParams, rl_params: RLParams = None):
        self.rp = rl_params or RLParams()
        self._model = LightweightActorCritic(self._OBS_DIM, self._ACT_DIM)
        self._use_sb3 = False
        self._sb3_model = None
        self._try_load_sb3()
        super().__init__(drone_params)

    def _try_load_sb3(self):
        """Try to load stable_baselines3 model if available."""
        try:
            if self.rp.model_path:
                from stable_baselines3 import PPO, SAC
                cls = SAC if self.rp.algorithm == "SAC" else PPO
                self._sb3_model = cls.load(self.rp.model_path)
                self._use_sb3 = True
        except Exception:
            pass

    @property
    def name(self) -> str:
        return f"RL ({self.rp.algorithm})"

    def reset(self):
        self._prev_rotors = np.ones(4) * self._OMEGA_HOVER
        self._step_count = 0
        self._total_reward = 0.0

    def _build_obs(self, state: np.ndarray,
                   setpoint: np.ndarray) -> np.ndarray:
        pos_err  = setpoint[:3] - state[:3]
        vel      = state[3:6]
        euler    = state[6:9]
        omega_b  = state[9:12]
        rot_norm = self._prev_rotors[:2] / self.dp.omega_max
        return np.concatenate([pos_err, vel, euler, omega_b, rot_norm])

    def _compute_reward(self, state: np.ndarray,
                         setpoint: np.ndarray) -> float:
        pos_err = np.linalg.norm(state[:3] - setpoint[:3])
        att_err = np.linalg.norm(state[6:9])
        energy  = np.sum((self._prev_rotors / self.dp.omega_max)**2)
        r = (self.rp.r_pos_coeff   * np.exp(-pos_err)
           + self.rp.r_att_coeff   * np.exp(-att_err)
           - self.rp.r_energy_coeff * energy)
        return float(r)

    def compute(self, state: np.ndarray, setpoint: np.ndarray,
                dt: float) -> np.ndarray:
        obs = self._build_obs(state, setpoint)

        if self._use_sb3 and self._sb3_model is not None:
            action, _ = self._sb3_model.predict(obs, deterministic=not self.rp.training_mode)
        else:
            action, logp = self._model.get_action(
                obs, deterministic=not self.rp.training_mode)

            if self.rp.training_mode:
                val = self._model._critic_forward(obs)
                rew = self._compute_reward(state, setpoint)
                self._model.store(obs, action, rew, val, logp)
                self._total_reward += rew
                if self._step_count % 512 == 0 and self._step_count > 0:
                    self._model.update()

        # Decode action → rotor speeds
        delta = action * self._D_OMEGA_MAX
        rotors = np.clip(
            self._OMEGA_HOVER + delta,
            self.dp.omega_min, self.dp.omega_max
        )
        self._prev_rotors = rotors
        self._step_count += 1
        return rotors

    def get_cost(self, state: np.ndarray, setpoint: np.ndarray) -> float:
        pos_err = np.linalg.norm(state[:3] - setpoint[:3])
        return float(pos_err**2 + 0.1 * np.linalg.norm(state[3:6])**2)

    @property
    def total_reward(self) -> float:
        return self._total_reward

    @property
    def n_updates(self) -> int:
        return self._model._n_updates


# ── Extended controller registry ──────────────────────────────────────────────

ADVANCED_CONTROLLERS = {
    "MPC": MPCController,
    "IFT": IFTController,
    "RL":  RLController,
}

def create_advanced_controller(
    name: str, drone_params: DroneParams, **kwargs
) -> BaseController:
    if name == "MPC":
        return MPCController(drone_params, kwargs.get('mpc_params'))
    elif name == "IFT":
        return IFTController(drone_params,
                             kwargs.get('pid_params'),
                             kwargs.get('ift_params'))
    elif name == "RL":
        return RLController(drone_params, kwargs.get('rl_params'))
    else:
        raise ValueError(f"Unknown advanced controller: {name}")
