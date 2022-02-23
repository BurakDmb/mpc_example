import gym
from gym import spaces
import numpy as np
import math
from scipy.integrate import solve_ivp

wheelbase = 2.96


# state is x, y, yaw, velocity
# u1 is throttle, u2 is steering_angle
def nonlinear_vehicle_dynamics(t, state, u1=0, u2=0):
    velocitydot = (u1)
    xdot = (state[3] * math.cos(state[2]))
    ydot = (state[3] * math.sin(state[2]))
    yawdot = (state[3] * math.tan(u2) / wheelbase)

    dxdt = np.array([xdot, ydot, yawdot, velocitydot])

    return dxdt


class DeterministicVehicle(gym.Env):
    """
    Linear and Deterministic Quadcopter System Dynamics
    System Input(actions): U(U1, U2, U3) torque values
    System Output(states): X = phi, phidot, theta, thetadot, psi, psidot
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, solver_func, t_start=0, t_end=5, simulation_freq=250,
                 control_freq=50,
                 dynamics_state=np.array([0, 0, 0, 0]),
                 ):
        super(DeterministicVehicle, self).__init__()

        self.wheelbase = wheelbase

        self.solver_func = solver_func

        self.u_max = np.float32(np.array([np.inf, np.inf]))

        self.high = np.float32(np.array([np.inf,
                                        np.inf,
                                        np.inf,
                                        np.inf]))

        self.action_len = len(self.u_max)
        self.state_len = len(self.high)
        self.dynamics_len = len(self.high)
        self.action_space = spaces.Box(low=-self.u_max,
                                       high=self.u_max)
        self.observation_space = spaces.Box(low=-self.high,
                                            high=self.high)

        self.t_start = t_start
        self.t_end = t_end
        self.simulation_freq = simulation_freq
        self.simulation_timestep = 1/simulation_freq
        self.control_freq = control_freq
        self.control_timestep = 1/control_freq

        self.current_timestep = 0
        self.episode_timestep = 0
        self.current_time = 0
        self.initial_dynamics_state = dynamics_state
        self.dynamics_state = self.initial_dynamics_state

        self.reference_state = self.dynamics_state
        self.state = self.reference_state - self.dynamics_state

        self.rnd_state = np.random.default_rng(None)
        self.env_reset_flag = False
        self.keep_history = True

        if self.keep_history:
            self.history = History(self.__class__.__name__)
            self.history.sol_x = np.array([])
            self.history.sol_t = np.array([])
            self.history.sol_ref = np.array([])
            self.history.sol_reward = np.array([])
            self.history.sol_actions = np.array([])

    def step(self, action):
        # Execute one time step within the environment
        self.current_time = self.current_timestep * self.control_timestep
        time_range = (self.current_time,
                      self.current_time + self.control_timestep)

        action_clipped = np.maximum(np.minimum(action, self.u_max),
                                    -self.u_max)

        u1, u2 = action_clipped
        # w1s, w2s, w3s, w4s = self.motor_mixing(u1, u2, u3)

        # solve_ivp for integrating a initial value problem for system of ODEs.
        # By using the max_step parameter, the simulation is ensured
        # to have minimum 250Hz freq.
        sol = solve_ivp(self.solver_func, time_range,
                        self.dynamics_state,
                        # args=(w1s, w2s, w3s, w4s),
                        args=(u1, u2),
                        vectorized=True,
                        max_step=self.simulation_timestep)

        next_state = sol.y[:, -1]
        next_time = sol.t[-1]

        current_reference_diff = self.reference_state - next_state

        reward = -(
            (current_reference_diff[0]**2 +
             current_reference_diff[1]**2 +
             current_reference_diff[2]**2 +
             current_reference_diff[3]**2))

        if self.keep_history:
            self.history.sol_x = (np.column_stack((self.history.sol_x,
                                                   next_state))
                                  if self.history.sol_x.size else next_state)

            self.history.sol_t = (np.column_stack((self.history.sol_t,
                                                   next_time))
                                  if self.history.sol_t.size else next_time)

            self.history.sol_ref = (np.column_stack(
                (self.history.sol_ref, self.reference_state))
                if self.history.sol_ref.size else self.reference_state)

            self.history.sol_reward = (np.column_stack(
                (self.history.sol_reward, reward))
                if self.history.sol_reward.size else np.array([reward]))

            self.history.sol_actions = (np.column_stack(
                (self.history.sol_actions, action_clipped))
                if self.history.sol_actions.size else action_clipped)

            self.t = np.reshape(self.history.sol_t, -1)

        self.dynamics_state = next_state

        self.state = self.reference_state - self.dynamics_state

        self.current_timestep += 1
        self.episode_timestep += 1
        info = {"episode": None}

        done = False

        if (self.episode_timestep >= self.t_end*self.control_freq):
            done = True
            self.env_reset_flag = True
            if self.keep_history:
                self.t = np.reshape(self.history.sol_t, -1)
        return self.state, reward, done, info

    def reset(self):

        self.episode_timestep = 0
        if self.env_reset_flag:
            self.dynamics_state = \
                self.initial_dynamics_state
            self.env_reset_flag = False

        # Generate a random reference(in radians)
        self.reference_state = self.rnd_state.uniform(low=-1,
                                                      high=1,
                                                      size=4)

        self.reference_state[[0, 1, 2, 3]] = 0.0
        self.reference_state[[0]] = 1.0

        self.state = self.reference_state - self.dynamics_state

        return self.state


class History:
    def __init__(self, env_name):
        self.env_name = env_name
