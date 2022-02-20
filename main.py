import numpy as np
import sys
import time

from quad import StochasticQuad

n_horizon = 20
control_freq = 50
simulation_freq = 250
t_end = 5


from linear_mpc import Linear_MPC
from quad import DeterministicQuad, linear_quad_dynamics
import time


def test_controller(controller, t_end, plot=False, save_plot=False,
                    constant_reference=None):

    if constant_reference is None:
        constant_reference = np.array([1, 0, 1, 0, 1, 0])
    custom_u_high = np.array([0.1, 0.1, 0.1])

    env1 =\
        createEnvs(t_end=t_end,
                   simulation_freq=simulation_freq,
                   control_freq=control_freq,
                   random_state_seed=0,
                   set_custom_u_limit=False,
                   custom_u_high=custom_u_high,
                   set_constant_reference=True,
                   constant_reference=constant_reference)

    num_episode = 1
    simulate_envs(controller, env1, num_episode)
    calculateControllerMetrics(env1)

    if plot:
        from plotter import Plotter
        plotter = Plotter(type(controller).__name__)
        plotter.plot_only_specific_element(env1,
                                           save_plot=save_plot)
        plotter.plot_only_specific_element(env1,
                                           save_plot=save_plot, axis=1)
        plotter.plot_all_with_reference(env1,
                                        save_plot=save_plot)
        plotter.plot_reward(env1, save_plot=save_plot)
        plotter.plot_actions(env1, save_plot=save_plot)
        if plot:
            plotter.show()
    return env1



def calculateControllerMetrics(env):
    resp_final = env.history.sol_x[0][-1]
    tol = 5e-2
    tl_ind = 0
    th_ind = 0
    ts_ind = 0
    tp = 0
    vl = 0.10*resp_final
    vh = 0.90*resp_final
    vs_l = 0.98*resp_final
    vs_h = 1.02*resp_final
    peak_val = 0
    for ind, val in enumerate(env.history.sol_x[0]):
        if abs(val-vl) < tol:
            tl_ind = ind
            break
    for ind, val in enumerate(env.history.sol_x[0]):
        if abs(val-vh) < tol:
            th_ind = ind
            break
    for ind, val in enumerate(env.history.sol_x[0]):
        if ts_ind != 0:
            if vs_l > val or vs_h < val:
                ts_ind = 0
        if vs_l < val and vs_h > val and ts_ind == 0:
            ts_ind = ind
        if val > peak_val:
            peak_val = val
            tp = ind
    rise_time = env.history.sol_t[0][th_ind]-env.history.sol_t[0][tl_ind]
    settling_time = env.history.sol_t[0][ts_ind]
    overshoot = (peak_val-resp_final)*100/resp_final
    peak = peak_val
    peak_time = env.history.sol_t[0][tp]
    ss_error = abs(env.reference_state[0] - resp_final)
    total_rew = env.history.sol_reward.sum()

    print("Env-Shape: ", env.t.shape, env.history.sol_ref.shape,
          env.history.sol_x.shape, env.history.sol_reward.shape)
    print("Rise time: %1.3f sec" % rise_time)
    print("Settling time: %1.3f sec" % settling_time)
    print("Overshoot: %2.3f percent" % overshoot)
    print("Peak time: %1.3f sec" % peak_time)
    print("Steady State Error: %2.3f rad" % ss_error)
    print("Total Reward: %2.3f" % total_rew)
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    return rise_time, settling_time, overshoot, peak, peak_time


def simulate_envs(controller, env1, num_episode):
    for epi in range(num_episode):
        obs = env1.reset()
        done = False
        while not done:
            action = controller.predict(obs, deterministic=True)
            if type(action) is tuple:
                action = action[0]
            obs, reward, done, _ = env1.step(action)


def createEnvs(t_end, simulation_freq,
               control_freq, random_state_seed,
               set_custom_u_limit,
               custom_u_high,
               set_constant_reference,
               constant_reference,
               dynamics_state=np.array([3.14/4, 0, 0, 0, 0, 0]),
               eval_env=True):
    from quad import DeterministicQuad
    from quad import linear_quad_dynamics

    # Linear deterministic quadcopter
    env1 = DeterministicQuad(linear_quad_dynamics, t_end=t_end,
                             simulation_freq=simulation_freq,
                             control_freq=control_freq, random_state_seed=0,
                             dynamics_state=dynamics_state,
                             set_custom_u_limit=set_custom_u_limit,
                             custom_u_high=custom_u_high,
                             set_constant_reference=True,
                             constant_reference=constant_reference,
                             eval_env=eval_env)

    return env1


def test_NonlinearMPC(plot=False, save_plot=False, loadmodel=False):
    from nonlinear_mpc import Nonlinear_MPC
    from quad import DeterministicQuad, linear_quad_dynamics
    import time

    env = DeterministicQuad(linear_quad_dynamics, t_end=t_end,
                            simulation_freq=250, control_freq=50,
                            keep_history=False)

    start = time.time()
    print("*** Function: ", sys._getframe().f_code.co_name, "***")
    nonlinear_mpc = Nonlinear_MPC(t_end=t_end,
                                  n_horizon=n_horizon,
                                  c_step=1/control_freq,
                                  s_step=1/simulation_freq,
                                  env=env)
    test_controller(nonlinear_mpc, t_end, plot=plot, save_plot=save_plot)
    end = time.time()
    print(end-start)


def test_LinearMPC(plot=False, save_plot=False, loadmodel=False):
    from linear_mpc import Linear_MPC
    import time
    from quad import DeterministicQuad, linear_quad_dynamics
    env = DeterministicQuad(linear_quad_dynamics, t_end=t_end,
                            simulation_freq=250, control_freq=50,
                            keep_history=False)

    start = time.time()
    print("*** Function: ", sys._getframe().f_code.co_name, "***")
    linear_mpc = Linear_MPC(t_end=t_end,
                            n_horizon=n_horizon,
                            c_step=1/control_freq,
                            s_step=1/simulation_freq,
                            env=env)
    test_controller(linear_mpc, t_end, plot=plot, save_plot=save_plot)
    end = time.time()
    print(end-start)


plot=True
test_LinearMPC(plot)
# test_NonlinearMPC(plot)