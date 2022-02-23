# Import do_mpc package:
import do_mpc
import numpy as np

# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
sys.path.append('../../')


# Dynamics taken from
# https://github.com/winstxnhdw/KinematicBicycleModel/blob/main/kinematic_model.py#L58
class Nonlinear_MPC:
    def __init__(self, t_end, n_horizon, c_step, s_step, env):
        # Model Paremeters

        # Creating the Model
        model_type = 'continuous'  # either 'discrete' or 'continuous'
        model = do_mpc.model.Model(model_type)

        # Model Variables
        x = model.set_variable(
            var_type='_x', var_name='x', shape=(1, 1))
        y = model.set_variable(
            var_type='_x', var_name='y', shape=(1, 1))
        yaw = model.set_variable(
            var_type='_x', var_name='yaw', shape=(1, 1))
        velocity = model.set_variable(
            var_type='_x', var_name='velocity', shape=(1, 1))

        throttle = model.set_variable(
            var_type='_u', var_name='throttle', shape=(1, 1))

        steering_angle = model.set_variable(
            var_type='_u', var_name='steering_angle', shape=(1, 1))

        if env is None:
            wheelbase = 2.96
        else:
            wheelbase = env.wheelbase

        # # Right-hand-side equation
        model.set_rhs('x', velocity * np.cos(yaw))
        model.set_rhs('y', velocity * np.sin(yaw))
        model.set_rhs('yaw', velocity * np.tan(steering_angle) / wheelbase)
        model.set_rhs('velocity', throttle)

        # Model Setup
        model.setup()

        # Configuring the MPC controller
        self.mpc = do_mpc.controller.MPC(model)

        # Optimizer Parameters
        # Parametreleri incele.
        setup_mpc = {
            'n_horizon': n_horizon,
            't_step': c_step,
            'n_robust': 1,
            'store_full_solution': True,
            'nlpsol_opts': {'ipopt.print_level': 0,
                            'print_time': 0},
            # Uncomment for MA27 solver
            # 'nlpsol_opts': {'ipopt.print_level': 0, 'ipopt.sb': 'yes',
            #                 'print_time': 0, 'ipopt.linear_solver': 'MA27'},
            # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'},

        }
        self.mpc.set_param(**setup_mpc)

        mterm = (
            x**2 + y**2 + yaw**2 + 0.1*(velocity**2)
        )

        lterm = (x**2 + y**2 + yaw**2 + 0.1*(velocity**2))

        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        # Penality for the Control Inputs
        self.mpc.set_rterm(
            throttle=0,
            steering_angle=0
        )

        # # Lower bounds on states:
        self.mpc.bounds['lower', '_x', 'x'] = -np.inf
        self.mpc.bounds['lower', '_x', 'y'] = -np.inf
        self.mpc.bounds['lower', '_x', 'yaw'] = -np.inf
        self.mpc.bounds['lower', '_x', 'velocity'] = -np.inf

        # # Upper bounds on states
        self.mpc.bounds['upper', '_x', 'x'] = np.inf
        self.mpc.bounds['upper', '_x', 'y'] = np.inf
        self.mpc.bounds['upper', '_x', 'yaw'] = np.inf
        self.mpc.bounds['upper', '_x', 'velocity'] = np.inf

        # Lower bounds on inputs:
        self.mpc.bounds['lower', '_u', 'throttle'] = -np.inf
        self.mpc.bounds['lower', '_u', 'steering_angle'] = -np.inf

        # Upper bounds on inputs:
        self.mpc.bounds['upper', '_u', 'throttle'] = np.inf
        self.mpc.bounds['upper', '_u', 'steering_angle'] = np.inf

        # Setup
        self.mpc.setup()

        # Configuring the Simulator
        self.simulator = do_mpc.simulator.Simulator(model)

        # Simulator parameters
        # Instead of supplying a dict with the splat operator (**),
        # as with the optimizer.set_param(),
        # we can also use keywords (and call the method
        # multiple times, if necessary):
        self.simulator.set_param(t_step=s_step)

        # Setup
        self.simulator.setup()

        # Creating the control loop
        # self.x0 = env.dynamics_state.reshape(-1, 1)
        self.x0 = np.array([1, 0, 0, 0])
        # Use the x0 property to set the initial state

        self.simulator.x0 = self.x0
        self.mpc.x0 = self.x0

        # Set the initial guess of the MPC optimization problem
        self.mpc.set_initial_guess()

        self.simulator.reset_history()
        self.simulator.x0 = self.x0
        self.mpc.reset_history()

    def predict(self, error, deterministic=True):

        u0 = -self.mpc.make_step(error)
        self.x0 = self.simulator.make_step(u0)
        return u0.reshape((2,))

    def test_mpc(self):
        u0 = np.zeros((2, 1))
        u0[0] = 0.1
        u0[1] = 0.5
        for i in range(250):
            self.simulator.make_step(u0)

        self.plot()

    def plot(self):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        # Customizing Matplotlib:
        mpl.rcParams['font.size'] = 18
        mpl.rcParams['lines.linewidth'] = 3
        mpl.rcParams['axes.grid'] = True
        mpc_graphics = do_mpc.graphics.Graphics(self.mpc.data)
        sim_graphics = do_mpc.graphics.Graphics(self.simulator.data)
        fig, ax = plt.subplots(2, sharex=True, figsize=(16, 9))
        fig.align_ylabels()
        for g in [sim_graphics, mpc_graphics]:
            g.add_line(var_type='_x', var_name='x', axis=ax[0])
            g.add_line(var_type='_x', var_name='y', axis=ax[0])
            g.add_line(var_type='_x', var_name='yaw', axis=ax[0])
            g.add_line(var_type='_x', var_name='velocity', axis=ax[0])

            g.add_line(var_type='_u', var_name='throttle', axis=ax[1])
            g.add_line(var_type='_u', var_name='steering_angle', axis=ax[1])

        ax[0].set_ylabel('angle position [rad]')
        ax[1].set_ylabel('motor angle [rad]')
        ax[1].set_xlabel('time [s]')

        sim_graphics.plot_results()
        # Reset the limits on all axes in graphic to show the data.
        sim_graphics.reset_axes()
        plt.show()


if __name__ == "__main__":
    n_horizon = 20
    control_freq = 50
    simulation_freq = 250
    t_end = 5
    nonlinear_mpc = Nonlinear_MPC(t_end=5,
                                  n_horizon=n_horizon,
                                  c_step=1/control_freq,
                                  s_step=1/simulation_freq,
                                  env=None)
    nonlinear_mpc.test_mpc()
