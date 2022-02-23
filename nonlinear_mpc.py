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
