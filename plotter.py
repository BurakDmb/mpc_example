
class Plotter:
    def __init__(self, controller_name):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.name = controller_name + '_'
        import os
        if not os.path.exists('results'):
            os.makedirs('results')

    def show(self):
        self.plt.show(block=True)


    def plot_only_specific_element(self,
                                   c1_env1,
                                   axis=0, save_plot=False):
        # Plotting the simulation recorded
        fig, ax1 = self.plt.subplots(1, 1)
        fig.set_size_inches(16, 8)
        self.plt.tight_layout(pad=5)

        legend = ['Ref', 'PID']
        ylabels = ['Phi', 'Phidot', 'Theta', 'Thetadot', 'Psi', 'Psidot']
        ylabel = ylabels[axis]

        # Axis1
        # Ref and PID
        ax1.plot(c1_env1.t, c1_env1.history.sol_ref[axis, :].T)
        ax1.plot(c1_env1.t, c1_env1.history.sol_x[axis, :].T)


        ax1.set_xlabel('Time(second)')
        ax1.set_ylabel(ylabel + ' Value')
        ax1.legend(legend, shadow=True)
        ax1.set_title('Deterministic and Linear Quadcopter')

        if save_plot:
            self.plt.savefig('results/' + self.name +
                             'plot_only_specific_element.png')


    def plot_reward(self, env1, save_plot=False):
        # Plotting the simulation recorded
        fig, ax1 = self.plt.subplots(1, 1)
        fig.set_size_inches(16, 8)
        self.plt.tight_layout(pad=5)
        ax1.plot(env1.t, env1.history.sol_reward[:].T)
        ax1.set_xlabel('Time(second)')
        ax1.set_ylabel('Reward')
        ax1.legend(['Reward'], shadow=True)
        ax1.set_ylim((-1.1, 0.1))
        ax1.set_title('Deterministic Linear Quadcopter Reward Values wrt t')

        if save_plot:
            self.plt.savefig('results/' + self.name + 'plot_reward.png')

    # KEEP
    def plot_actions(self, env1, save_plot=False):
        # Plotting the simulation recorded
        fig, ax1 = self.plt.subplots(1, 1)
        fig.set_size_inches(16, 8)
        self.plt.tight_layout(pad=5)
        ax1.plot(env1.t, env1.history.sol_actions[:].T)
        ax1.set_xlabel('Time(second)')
        ax1.set_ylabel('Action(Torque)')
        ax1.legend(['U1', 'U2', 'U3'], shadow=True)
        ax1.set_title('Deterministic Linear Quadcopter - ' +
                      'Action(Torque) wrt t')

        if save_plot:
            self.plt.savefig('results/' + self.name + 'plot_actions.png')


    def plot_all_with_reference(self, env1, save_plot=False):
        # Plotting the simulation recorded
        fig, ax1 = self.plt.subplots(1, 1)
        fig.set_size_inches(16, 8)
        self.plt.tight_layout(pad=5)
        ax1.plot(env1.t, env1.history.sol_ref[:].T)
        ax1.plot(env1.t, env1.history.sol_x[:].T)
        ax1.set_xlabel('Time(second)')
        ax1.set_ylabel('Attitude and Angular Rate Values')
        ax1.legend(['Ref phi(rad)', 'Ref phidot(rad/s)', 'Ref theta(rad)',
                    'Ref thetadot(rad/s)', 'Ref psi(rad)', 'Ref psidot(rad/s)',
                    'phi(rad)', 'phidot(rad/s)', 'theta(rad)',
                    'thetadot(rad/s)', 'psi(rad)', 'psidot(rad/s)'],
                   shadow=True)
        ax1.set_title('Deterministic Linear Quadcopter Attitude and ' +
                      'Angular Rate Values wrt t')

        if save_plot:
            self.plt.savefig('results/' + self.name +
                             'plot_all_with_reference.png')
