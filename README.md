# Example package for an MPC example

For this project, quadcopter rotational dynamics (https://arxiv.org/abs/2202.07021) are modelled and rotational references are controlled with Model Predictive Control (DO-MPC Library was used for linear and nonlinear implementations.)

## Install Guide

```
conda create -n mpc python=3.8

conda activate mpc
conda install -c conda-forge libgfortran flake8 -y
pip install do-mpc scipy numpy gym
```

### [Optional] MA27 Solver Usage
After the installation, you can enable the use of MA27 nonlinear solver to speed up the MPC execution time. For this, please update your .bashrc(.zshrc in mac) by adding this line:

Note: Please correct the directory of ~/Documents/hsl/lib to your own configuration.
```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:~/Documents/hsl/lib"
```

To use the MA27 solver, please use uncomment code block in `linear_mpc.py` or `nonlinear_mpc.py`:

```
setup_mpc = {
            'n_horizon': n_horizon,
            't_step': c_step,
            'n_robust': 1,
            'store_full_solution': True,
            # Uncomment for MA27 solver
            # 'nlpsol_opts' : {'ipopt.linear_solver': 'MA27'},
            # 'nlpsol_opts': {'ipopt.print_level': 0, 'ipopt.sb': 'yes',
            #                 'print_time': 0, 'ipopt.linear_solver': 'MA27'},
            'nlpsol_opts': {'ipopt.print_level': 0},
        }
```
