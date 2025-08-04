# Robot Control Playground (FR3 Edition)

This project provides a simple playground for experimenting with control
algorithms on the **Franka Research 3 (FR3)** robot and other toy
systems.  It uses **MuJoCo** for physics simulation and exposes
extensible interfaces so that you can swap in other simulators (for
example, Genesis or Pinocchio) or write your own controllers.  The code
is intentionally kept modular and easy to read so that it can be used as
the basis for blog posts or teaching material.

The repository is organised into several packages:

- `simulators/` contains wrappers for MuJoCo and Pinocchio.  These
  classes load models (MJCF or URDF) and provide methods to step the
  simulation, compute forward and inverse dynamics, or modify physical
  parameters on the fly.
- `controllers/` implements a few basic control laws.  At the moment
  there is a simple Cartesian impedance controller which can be used to
  command an end–effector to a desired pose while compensating for
  gravity and dynamics.  Additional controllers (PID, admittance,
  operational–space, etc.) can be added under this package.
- `models/` stores robot descriptions.  A placeholder URDF for the FR3
  robot (`fr3.urdf`) is expected here.  You should replace this file
  with an actual URDF or MJCF description of the FR3 robot and any
  associated mesh resources.  For testing, a minimal mass–spring–damper
  model is also provided.
- `experiments/` contains example scripts that demonstrate how to
  combine the simulators and controllers.  See
  `experiments/fr3_impedance.py` for a simple impedance–control
  experiment on the FR3.

Before running any experiments you will need to install the dependencies
listed in `pixi.toml` or simply run `pip install mujoco pinocchio
mujoco-sysid numpy matplotlib` in your environment.  To run an
experiment, navigate into `experiments/` and execute the script with
Python.

```bash
cd experiments
python fr3_impedance.py
```

This will load the FR3 model, instantiate a Cartesian impedance
controller, and simulate the robot moving toward a target position.

Feel free to extend the code by adding more sophisticated controllers,
additional simulation backends, or alternative robots.  Pull requests
are welcome!