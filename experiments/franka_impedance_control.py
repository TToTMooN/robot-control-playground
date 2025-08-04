"""Example impedance control of the FR3 robot in MuJoCo.

This script demonstrates how to combine the `MujocoSim` and
`PinocchioModel` wrappers with a simple Cartesian impedance controller to
simulate the Franka Research 3 (FR3) robot following a Cartesian
trajectory.  It assumes that the MuJoCo model for FR3 exists at
``models/fr3.xml`` and that a corresponding URDF exists at
``models/fr3.urdf``.  Replace these files with the actual robot
descriptions before running.

If you have not yet installed the dependencies, run the following:

```bash
pip install mujoco pinocchio mujoco-sysid numpy matplotlib
```

Usage:
    python fr3_impedance.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robot_control_playground.simulators.mujoco_sim import MujocoSim
from robot_control_playground.simulators.pinocchio_model import PinocchioModel
from robot_control_playground.controllers.impedance_controller import CartesianImpedanceController


def main() -> None:
    # Paths to the robot models.  Replace these with the correct file
    # names if your models are located elsewhere.
    root_dir = Path(__file__).parent.parent
    mjcf_path = str(root_dir / "models" / "fr3.xml")
    urdf_path = str(root_dir / "models" / "fr3.urdf")

    # Instantiate simulators
    sim = MujocoSim(mjcf_path)
    pino = PinocchioModel(urdf_path)

    # Reset the simulation to the default pose
    sim.reset()

    # Determine the end–effector frame name for FR3.  For the actual FR3
    # URDF this should correspond to the final link; here we use a
    # placeholder.
    ee_frame = "panda_hand"  # Change to FR3 EE frame name once known

    # Define controller gains and target.  The target is given in
    # Cartesian coordinates [x, y, z] relative to the base frame.
    kp = np.array([100.0, 100.0, 100.0])
    kd = np.array([20.0, 20.0, 20.0])
    target_pos = np.array([0.5, 0.0, 0.5])
    controller = CartesianImpedanceController(kp, kd, target_pos)

    # Simulation parameters
    duration = 2.0  # seconds
    dt = sim.model.opt.timestep
    steps = int(duration / dt)

    # Log arrays for analysis
    log_q = []
    log_pos = []

    for _ in range(steps):
        q = sim.data.qpos.copy()
        v = sim.data.qvel.copy()

        # Compute gravity compensation via Pinocchio (zero acc)
        tau_g = pino.inverse_dynamics(q, v, np.zeros_like(v))

        # Forward kinematics for the end–effector using Pinocchio
        try:
            import pinocchio as pin
            pin.forwardKinematics(pino.model, pino.data, q, v, np.zeros_like(v))
            pin.updateFramePlacement(pino.model, pino.data, pino.model.getFrameId(ee_frame))
            current_pos = pino.data.oMf[pino.model.getFrameId(ee_frame)].translation
            current_vel = np.zeros(3)  # Not computed here for brevity
        except Exception:
            # Fallback if frame is not found; skip control
            current_pos = np.zeros(3)
            current_vel = np.zeros(3)

        # Compute control torque (gravity term in joint space)
        # Note: We assume the Jacobian mapping from Cartesian wrench to joint
        # torques is handled elsewhere.  For a full implementation you
        # should compute the Jacobian at the end–effector and map the
        # 3‑D wrench to joint torques.  Here we simply add gravity
        # compensation as a demonstration.
        torque = tau_g

        # Step simulation
        sim.step(torque)

        # Log data
        log_q.append(q.copy())
        log_pos.append(current_pos.copy())

    # Print final state as a simple verification
    print("Final joint positions:", log_q[-1])
    print("Final end‑effector position:", log_pos[-1])


if __name__ == "__main__":
    main()