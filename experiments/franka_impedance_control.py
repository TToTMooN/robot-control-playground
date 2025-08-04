from robot_control_playground.simulators.mujoco_sim import MujocoSim
from robot_control_playground.controllers.impedance_controller import CartesianImpedanceController
from robot_control_playground.simulators.pinocchio_model import PinocchioModel
import numpy as np
import pinocchio as pin

# Load models
sim = MujocoSim("models/franka_panda.xml")
pino = PinocchioModel("models/franka_panda.urdf")

# Desired end-effector target (3‑D position) and impedance gains
target_pos = np.array([0.5, 0.0, 0.5])
target_vel = np.zeros(3)
kp = np.array([50., 50., 50.])  # stiffness
kv = np.array([5., 5., 5.])     # damping

controller = CartesianImpedanceController(kp, kv, target_pos, target_vel)

# Simulation loop
sim.reset()
T = 5.0  # seconds
dt = sim.model.opt.timestep
steps = int(T / dt)
for i in range(steps):
    # Read current joint state from MuJoCo
    q = sim.data.qpos.copy()
    v = sim.data.qvel.copy()

    # Forward kinematics: get end-effector pose using MuJoCo functions or Pinocchio
    # (here we use Pinocchio for example)
    ee_frame = pino.model.getFrameId("panda_hand")
    pin.forwardKinematics(pino.model, pino.data, q, v, np.zeros_like(v))
    pin.updateFramePlacement(pino.model, pino.data, ee_frame)
    current_pos = pino.data.oMf[ee_frame].translation

    # Gravity compensation using Pinocchio inverse dynamics (zero desired acceleration)
    tau_g = pino.inverse_dynamics(q, v, np.zeros_like(v))

    # Compute control torques
    u = controller.compute(current_pos, np.zeros(3), gravity_torque=tau_g)

    # Step simulation
    sim.step(u)

    # log data, optionally plot or save
