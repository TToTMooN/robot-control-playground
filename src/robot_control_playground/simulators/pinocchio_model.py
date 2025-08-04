import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

class PinocchioModel:
    """Wrapper for Pinocchio kinematics/dynamics."""
    def __init__(self, urdf_path, mesh_dir=None):
        self.robot = RobotWrapper.BuildFromURDF(urdf_path, mesh_dir)
        self.model = self.robot.model
        self.data = self.robot.data

    def inverse_dynamics(self, q, v, a):
        """Compute torques using RNEA (gravity, Coriolis, inertial terms)."""
        return pin.rnea(self.model, self.data, q, v, a)

    def jacobian(self, q):
        """Compute the full joint Jacobian."""
        pin.computeJointJacobians(self.model, self.data, q)
        return pin.getJointJacobian(self.model, self.data, self.model.nq - 1, pin.ReferenceFrame.LOCAL)

    def mass_matrix(self, q):
        """Compute joint-space inertia matrix using CRBA."""
        return pin.crba(self.model, self.data, q)
