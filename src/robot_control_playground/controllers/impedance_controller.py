import numpy as np

class CartesianImpedanceController:
    def __init__(self, kp, kv, target_pos, target_vel):
        self.kp = np.asarray(kp)
        self.kv = np.asarray(kv)
        self.target_pos = np.asarray(target_pos)
        self.target_vel = np.asarray(target_vel)

    def compute(self, current_pos, current_vel, gravity_torque=None):
        """Compute desired forces/torques based on impedance law.
        
        gravity_torque: feed-forward gravity compensation (from Pinocchio or MuJoCo inverse dynamics).
        """
        pos_error = self.target_pos - np.asarray(current_pos)
        vel_error = self.target_vel - np.asarray(current_vel)
        u = self.kp * pos_error + self.kv * vel_error
        if gravity_torque is not None:
            u += gravity_torque
        return u
