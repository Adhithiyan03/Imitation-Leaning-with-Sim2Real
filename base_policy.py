import numpy as np


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

        self.curr_left_waypoint = None
        self.curr_right_waypoint = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        def get_actions(trajectory, curr_waypoint):
            """
            Generate the actions from the trajectory
            """

            # Obtain the waypoints
            if trajectory[0]['t'] == self.step_count:
                curr_waypoint = trajectory.pop(0)
            next_waypoint = trajectory[0]

            # Interpolate between waypoints to obtain current pose and gripper command
            xyz, quat, gripper = self.interpolate(curr_waypoint, next_waypoint, self.step_count)

            # Inject noise
            if self.inject_noise:
                scale = 0.01
                xyz = xyz + np.random.uniform(-scale, scale, xyz.shape)

            action = np.concatenate([xyz, quat, [gripper]])

            return action, curr_waypoint

        action_left, self.curr_left_waypoint = get_actions(self.left_trajectory, curr_waypoint=self.curr_left_waypoint)

        action_right = None
        if self.right_trajectory is not None:
            action_right, self.curr_right_waypoint = get_actions(self.right_trajectory,
                                                                 curr_waypoint=self.curr_right_waypoint)

        self.step_count += 1

        if action_right is None:
            return action_left

        return np.concatenate([action_left, action_right])
