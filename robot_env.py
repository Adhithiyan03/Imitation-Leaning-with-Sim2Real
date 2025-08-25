import numpy as np
import collections
import os

from constants import DT, XML_DIR, START_ARM_POSE, PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN
from gripper_utils import gripper_normalize, gripper_un_normalize

from pointclouds.points_clouds import generate_point_cloud
from utils import sample_box_pose, sample_insertion_pose
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base


class RobotBase():
    n_robots = 1
    ctrl_offset = 0
    ctrl_idx = None
    gripper_close_value = 0.0
    gripper_open_value = 1.0


class BimanualViperXBase(RobotBase):
    def __init__(self):
        self.n_robots = 2
        self.ctrl_offset = 0
        self.gripper_close_value = [
            PUPPET_GRIPPER_POSITION_CLOSE, PUPPET_GRIPPER_POSITION_CLOSE]
        self.gripper_open_value = [
            PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_OPEN]
        self.ctrl_idx = None

    @staticmethod
    def get_qpos(physics, n_robots, gripper_close_value, gripper_open_value):
        qpos_raw = physics.data.qpos.copy()

        left_qpos_raw = qpos_raw[:8]
        left_arm_qpos = left_qpos_raw[:6]
        left_gripper_qpos = [gripper_normalize(
            left_qpos_raw[6], gripper_close_value[0], gripper_open_value[0])]
        # left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]

        if n_robots > 1:
            right_qpos_raw = qpos_raw[8:16]
            right_arm_qpos = right_qpos_raw[:6]
            right_gripper_qpos = [gripper_normalize(
                right_qpos_raw[6], gripper_close_value[1], gripper_open_value[1])]
            # right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        else:
            right_arm_qpos = []
            right_gripper_qpos = []

        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics, n_robots):
        qvel_raw = physics.data.qvel.copy()

        left_qvel_raw = qvel_raw[:8]
        left_arm_qvel = left_qvel_raw[:6]
        # left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        left_gripper_qvel = [left_qvel_raw[6]]

        if n_robots > 1:
            right_qvel_raw = qvel_raw[8:16]
            right_arm_qvel = right_qvel_raw[:6]
            # right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
            right_gripper_qvel = [right_qvel_raw[6]]
        else:
            right_arm_qvel = []
            right_gripper_qvel = []

        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_pos(physics, n_robots, gripper_close_value, gripper_open_value, ee_body_name):
        # Get the id of ee_body_name
        ee_body_id = physics.model.body(ee_body_name).id

        # Get the position of the end effector
        left_ee_pos = physics.data.xpos[ee_body_id].copy()
        left_ee_quat = physics.data.xquat[ee_body_id].copy()

        # Get the position of the gripper
        left_gripper_pos = physics.data.qpos[6].copy()
        left_gripper_pos = gripper_normalize(
            left_gripper_pos, gripper_close_value[0], gripper_open_value[0])

        if n_robots > 1:
            right_ee_pos = physics.data.xpos[ee_body_id + 1].copy()
            right_ee_quat = physics.data.xquat[ee_body_id + 1].copy()

            right_gripper_pos = physics.data.qpos[13].copy()
            right_gripper_pos = gripper_normalize(
                right_gripper_pos, gripper_close_value[1], gripper_open_value[1])

        # Return the combined position, quaternion, and gripper position
        if n_robots > 1:
            return np.concatenate([left_ee_pos, left_ee_quat, left_gripper_pos, right_ee_pos, right_ee_quat, right_gripper_pos])
        else:
            return np.concatenate([left_ee_pos, left_ee_quat, [left_gripper_pos]])

    @ staticmethod
    def get_env_state(physics):
        raise NotImplementedError


class BimanualViperXEETask(base.Task, BimanualViperXBase):
    def __init__(self, random=None, ee_body_name=None):
        base.Task.__init__(self, random=random)
        BimanualViperXBase.__init__(self)
        self.ee_body_name = ee_body_name
        self.use_point_clouds = True  # TODO: Pass this from the main function yaml parameters
        self.pc_min_bound = [-3, -3, -3]  # TODO: Get this from the yaml file
        self.pc_max_bound = [3, 3, 3]  # TODO: Get this from the yaml file

    def before_step(self, action, physics):
        a_len = len(action) // self.n_robots
        action_left = action[:a_len]

        # set mocap position and quat
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])

        # set gripper
        # g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        g_left_ctrl = gripper_un_normalize(
            action_left[7], self.gripper_close_value[0], self.gripper_open_value[0])

        # right
        if self.n_robots == 1:
            np.copyto(physics.data.ctrl[self.ctrl_offset:], np.array(
                [g_left_ctrl, -g_left_ctrl]))
        else:
            action_right = action[a_len:]
            np.copyto(physics.data.mocap_pos[1], action_right[:3])
            np.copyto(physics.data.mocap_quat[1], action_right[3:7])
            # g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
            g_right_ctrl = gripper_un_normalize(action_right[7], self.gripper_close_value[1],
                                                self.gripper_open_value[1])
            np.copyto(physics.data.ctrl, np.array(
                [g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]))

    def initialize_robots(self, physics):
        # reset joint position
        # physics.named.data.qpos[:16] = self.start_pose
        np.copyto(physics.data.qpos, physics.model.key_qpos[0])

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to start pose from the xml file
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side
        np.copyto(physics.data.mocap_pos[0], physics.model.key_mpos[0][:3])
        np.copyto(physics.data.mocap_quat[0], physics.model.key_mquat[0][:4])
        # right
        if physics.data.mocap_pos.shape[0] > 1:
            np.copyto(physics.data.mocap_pos[1],
                      physics.model.key_mpos[0][3:6])
            np.copyto(physics.data.mocap_quat[1],
                      physics.model.key_mquat[0][4:8])

        # reset gripper control
        np.copyto(physics.data.ctrl, physics.model.key_ctrl[0])

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(
            physics, self.n_robots, self.gripper_close_value, self.gripper_open_value)
        obs['qvel'] = self.get_qvel(physics, self.n_robots)
        if self.ee_body_name is not None:
            obs['pos'] = self.get_pos(
                physics, self.n_robots, self.gripper_close_value, self.gripper_open_value, self.ee_body_name)
        obs['env_state'] = self.get_env_state(physics)

        # Use the cameras to get the images for training
        obs['images'] = dict()
        obs['images']['top'] = physics.render(
            height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(
            height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(
            height=480, width=640, camera_id='front_close')

        # Depth images
        obs['depths'] = dict()
        obs['depths']['top'] = physics.render(
            height=480, width=640, camera_id='top', depth=True)
        obs['depths']['angle'] = physics.render(
            height=480, width=640, camera_id='angle', depth=True)
        obs['depths']['vis'] = physics.render(
            height=480, width=640, camera_id='front_close', depth=True)

        # Convert the depth images to point clouds
        if self.use_point_clouds:
            obs['point_cloud'] = dict()
            for camera, dcamera in [('top', 'top'), ('angle', 'angle'), ('front_close', 'vis')]:
                # Get the depth image and dimensions
                depth_image = obs['depths'][dcamera]
                depth_image = np.ascontiguousarray(
                    depth_image).astype(np.float32)
                height, width = depth_image.shape

                # Get the camera parameters
                camera_id = physics.model.camera(camera).id
                fovy = physics.model.cam_fovy[camera_id]
                cam_pos = physics.data.cam_xpos[camera_id]
                cam_xmat = physics.data.cam_xmat[camera_id]

                point_cloud = generate_point_cloud(depth_image, width, height, fovy, cam_pos, cam_xmat, self.pc_min_bound,
                                                   self.pc_max_bound)
                points = np.asarray(point_cloud.points)
                obs['point_cloud'][dcamera] = points

        # used in scripted policy to obtain starting pose
        obs['mocap_pose_left'] = np.concatenate(
            [physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        if self.n_robots > 1:
            obs['mocap_pose_right'] = np.concatenate(
                [physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()

        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl[self.ctrl_idx].copy()

        return obs

    def get_reward(self, physics):
        raise NotImplementedError


class BimanualViperXTask(base.Task, BimanualViperXBase):
    def __init__(self, random=None):
        base.Task.__init__(self, random=random)
        BimanualViperXBase.__init__(self)

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        normalized_left_gripper_action = action[6]
        # left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)
        left_gripper_action = gripper_un_normalize(normalized_left_gripper_action, self.gripper_close_value[0],
                                                   self.gripper_open_value[0])
        # print(left_gripper_action)
        full_left_gripper_action = [left_gripper_action, -left_gripper_action]

        if self.n_robots > 1:
            right_arm_action = action[7:7 + 6]
            normalized_right_gripper_action = action[7 + 6]
            # right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)
            right_gripper_action = gripper_un_normalize(normalized_right_gripper_action, self.gripper_close_value[1],
                                                        self.gripper_open_value[1])
            full_right_gripper_action = [
                right_gripper_action, -right_gripper_action]

            env_action = np.concatenate(
                [left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])
        else:
            env_action = np.concatenate(
                [left_arm_action, full_left_gripper_action])

        super().before_step(env_action, physics)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(
            physics, self.n_robots, self.gripper_close_value, self.gripper_open_value)
        obs['qvel'] = self.get_qvel(physics, self.n_robots)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(
            height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(
            height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(
            height=480, width=640, camera_id='front_close')
        obs['depths'] = dict()
        obs['depths']['top'] = physics.render(
            height=480, width=640, camera_id='top', depth=True)
        obs['depths']['angle'] = physics.render(
            height=480, width=640, camera_id='angle', depth=True)
        obs['depths']['vis'] = physics.render(
            height=480, width=640, camera_id='front_close', depth=True)

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError
