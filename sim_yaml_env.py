import numpy as np

from constants import START_ARM_POSE
from mujoco_utils import get_all_contact_pairs
from rewards import Rewards
from robot_env import BimanualViperXEETask, BimanualViperXTask

"""
Contains both the YAML Sim Task and YAML Sim EE Task
"""


class YAMLSimTaskBase():
    def __init__(self, args=None):
        self.current_state = 0
        self.args = args
        self.rewards = Rewards(args['reward_data'])
        self.max_reward = self.rewards.max_reward

    def update_reward(self, physics):
        # Get all the contact pairs
        all_contact_pairs = get_all_contact_pairs(physics)
        # Update the reward
        self.rewards.update_reward_state(all_contact_pairs)

    @staticmethod
    def get_env_state(physics):
        """
        Return the qpos
        Todo: Seems to always be the same, need to check
        """
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        self.update_reward(physics)
        return self.rewards.reward


class YAMLSimTask(YAMLSimTaskBase, BimanualViperXTask):
    def __init__(self, args, random=None):
        BimanualViperXTask.__init__(self, random=random)
        YAMLSimTaskBase.__init__(self, args=args)

    def initialize_episode(self, physics):
        """
        Set the qpos and ctrl to the keypoint when initializing the episode
        """
        with physics.reset_context():
            np.copyto(
                physics.named.data.qpos[:physics.model.nq], physics.named.model.key_qpos[0])
            np.copyto(physics.data.ctrl, physics.model.key_ctrl[0])
            return super().initialize_episode(physics)


class YAMLSimEETask(YAMLSimTaskBase, BimanualViperXEETask):
    def __init__(self, args, random=None):
        BimanualViperXEETask.__init__(
            self, random=random, ee_body_name=args['ee_body_name'])
        YAMLSimTaskBase.__init__(self, args=args)

    def initialize_episode(self, physics):
        self.initialize_robots(physics)
        return super().initialize_episode(physics)
