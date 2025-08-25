import time
import os

import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py
import yaml

from ee_sim_env import make_ee_sim_env
from gripper_utils import gripper_normalize
from scripted_yaml_policy import YAMLPolicy
from sim_env import make_sim_env, BOX_POSE
from scripted_policy import PickAndTransferPolicy, InsertionPolicy

import IPython

from utils import depth_to_8bit

e = IPython.embed


def main(args):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """

    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    gzip = args['gzip']
    sim_script = args['sim_script']
    inject_noise = False
    render_cam_name = 'top'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    class YAMLPolicyMeta(type):
        """
        Just to get around the fact that Policy is forced to be a single arg constructor class
        """

        def __new__(cls, name, bases, attrs):
            ParentClass = bases[0]

            def __init__(self, inject_noise_arg):
                ParentClass.__init__(self, sim_script, inject_noise_arg)

            attrs['__init__'] = __init__

            return super().__new__(cls, name, bases, attrs)

    class YAMLPolicySpecialize(YAMLPolicy, metaclass=YAMLPolicyMeta):
        pass

    episode_len = args['episode_len']
    camera_names = args['camera_names']
    if task_name == 'sim_transfer_cube_scripted':
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_insertion_scripted':
        policy_cls = InsertionPolicy
    elif task_name == 'yaml_policy':
        policy_cls = YAMLPolicySpecialize
    else:
        raise NotImplementedError

    success = []
    for episode_idx in range(num_episodes):
        print(f'{episode_idx=}')
        print('Rollout out EE space scripted policy')

        # Create the policy that we have selected up above
        policy = policy_cls(inject_noise)

        # setup the environment
        env = make_ee_sim_env(task_name, policy)
        ts = env.reset()
        episode = [ts]

        # setup plotting
        if onscreen_render:
            # Show the matplot lib window
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()

        gripper_center_body_data = []
        gripper_center_body_id = env.physics.model.body('gripper_center_body').id

        for step in range(episode_len):
            action = policy(ts)
            if action is None:
                break
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.002)

            # Position of the body gripper_center_body
            gripper_center_body_pos = env.physics.data.xpos[gripper_center_body_id]
            gripper_center_body_data.append(gripper_center_body_pos.copy())

        # Write the data to file as csv file
        gripper_center_body_data = np.array(gripper_center_body_data)
        np.savetxt(os.path.join(dataset_dir, f'episode_{episode_idx}_gripper_center_body.csv'),
                   gripper_center_body_data, delimiter=',')

        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        if episode_max_reward == env.task.max_reward:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

        joint_traj = [ts.observation['qpos'] for ts in episode]
        # replace gripper pose with gripper control
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            left_ctrl = gripper_normalize(ctrl[0], policy.gripper_close[0], env.task.gripper_open_value[0])
            # left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            joint[6] = left_ctrl

            if policy.n_robots > 1:
                right_ctrl = gripper_normalize(ctrl[2], policy.gripper_close[1], env.task.gripper_open_value[1])
                # right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
                joint[6 + 7] = right_ctrl

        subtask_info = episode[0].observation['env_state'].copy()  # box pose at step 0

        # clear unused variables
        del env
        del episode

        # Save the joint trajectory to a file in the dataset directory
        np.save(os.path.join(dataset_dir, f'episode_jt_{episode_idx}.npy'), joint_traj)

        # setup the environment
        print('Replaying joint commands')
        env = make_sim_env(task_name, policy)
        BOX_POSE[0] = subtask_info  # make sure the sim_env has the same object configurations as ee_sim_env
        ts = env.reset()

        episode_replay = [ts]
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for t in range(len(joint_traj)):  # note: this will increase episode length by 1
            action = joint_traj[t]
            ts = env.step(action)
            episode_replay.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.002)

        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        if episode_max_reward == env.task.max_reward:
            success.append(1)
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{episode_idx=} Failed")

        plt.close()

        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (14,)         'float64'
        """

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
        # truncate here to be consistent
        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                # If cam_name ends in RGBD, then combine the RGB and Depth images into a single array
                if cam_name.endswith('_RGBD'):
                    # Camera name is cam_name without the _RGBD
                    cam_name_internal = cam_name[:-5]
                    # Get the RGB and Depth images
                    rgb_image = ts.observation['images'][cam_name_internal]
                    depth_image = depth_to_8bit(ts.observation['depths'][cam_name_internal])
                    # Combine the RGB and Depth images into a single array
                    image = np.concatenate([rgb_image, depth_image[..., None]], axis=-1)
                    # Add it to the data_dict
                    data_dict[f'/observations/images/{cam_name}'].append(image)
                else:
                    data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')

            # Default is no compression. Otherwise, use gzip with the given compression level
            compression_opts = {}
            if gzip is not None:
                compression_opts = {'compression': 'gzip', 'compression_opts': gzip}

            for cam_name in camera_names:
                # If the cam_name ends in RGBD, then combine the RGB and Depth images into a single array
                if cam_name.endswith('RGBD'):
                    image.create_dataset(cam_name, (max_timesteps, 480, 640, 4), dtype='uint8',
                                         chunks=(1, 480, 640, 4), **compression_opts)
                else:
                    image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), **compression_opts)

            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, policy.n_robots * 7))
            qvel = obs.create_dataset('qvel', (max_timesteps, policy.n_robots * 7))
            action = root.create_dataset('action', (max_timesteps, policy.n_robots * 7))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action='store', type=str, help='config file', required=True)
    cl_args = parser.parse_args()

    # Get the config file from the command line, read the yaml contents as a dictionary
    config_file = cl_args.config
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Start creating the dataset
    main(config)
