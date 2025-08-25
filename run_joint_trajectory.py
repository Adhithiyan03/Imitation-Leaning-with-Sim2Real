import os
import mujoco
import mujoco.viewer
import argparse
import numpy as np

from gripper_utils import gripper_un_normalize

"""
Utility to run the joint trajectory from the dataset. Currently only works for the UR10e XML since the body 
`gripper_center_body` is hardcoded.
TODO: Make this a more general utility for replaying joint trajectories
"""


def get_joint_traj(episode_idx, dataset_dir):
    joint_traj = np.load(os.path.join(dataset_dir, f'episode_jt_{episode_idx}.npy'))
    return joint_traj


def run_joint_trajectory(episode_idx, dataset_dir, xml_file, body_name):
    # Load the pickle from the dataset directory of the joint trajectory
    joint_traj = get_joint_traj(episode_idx, dataset_dir)

    def get_traj_point(t):
        """
        Get the trajectory point. The gripper value is repeated twice since we have to put two points for grippers and
        has to be normalized
        """
        traj_point = joint_traj[t].copy()
        gripper_value = traj_point[-1]
        traj_point[-1] = gripper_un_normalize(gripper_value, 0.0, 0.036)
        traj_point = np.append(traj_point, gripper_un_normalize(gripper_value, 0.0, -0.036))
        return traj_point

    # Run the mujoco simulation
    model = mujoco.MjModel.from_xml_path(xml_file)
    data = mujoco.MjData(model)

    # Set the qpos from the keypoint
    data.qpos = model.key_qpos[0]
    data.ctrl = model.key_ctrl[0]

    # Store for the actual trajectory in the mujoco simulation
    actual_joint_traj = []

    gripper_center_body_data = []
    gripper_center_body_id = model.body(body_name).id

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for t in range(len(joint_traj)):
                # Get the trajectory point. Repeat the last point since we have to put two points for grippers
                traj_point = get_traj_point(t)
                # print(traj_point)

                data.ctrl[:] = traj_point

                # model.opt.viscosity = 10.0

                for i in range(1):
                    # mujoco.mj_step(model, data)
                    mujoco.mj_step(model, data)

                # Store the actual joint trajectory
                actual_joint_traj.append(data.qpos.copy())

                viewer.sync()

                # Position of the body gripper_center_body
                gripper_center_body_pos = data.xpos[gripper_center_body_id]
                gripper_center_body_data.append(gripper_center_body_pos.copy())

            # Write the data to file as csv file
            gripper_center_body_data = np.array(gripper_center_body_data)
            np.savetxt(os.path.join(dataset_dir, f'episode_{episode_idx}_{body_name}_replay.csv'),
                       gripper_center_body_data, delimiter=',')

            # Reset the data
            mujoco.mj_resetData(model, data)

            data.qpos = model.key_qpos[0]
            data.ctrl = model.key_ctrl[0]
            mujoco.mj_step(model, data)
            gripper_center_body_data = []


if __name__ == '__main__':
    # Obtain the episode number, dataset directory and xml file as the command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode_idx', type=int, default=0)
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--xml_file', type=str, default='assets/__temp__.xml')
    parser.add_argument('--body_name', type=str, default='gripper_center_body')
    args = parser.parse_args()

    run_joint_trajectory(args.episode_idx, args.dataset_dir, args.xml_file, args.body_name)
