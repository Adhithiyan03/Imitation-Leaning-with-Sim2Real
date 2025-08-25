import os
from time import sleep

import dm_env
import mujoco
import mujoco.viewer
import numpy as np
import yaml
import xml.etree.ElementTree as ET

from gripper_utils import gripper_un_normalize
from scripted_policy import BasePolicy

from quat_math import euler2quat, quaternion_multiply, mul_pose


def get_yaml(yaml_file: str):
    """
    Get the contents of the yaml file as a dictionary
    """
    with open(yaml_file, 'r') as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_data


def get_temp_xml_filename(xml_file, yaml_data, tag='temp_ee_xml', default_temp_xml='__temp__.xml'):
    """
    Get the temp xml file name from the yaml data
    """
    # Write the file to the same directory as xml file
    xml_dir = os.path.dirname(xml_file)

    # Check if yaml_data has temp_xml field for the temp xml file name else use __temp__.xml
    if tag in yaml_data:
        temp_xml_file = yaml_data[tag]
    else:
        temp_xml_file = os.path.join(xml_dir, default_temp_xml)

    return temp_xml_file


def mujoco_step(model, data, threshold=1e-6, max_iters=250):
    for n in range(max_iters):
        mujoco.mj_step(model, data)


def mujoco_step3(model, data, threshold=1e-6, max_iters=250):
    mocap_ids = [model.body('mocap_left').id, model.body('mocap_right').id]
    center_bodies = [model.body('left_gripper_center_body').id, model.body('right_gripper_center_body').id]

    for n in range(max_iters):
        mujoco.mj_step(model, data)

        # Get the mocap positions
        mocap_pos = [data.xpos[mocap_id] for mocap_id in mocap_ids]

        # Get the center body positions
        center_body_pos = [data.xpos[center_body] for center_body in center_bodies]

        # Check if the mocap positions are close to the center body positions
        mocap_center_diff = [np.linalg.norm(mocap_pos[i] - center_body_pos[i]) for i in range(len(mocap_pos))]
        if all([abs(diff) < threshold for diff in mocap_center_diff]):
            print(n, mocap_center_diff)
            break


def mujoco_step2(model, data, threshold=1e-6, max_iters=250):
    """
    Step the mujoco simulation until the threshold is reached or max_iters is reached
    """

    gripper_body_ids = [model.body('vx300s_left/gripper_link').id, model.body('vx300s_right/gripper_link').id]

    last_max_norm = 1e10
    for n in range(max_iters):
        max_norm = 0
        for i in range(len(gripper_body_ids)):
            mocap_pos = data.mocap_pos[i]
            gripper_site_pos = data.xpos[gripper_body_ids[i]]
            diff_norm = np.linalg.norm(mocap_pos - gripper_site_pos)
            max_norm += max(max_norm, diff_norm)

        norm_diff = abs(max_norm - last_max_norm)
        if norm_diff < threshold:
            print(n, norm_diff)
            break
        else:
            last_max_norm = max_norm

        mujoco.mj_step(model, data)
        if i == max_iters - 1:
            print('Max iters reached')


class YAMLPolicy(BasePolicy):
    # Constructor that takes a yaml file
    def __init__(self, yaml_file, inject_noise=False):
        super().__init__(inject_noise=inject_noise)
        self.trajectories = None
        self.temp_ee_xml = None
        self.temp_xml = None
        self.yaml_data = None
        self.points = None
        self.yaml_file = yaml_file

        # Local simulation parameters
        self.trajectory_names = None
        self.n_robots = None
        self.ctrl = None
        self.gripper_open = None
        self.gripper_close = None
        self.ctrl_offset = None

        self.setup_simulation()

    def setup_simulation(self):
        # Use the yaml file to build the xml file
        self.yaml_data = get_yaml(self.yaml_file)

        # Make the points from the yaml data
        self.points = self.make_points(self.yaml_data)

        # Set up the environment and xml file.
        self.temp_ee_xml, self.temp_xml = self.make_temp_sim_file(self.yaml_data)

    def add_data_to_xml(self, yaml_data, root):
        # Add the includes to the xml file after all the other includes if we have an include statement
        if 'includes' in yaml_data:
            for index, child in enumerate(root):
                if child.tag != 'include':
                    for include in yaml_data['includes']:
                        root.insert(index, ET.fromstring(f'<include file="{include}"/>'))
                    break

        # Update the settings in the xml file
        self.update_settings(root, yaml_data)

        # Add the parts to the xml file
        self.add_parts(root, yaml_data)

    def make_xml(self, yaml_data, xml_tag):
        """
        Create the XML data structure from the YAML data
        """
        # This is the main xml file
        xml_file = yaml_data[xml_tag]
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Add the includes to the xml file after all the other includes
        self.add_data_to_xml(yaml_data, root)

        return tree, xml_file

    def make_temp_sim_file(self, yaml_data):
        """
            Load the XML and add all the additional data to it to run mujoco
        """

        # These are the tags for the xml files (ee and non-ee)
        env_tags = ['env_ee_xml', 'env_xml']
        temp_tags = ['temp_ee_xml', 'temp_xml']
        defaults = ['__temp_ee__.xml', '__temp__.xml']
        temp_xml_files = []

        for env_tag, temp_tag, default in zip(env_tags, temp_tags, defaults):
            # Make the xml file with ee
            tree, xml_file = self.make_xml(yaml_data, xml_tag=env_tag)

            # Write the file to the same directory as xml file but called __temp__.xml or whatever is specified in yaml
            temp_xml_file = get_temp_xml_filename(xml_file, yaml_data, tag=temp_tag, default_temp_xml=default)
            temp_xml_files.append(temp_xml_file)

            # XML string
            xml_str = ET.tostring(tree.getroot(), encoding='unicode', method='xml')

            with open(temp_xml_file, 'w') as f:
                # Convert the file to a string
                f.write(xml_str)

        return temp_xml_files

    def make_points(self, yaml_data):
        """
        Make the points from the yaml data
        :param yaml_data: YAML data to make the points from
        :return: Dictionary of points
        """
        points = {}
        for point in yaml_data['points']:
            pt = point['pos']

            # Check if point has rand and if so generate it from the random interval with uniform distribution
            if 'rand' in point:
                rand = point['rand']
                intervals = [[mid - delta / 2, mid + delta / 2] for mid, delta in zip(pt, rand)]
                pt = [np.random.uniform(interval[0], interval[1]) for interval in intervals]

            # Get the euler angles or quaternions
            if 'euler' in point:
                euler = point['euler']
                quat = euler2quat(euler)
            elif 'quat' in point:
                quat = point['quat']
            else:
                # Use default value of [1, 0, 0, 0]
                quat = [1, 0, 0, 0]

            points[point['name']] = {'pos': pt, 'quat': quat}

        return points

    def update_settings(self, root, yaml_data):
        """
        Given the root node of the xml tree, update the settings
        """

        # Setting for the XML file

        # Update the imp-ratio settings. If it is in the yaml file, then use that value. Otherwise, ignore it.
        # If it is also in the root, update the value. If not, add it to the xml tree
        if 'impratio' in yaml_data:
            impratio = yaml_data['impratio']

            # The root is the mujoco xml file. Check all the options in the root
            fount_impratio = False
            for index, child in enumerate(root):
                if child.tag == 'option':
                    if 'impratio' in child.attrib:
                        # Update the value if it is already there
                        child.attrib['impratio'] = str(impratio)
                        fount_impratio = True
                        break

            # If it is not found, add it to the xml tree
            if not fount_impratio:
                # Find the last option, include and add it after that
                index = 0
                for index, child in enumerate(root):
                    if child.tag == 'option' or child.tag == 'include':
                        continue
                    else:
                        break
                root.insert(index, ET.fromstring(f'<option impratio="{str(impratio)}"/>'))

        # Setting for the mujoco simulation

        # Look for the ctrl field in the yaml file. If it is there, then use that value. Otherwise, use the default
        # value of 6 and 7
        if 'ctrl' in yaml_data:
            ctrl = yaml_data['ctrl']
        else:
            ctrl = [6, 7]
        self.ctrl = ctrl

        # Get the value for ctrl_offset if it is in the yaml file. Otherwise, use the default value of 0.0
        if 'ctrl_offset' in yaml_data:
            self.ctrl_offset = yaml_data['ctrl_offset']
        else:
            self.ctrl_offset = 0

        # Look for the gripper_open field in the yaml file. If it is there, then use that value. Otherwise, use the
        # default value of 0.0 for closed and 1.0 for open
        if 'gripper_open' in yaml_data:
            gripper_open = yaml_data['gripper_open']
            gripper_close = yaml_data['gripper_close']
        else:
            gripper_open = 1.0
            gripper_close = 0.0
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close

        # Fill in trajectory names
        if 'trajectories' in yaml_data:
            self.trajectory_names = yaml_data['trajectories']
            self.n_robots = len(self.trajectory_names)

    def add_parts(self, root_node, yaml_data):
        """
        Add the parts to the xml tree
        :param xml_tree: XML tree to add the parts to
        :param yaml_data: YAML data to add the parts from
        :return: XML tree with the parts added
        """

        # First check if there are even any parts to add
        if 'parts' not in yaml_data:
            return

        # Get the body node
        world_body_node = root_node.find('worldbody')

        for part in yaml_data['parts']:
            part_name = part['name']
            part_pos = self.get_pos(part['pos'])

            # Check if the part has euler angles or quaternions
            if 'euler' in part:
                part_euler = part['euler']
                part_quat = euler2quat(part_euler)
            elif 'quat' in part:
                part_quat = part['quat']
            else:
                raise ValueError('Part must have either euler or quat')

            # Check if part has color
            if 'color' in part:
                # Convert string of array of float into a space seperated string
                part_color = part['color']
            else:
                part_color = '0.5 0.5 0.5 1'

            # Check for friction
            if 'friction' in part:
                part_friction = part['friction']
            else:
                part_friction = '4 0.1 0.01'

            part_pos_str = ' '.join([str(x) for x in part_pos])
            part_quat_str = ' '.join([str(x) for x in part_quat])

            body_xml = f'<body name="{part_name}" pos="{part_pos_str}" quat="{part_quat_str}" >' \
                       f'<joint name="{part_name}_joint" type="free" frictionloss="0.01"/>' \
                       f'<inertial pos="0 0 0" mass="0.05" diaginertia="0.002 0.002 0.002"/> ' \
                       f'<geom condim="4" solimp="2 1 0.01" solref="0.01 1" friction="{part_friction}" ' \
                       f'type="mesh" mesh="{part_name}" name="{part_name}" rgba="{part_color}"/>' \
                       f'</body>'
            world_body_node.append(ET.fromstring(body_xml))

            keyframe_node = root_node.find('keyframe')
            key_node = keyframe_node.find('key')
            key_node.attrib['qpos'] += ' ' + part_pos_str + ' ' + part_quat_str

        # Weld the requested parts together
        # Find the node equality
        equality_node = root_node.find('equality')
        if 'welds' in yaml_data:
            for weld in yaml_data['welds']:
                part1, part2 = weld
                equality_xml = f'<weld body1="{part1}" body2="{part2}"/>'
                equality_node.append(ET.fromstring(equality_xml))

    def get_pos(self, pos_data):
        """
        Given the data in `pos`, return the numpy array that is relevant to it
        :param pos_data:
        :return:
        """

        # If pos_data is a list, then return it as a numpy array
        if isinstance(pos_data, list):
            return np.array(pos_data)
        # If it is a string, it is a reference to a position
        elif isinstance(pos_data, str):
            return self.points[pos_data]['pos']

    def get_pose_global(self, point_data):
        """
        Get the pose of the given point in the global frame
        :return: pos and quat in global frame
        """
        pos_type = point_data['pos_type']
        if pos_type == 'global':
            xyz = self.get_pos(point_data['pos'])
        elif pos_type == 'relative':
            # What is it relative to?
            pose_relative = point_data['pos_relative']
            xyz = self.points[pose_relative]['pos'] + self.get_pos(point_data['pos'])
        else:
            raise ValueError(f'Pose type {pos_type} not supported')

        orientation_type = point_data['orientation_type']
        if orientation_type == 'global':
            # Check if we have euler or quat
            if 'euler' in point_data:
                euler = point_data['euler']
                quat_final = euler2quat(euler)
            elif 'quat' in point_data:
                quat_final = point_data['quat']
            else:
                raise ValueError('Point must have either euler or quat')

        elif orientation_type == 'relative':
            # What is it relative to?
            orientation_relative = point_data['orientation_relative']

            # Get the pose of the relative point
            body_quat = self.points[orientation_relative]['quat']

            # Check if we have euler or quat
            if 'euler' in point_data:
                euler = point_data['euler']
                quat = euler2quat(euler)
            elif 'quat' in point_data:
                quat = point_data['quat']
            else:
                raise ValueError('Point must have either euler or quat')

            quat_final = quaternion_multiply(quat, body_quat)

        return xyz, quat_final

    def generate_trajectory(self, ts_first):
        # Get the initial mocap_left and mocap_right poses from ts_first (pose includes 3 xyz and 4 quaternion)
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        # Add these to points
        self.points['mocap_left'] = {'pos': init_mocap_pose_left[:3], 'quat': init_mocap_pose_left[3:3 + 4]}

        # Do the same for the right
        if self.n_robots > 1:
            init_mocap_pose_right = ts_first.observation['mocap_pose_right']
            self.points['mocap_right'] = {'pos': init_mocap_pose_right[:3], 'quat': init_mocap_pose_right[3:3 + 4]}

        # Get the names of the trajectories from the yaml file
        trajectory_names = self.yaml_data['trajectories']
        mocap_names = self.yaml_data['mocap']

        # Make the trajectories with mocap_left and mocap_right as the start
        trajectories = [[{'t': 0, "xyz": self.points[mocap_names[i]]['pos'].copy(),
                          "quat": self.points[mocap_names[i]]['quat'].copy(), "gripper": 1.0}] for i in
                        range(len(trajectory_names))]

        for point_i, point_name in enumerate(trajectory_names):
            if point_name not in self.yaml_data:
                continue

            for point in self.yaml_data[point_name]:
                if 't' in point:
                    t = point['t']
                elif 'dt' in point:
                    dt = point['dt']
                    t = trajectories[point_i][-1]['t'] + dt
                xyz, quat = self.get_pose_global(point)
                gripper = point['gripper']
                trajectories[point_i].append({'t': t, "xyz": xyz, "quat": quat, "gripper": gripper})

        # Make sure that the trajectories end at the same time. If not, pad the shorter one with the last point
        # of the largest one
        max_t = max([traj[-1]['t'] for traj in trajectories])
        for traj in trajectories:
            if traj[-1]['t'] < max_t:
                traj.append(traj[-1].copy())
                traj[-1]['t'] = max_t

        self.trajectories = trajectories
        self.left_trajectory = trajectories[0]
        if len(trajectories) > 1:
            self.right_trajectory = trajectories[1]

        return trajectories


"""
Test functions
"""


def basic_trajectory_generate_test():
    """
    # Generate a trajectory from a test yaml file. Simple smoke test.
    """
    test_yaml_file = 'sim_scripts/test.yaml'
    yaml_policy = YAMLPolicy(test_yaml_file)
    yaml_policy.setup_simulation()
    ts_first_dm_env = dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=None,
        discount=None,
        observation={
            'mocap_pose_left': np.array([-0.31718881, 0.5, 0.29525084, 1.0, 0.0, 0.0, 0.0]),
            'mocap_pose_right': np.array([0.31718881, 0.49999888, 0.29525084, 1.0, 0.0, 0.0, 0.0]),
        }
    )
    trajectory = yaml_policy.generate_trajectory(ts_first=ts_first_dm_env)
    for traj in trajectory:
        for point in traj:
            print(point)
        print()


if __name__ == '__main__':
    basic_trajectory_generate_test()


def set_key_data(model, data, n_trajectories):
    # Set the qpos from the key
    data.qpos = model.key_qpos[0]
    data.ctrl = model.key_ctrl[0]
    for i in range(n_trajectories):
        data.mocap_pos[i] = model.key_mpos[0][i * 3:i * 3 + 3]
        data.mocap_quat[i] = model.key_mquat[0][i * 4:i * 4 + 4]


def mujoco_viewer_for_trajectory():
    """
    Run a trajectory in mujoco viewer using the end effector positions
    """
    # test_yaml_file = 'sim_scripts/viperx_med_bumper.yaml'
    test_yaml_file = 'sim_scripts/ur10e_med_bumper_move.yaml'
    # test_yaml_file = 'sim_scripts/ur10e_2f85.yaml'
    yaml_policy = YAMLPolicy(test_yaml_file)

    model = mujoco.MjModel.from_xml_path(yaml_policy.temp_ee_xml)
    data = mujoco.MjData(model)

    n_trajectories = len(yaml_policy.trajectory_names)
    set_key_data(model, data, n_trajectories)

    ts_first_dm_env = dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=None,
        discount=None,
        observation={
            'mocap_pose_left': np.append(model.key_mpos[0][:3].copy(), model.key_mquat[0][:4].copy()),
            'mocap_pose_right': np.append(model.key_mpos[0][3:].copy(), model.key_mquat[0][4:].copy()),
        }
    )

    for i, name in enumerate(['mocap_pose_left']):
        mocap_pos = ts_first_dm_env.observation[name][:3]
        mocap_quat = ts_first_dm_env.observation[name][3:3 + 4]

        data.mocap_pos[i] = mocap_pos
        data.mocap_quat[i] = mocap_quat

    for i in range(10000):
        mujoco.mj_step(model, data)

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        while viewer.is_running():
            # Iterate over time
            pt = yaml_policy(ts_first_dm_env)

            # Set up the mocap
            for i in range(n_trajectories):
                data.mocap_pos[i] = pt[i * 8:i * 8 + 3]
                data.mocap_quat[i] = pt[i * 8 + 3:i * 8 + 7]

            # Set the gripper using ctrl
            ctrl_step = yaml_policy.ctrl_offset + len(yaml_policy.ctrl)
            for i in range(n_trajectories):
                # Iterate over the number of parameters for the grippers and end effectors
                for j, ctrl_j in enumerate(yaml_policy.ctrl):
                    # Numbers 8 and 7 are hard-wired here because we have 8 parameters for the end effectors
                    # (3 pos + 4 quat + 1 gripper) and the 7th value is always the gripper value
                    data.ctrl[ctrl_step * i + ctrl_j] = gripper_un_normalize(pt[i * 8 + 7],
                                                                             yaml_policy.gripper_close[j],
                                                                             yaml_policy.gripper_open[j])

            mujoco_step(model, data, threshold=1e-3, max_iters=100)
            viewer.sync()

            # Check if the last time in the left trajectory is reached
            if yaml_policy.step_count >= yaml_policy.left_trajectory[-1]['t']:
                mujoco.mj_resetData(model, data)

                data.qpos = model.key_qpos[0]
                data.ctrl = model.key_ctrl[0]
                for i in range(1):
                    data.mocap_pos[i] = model.key_mpos[0][i * 3:i * 3 + 3]
                    data.mocap_quat[i] = model.key_mquat[0][i * 4:i * 4 + 4]

                yaml_policy.step_count = 0


if __name__ == '__main__':
    mujoco_viewer_for_trajectory()
