import dm_env
import numpy as np
import yaml

from scripted_yaml_policy import YAMLPolicy, get_yaml, get_temp_xml_filename
import os


def prep_test():
    """
    Change the directory up over level from where the test is being run
    Todo: How to fix this for running outside of PyCharm?
    """

    # Get the current directory of the test
    test_dir = os.path.dirname(os.path.abspath(__file__))

    # Change the directory up over level from the test directory
    parent_dir = os.path.dirname(test_dir)
    os.chdir(parent_dir)


def test_scripted_yaml_policy_whole():
    """
    Run through the whole YAML policy to do the smoke test
    """
    # Change the directory up over level from where the test is being run since
    prep_test()

    test_scripts = ['sim_scripts/test.yaml', 'sim_scripts/ur10e_med_bumper_move.yaml',
                    'sim_scripts/ur10e_med_bumper_move.yaml', 'sim_scripts/viperx_med_bumper.yaml']
    for test_script in test_scripts:
        policy = YAMLPolicy(test_script)
        assert policy is not None


def test_xml_build():
    """
    Test that the XML object is built correctly
    """
    prep_test()

    test_script = 'sim_scripts/test.yaml'
    yaml_data = get_yaml(test_script)
    assert yaml_data['env_ee_xml'] == './assets/bimanual_viperx_ee_tape_free.xml'

    policy = YAMLPolicy(test_script)
    tree, xml_file = policy.make_xml(yaml_data, xml_tag='env_xml')

    # Get the name of the temp xml file
    temp_xml_file = get_temp_xml_filename(xml_file, yaml_data, tag='temp_xml')
    assert temp_xml_file == './assets/__temp__.xml'

    # Check the contents of the xml tree
    root = tree.getroot()
    assert root.tag == 'mujoco'

    # Check that the door latch dependencies are in the includes
    includes = root.findall('include')
    assert len(includes) == 3
    # Check that one of the includes is ./door_latch/door_latch_dependencies.xml
    assert any(include.attrib['file'] == './door_latch/door_latch_dependencies.xml' for include in includes)

    # Create the sim file and check that the XML file was created
    policy.make_temp_sim_file(yaml_data)
    assert os.path.exists(temp_xml_file)


def test_xml_parts():
    prep_test()

    test_script = 'sim_scripts/test.yaml'
    yaml_data = get_yaml(test_script)

    policy = YAMLPolicy(test_script)
    tree, xml_file = policy.make_xml(yaml_data, xml_tag='env_ee_xml')

    # Check that when making the xml file, the points were also created and we have the start_pos and end_pos
    assert policy.points is not None
    assert 'start_pos' in policy.points
    assert 'end_pos' in policy.points
    np.testing.assert_array_equal(policy.points['lift_pt']['pos'],[-0.05, 0.55, 0.1])
    np.testing.assert_array_equal(policy.points['start_pos']['quat'], [1, 0, 0, 0])
    np.testing.assert_array_equal(policy.points['end_pos']['quat'], [1, 0, 0, 0])

    # Check that the parts are added to the xml file and that the part door_latch_medium_bumper is added
    root = tree.getroot()
    world_body = root.find('worldbody')
    parts = world_body.findall('body')
    assert any(part.attrib['name'] == 'door_latch_medium_bumper' for part in parts)

    # Find the part named door_latch_medium_bumper and check its position
    door_latch_medium_bumper = filter(lambda part: part.attrib['name'] == 'door_latch_medium_bumper', parts)
    door_latch_medium_bumper = next(door_latch_medium_bumper)
    pos_data = [float(num) for num in door_latch_medium_bumper.attrib['pos'].split()]
    assert pos_data == policy.points['start_pos']['pos']


def test_trajectory_generation():
    prep_test()

    test_script = 'sim_scripts/test.yaml'
    yaml_data = get_yaml(test_script)

    policy = YAMLPolicy(test_script)
    policy.setup_simulation()

    ts_first_dm_env = dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=None,
        discount=None,
        observation={
            'mocap_pose_left': np.array([-0.31718881, 0.5, 0.29525084, 1.0, 0.0, 0.0, 0.0]),
            'mocap_pose_right': np.array([0.31718881, 0.49999888, 0.29525084, 1.0, 0.0, 0.0, 0.0]),
        }
    )
    trajectories = policy.generate_trajectory(ts_first=ts_first_dm_env)

    assert 'mocap_left' in policy.points
    np.testing.assert_array_equal(policy.points['mocap_left']['pos'], [-0.31718881, 0.5, 0.29525084])
    np.testing.assert_array_equal(policy.points['mocap_left']['quat'], [1.0, 0.0, 0.0, 0.0])
    assert 'mocap_right' in policy.points
    np.testing.assert_array_equal(policy.points['mocap_right']['pos'], [0.31718881, 0.49999888, 0.29525084])
    np.testing.assert_array_equal(policy.points['mocap_right']['quat'], [1.0, 0.0, 0.0, 0.0])

    assert trajectories is not None
    assert len(trajectories) == 2

    # Check the left trajectory
    left_trajectory = trajectories[0]
    assert len(left_trajectory) == 12

    # Check the right trajectory
    right_trajectory = trajectories[1]
    assert len(right_trajectory) == 11
