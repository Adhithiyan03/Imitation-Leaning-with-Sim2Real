from sim_yaml_env import YAMLSimTask, YAMLSimEETask
import yaml
import dm_control.mujoco as mujoco


def load_reward_yaml():
    """
    Read in the reward yaml file rewards.yaml
    The rewards yaml show each step where rewards are handed out
    """
    with open('rewards.yaml', 'r') as f:
        rewards = yaml.safe_load(f)
    return rewards


def load_physics(mocap_flag=False):
    """
    Load the physics from the mujoco xml file
    """
    model_name = '../assets/bimanual_viperx_ee_tape_free.xml' if mocap_flag else '../assets/bimanual_viperx_tape_free.xml'
    physics = mujoco.Physics.from_xml_path(model_name)
    return physics


def test_create_yaml_sim_env():
    """
    Test the creation of the yaml sim env
    """
    reward_data = load_reward_yaml()
    task = YAMLSimTask(args={'reward_data': reward_data['rewards']})

    # Load the model from the xml file
    physics = load_physics()

    task.initialize_episode(physics)
    state = task.get_env_state(physics)
    assert state.shape == (0,)
    reward = task.get_reward(physics)
    assert reward == 0.0


def test_create_yaml_sim_ee_env():
    """
    Test the creation of the yaml sim ee env
    """
    reward_data = load_reward_yaml()
    task = YAMLSimEETask(args={'reward_data': reward_data['rewards']})

    # Load the model from the xml file
    physics = load_physics(mocap_flag=True)

    task.initialize_episode(physics)
    state = task.get_env_state(physics)
    assert state.shape == (0,)
    reward = task.get_reward(physics)
    assert reward == 0.0
