from rewards import Rewards

from scripted_yaml_policy import get_yaml


def test_reward_basic():
    """
    Check that the yaml rewards are loaded correctly and that the state progresses correctly
    """
    # Load the yaml data 'temp.yaml' and get the rewards data
    test_script = 'rewards.yaml'
    yaml_data = get_yaml(test_script)
    rewards_data = yaml_data['rewards']

    reward = Rewards(rewards_data)
    assert reward.max_reward == 4

    # Check that we can update the reward state from the list of contact pairs

    # Stay in reward state 0
    contact_pairs = [('door_latch_medium_bumper', 'start_box')]
    reward.update_reward_state(contact_pairs)
    assert reward.reward == 0

    # Stay in reward state 0 since start_box is still touching the door_latch_medium_bumper
    contact_pairs = [('door_latch_medium_bumper', 'start_box'),
                     ('door_latch_medium_bumper', 'vx300s_left/10_left_gripper_finger'),
                     ('door_latch_medium_bumper', 'vx300s_left/10_right_gripper_finger')]
    reward.update_reward_state(contact_pairs)
    assert reward.reward == 0

    # Move to reward state 1 since start_box is not touching the door_latch_medium_bumper
    contact_pairs = [('door_latch_medium_bumper', 'vx300s_left/10_left_gripper_finger'),
                     ('door_latch_medium_bumper', 'vx300s_left/10_right_gripper_finger')]
    reward.update_reward_state(contact_pairs)
    assert reward.reward == 1

    # Stay in reward state 1 until the door_latch_medium_bumper is touching the end_box and the grippers are not
    contact_pairs = [('door_latch_medium_bumper', 'vx300s_left/10_left_gripper_finger'),
                     ('door_latch_medium_bumper', 'vx300s_left/10_right_gripper_finger'),
                     ('door_latch_medium_bumper', 'end_box')]
    reward.update_reward_state(contact_pairs)
    assert reward.reward == 1

    # Move to reward state 2 since the door_latch_medium_bumper is touching the end_box and the grippers are not
    contact_pairs = [('door_latch_medium_bumper', 'end_box')]
    reward.update_reward_state(contact_pairs)
    assert reward.reward == 2

    # Move to reward state 3 where the right grippers are touching the bumper but not touching the end box
    contact_pairs = [('door_latch_medium_bumper', 'vx300s_right/10_left_gripper_finger'),
                     ('door_latch_medium_bumper', 'vx300s_right/10_right_gripper_finger')]
    reward.update_reward_state(contact_pairs)
    assert reward.reward == 3

    # Move to reward state 4 where the right grippers are not touching the medium bumper but the bumper is touching the
    # start box
    contact_pairs = [('door_latch_medium_bumper', 'start_box')]
    reward.update_reward_state(contact_pairs)
    assert reward.reward == 4
