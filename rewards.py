class Rewards():
    """
    Class to handle the reward state.
    States are advanced by checking if the contact pairs are in the reward data
    """
    def __init__(self, reward_data):
        self.reward = 0
        self.reward_data = reward_data
        self.max_reward = len(reward_data)

    def update_reward_state(self, contact_pairs):
        """
        Update the reward state based on the list of contact pairs
        """

        # Don't do anything if we are already at the max reward
        if self.reward == self.max_reward:
            return

        # Add contacts_pairs that are flipped of the current contact pairs
        flipped_contact_pairs = [(b, a) for a, b in contact_pairs]
        contact_pairs += flipped_contact_pairs

        # This is the current reward data for the reward state
        current_reward_data = self.reward_data[self.reward]

        # Check if we have the 'y' and 'n' keys to make the data
        if 'y' in current_reward_data.keys():
            current_yes_reward_data = current_reward_data['y']
        else:
            current_yes_reward_data = []

        if 'n' in current_reward_data.keys():
            current_no_reward_data = current_reward_data['n']
        else:
            current_no_reward_data = []

        # All the yes and no contact pairs for the current reward state
        if current_yes_reward_data is None or len(current_yes_reward_data) == 0:
            y_contact_pairs = []
        else:
            y_contact_pairs = [(a, b) for a, b in zip(current_yes_reward_data[::2], current_yes_reward_data[1::2])]

        if current_no_reward_data is None or len(current_no_reward_data) == 0:
            n_contact_pairs = []
        else:
            n_contact_pairs = [(a, b) for a, b in zip(current_no_reward_data[::2], current_no_reward_data[1::2])]

        # Check if the yes contact pairs are in the contact pairs
        y_in_contact_pairs = all([y in contact_pairs for y in y_contact_pairs])

        # Check if the no contact pairs are not in the contact pairs
        n_in_contact_pairs = all([n not in contact_pairs for n in n_contact_pairs])

        # If y are in contact pairs and n are not in, then we can move to the next reward state
        if y_in_contact_pairs and n_in_contact_pairs:
            self.reward += 1
            print('Reward achieved! Switching to next reward state of state: ', self.reward)
