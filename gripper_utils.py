def gripper_normalize(value, close_value, open_value):
    """
    Normalize gripper position to [0, 1] range.

    :param value: the gripper position
    :param close_value: the gripper position when closed
    :param open_value: the gripper position when open
    :return: the normalized gripper position where 0 is closed and 1 is open (irrespective of the actual direction
    of the gripper or the direction it moves to open/close)
    """
    return (value - close_value) / (open_value - close_value)


def gripper_un_normalize(value, close_value, open_value):
    """
    Un-normalize gripper position from [0, 1] range.

    :param value: the normalized gripper position where 0 is closed and 1 is open (irrespective of the actual direction
    of the gripper or the direction it moves to open/close)
    :param close_value: the gripper position when closed
    :param open_value: the gripper position when open
    :return: the gripper position
    """
    return value * (open_value - close_value) + close_value
