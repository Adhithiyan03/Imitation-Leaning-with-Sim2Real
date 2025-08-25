from gripper_utils import gripper_normalize, gripper_un_normalize


def test_gripper_normalize():
    assert gripper_normalize(0.0, 0.0, 1.0) == 0.0
    assert gripper_normalize(1.0, 0.0, 1.0) == 1.0
    assert gripper_normalize(0.5, 0.0, 1.0) == 0.5

    assert gripper_normalize(0.5, 0.5, 1.0) == 0.0
    assert gripper_normalize(1.0, 0.5, 1.0) == 1.0
    assert gripper_normalize(0.75, 0.5, 1.0) == 0.5

    assert gripper_normalize(0.0, 0.0, -0.5) == 0.0
    assert gripper_normalize(-0.5, 0.0, -0.5) == 1.0
    assert gripper_normalize(-0.25, 0.0, -0.5) == 0.5


def test_gripper_un_normalize():
    assert gripper_un_normalize(0.0, 0.0, 1.0) == 0.0
    assert gripper_un_normalize(1.0, 0.0, 1.0) == 1.0
    assert gripper_un_normalize(0.5, 0.0, 1.0) == 0.5

    assert gripper_un_normalize(0.5, 0.5, 1.0) == 0.75
    assert gripper_un_normalize(1.0, 0.5, 1.0) == 1.0
    assert gripper_un_normalize(0.75, 0.5, 1.0) == 0.875

    assert gripper_un_normalize(0.0, 0.0, -0.5) == 0.0
    assert gripper_un_normalize(1.0, 0.0, -0.5) == -0.5
    assert gripper_un_normalize(0.5, 0.0, -0.5) == -0.25
