import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from quat_math import euler2quat, quaternion_multiply, quaternion_rotate, mul_pose


def test_quaternion_multiply():
    q1 = np.array([1, 0, 0, 0])
    q2 = np.array([1, 0, 0, 0])
    q3 = quaternion_multiply(q1, q2)
    assert np.allclose(q3, np.array([1, 0, 0, 0]))

    q1 = np.array([0, 1, 0, 0])
    q2 = np.array([0, 0, 1, 0])
    q3 = quaternion_multiply(q1, q2)
    assert np.allclose(q3, np.array([0, 0, 0, 1]))


def test_euler_to_quaternions():
    def _euler2quat(euler):
        rotation = R.from_euler('XYZ', euler, degrees=True)
        quat = rotation.as_quat()
        # Fix the order of the quaternion as (w, x, y, z) from (x, y, z, w)
        quat = [quat[3], quat[0], quat[1], quat[2]]
        return quat

    quat1 = euler2quat([0, 0, 0])
    assert np.allclose(quat1, np.array([1, 0, 0, 0]))
    quat1_v = _euler2quat([0, 0, 0])
    assert np.allclose(quat1, quat1_v)

    euler = [0, 0, 90]
    quat2 = euler2quat(euler)
    assert np.allclose(quat2, np.array([0.70710678, 0, 0, 0.70710678]))
    quat2_v2 = _euler2quat(euler)
    assert np.allclose(quat2, quat2_v2)

    euler = [0, 90, 0]
    quat3 = euler2quat(euler)
    assert np.allclose(quat3, np.array([0.70710678, 0, 0.70710678, 0]))
    quat3_v2 = _euler2quat(euler)
    assert np.allclose(quat3, quat3_v2)

    euler = [90, 0, 0]
    quat4 = euler2quat(euler)
    assert np.allclose(quat4, np.array([0.70710678, 0.70710678, 0, 0]))
    quat4_v2 = _euler2quat(euler)
    assert np.allclose(quat4, quat4_v2)

    euler = [90, 90, 0]
    quat5 = euler2quat(euler)
    assert np.allclose(quat5, np.array([0.5, 0.5, 0.5, 0.5]))
    quat5_v2 = _euler2quat(euler)
    assert np.allclose(quat5, quat5_v2)

    for i in range(100):
        euler = np.random.rand(3) * 360
        quat = euler2quat(euler)
        quat_v2 = _euler2quat(euler)
        assert np.allclose(quat, quat_v2)


def test_quaternion_rotate():
    # Test identity
    quat = np.array([1, 0, 0, 0])
    pos = np.array([1, 0, 0])
    pos_rot = quaternion_rotate(quat, pos)
    assert np.allclose(pos_rot, np.array([1, 0, 0]))

    # Test quaternion rotation by 180 degrees along the x-axis
    quat = np.array([0, 1, 0, 0])
    pos = np.array([1, 0, 0])
    pos_rot = quaternion_rotate(quat, pos)
    assert np.allclose(pos_rot, np.array([1, 0, 0]))

    # Test quaternion rotation by 180 degrees along the y-axis
    quat = np.array([0, 0, 1, 0])
    pos = np.array([1, 0, 0])
    pos_rot = quaternion_rotate(quat, pos)
    assert np.allclose(pos_rot, np.array([-1, 0, 0]))

    # Test quaternion rotation by 180 degrees along the z-axis
    quat = np.array([0, 0, 0, 1])
    pos = np.array([1, 0, 0])
    pos_rot = quaternion_rotate(quat, pos)
    assert np.allclose(pos_rot, np.array([-1, 0, 0]))

    # Test quaternion rotation by 90 degrees along the x-axis
    quat = np.array([0.70710678, 0.70710678, 0, 0])
    pos = np.array([1, 0, 0])
    pos_rot = quaternion_rotate(quat, pos)
    assert np.allclose(pos_rot, np.array([1, 0, 0]))

    # Test quaternion rotation by 90 degrees along the y-axis
    quat = np.array([0.70710678, 0, 0.70710678, 0])
    pos = np.array([1, 0, 0])
    pos_rot = quaternion_rotate(quat, pos)
    assert np.allclose(pos_rot, np.array([0, 0, -1]))

    # Test quaternion rotation by 90 degrees along the z-axis
    quat = np.array([0.70710678, 0.0, 0.0, 0.70710678])
    pos = np.array([1, 0, 0])
    pos_rot = quaternion_rotate(quat, pos)
    assert np.allclose(pos_rot, np.array([0, 1, 0]))

    quat = np.array([0.5, 0.5, -0.5, 0.5])
    pos = np.array([1, 0, 0])
    pos_rot = quaternion_rotate(quat, pos)
    assert np.allclose(pos_rot, np.array([0.0, 0.0, 1.0]))


def test_mju_mul_pose():
    for i in range(100):
        # Set i as the seed
        np.random.seed(i)

        pos1 = np.random.rand(3)
        quat1 = np.random.rand(4)
        quat1 = quat1 / np.linalg.norm(quat1)
        pos2 = np.random.rand(3)
        quat2 = np.random.rand(4)
        quat2 = quat2 / np.linalg.norm(quat2)

        pos = np.zeros(3)
        quat = np.zeros(4)
        mujoco.mju_mulPose(pos, quat, pos1, quat1, pos2, quat2)

        rot_pose = quaternion_rotate(quat1, pos2)
        pos3 = pos1 + rot_pose
        quat3 = quaternion_multiply(quat1, quat2)
        quat3 = quat3 / np.linalg.norm(quat3)

        assert np.allclose(pos, pos3)
        assert np.allclose(quat, quat3)

        assert np.allclose(pos, mul_pose(pos1, quat1, pos2, quat2)[0])
        assert np.allclose(quat, mul_pose(pos1, quat1, pos2, quat2)[1])
