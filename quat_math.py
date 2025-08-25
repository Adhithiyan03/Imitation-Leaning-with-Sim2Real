import numpy as np

"""
Quaternion math functions
Format used in mujoco internally is (w, x, y, z) and stored as (4,) numpy array
"""


def mul_pose(pos1, quat1, pos2, quat2):
    """
    Multiplies two poses, meaning that it transforms the first pose by the second pose (the order is important)
    Meant to be the same as the mujoco function mj_mulPose
    """
    rot_pose = quaternion_rotate(quat1, pos2)
    pos3 = pos1 + rot_pose
    quat3 = quaternion_multiply(quat1, quat2)
    quat3 = quat3 / np.linalg.norm(quat3)

    return pos3, quat3


def quaternion_rotate(quat, pos):
    """
    Rotate a vector by a quaternion
    """
    q_pos = np.concatenate([np.zeros(1), pos])

    # Inverse of the quaternion
    q_inv = np.array([quat[0], -quat[1], -quat[2], -quat[3]])

    # Multiply the quaternion with the vector
    q_rot = quaternion_multiply(quaternion_multiply(quat, q_pos), q_inv)

    return q_rot[1:]


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions which are given in the format (w, x, y, z)
    :param q1:
    :param q2:
    :return:
    """

    # Scalar components
    a1 = q1[0]
    a2 = q2[0]

    # Vector components
    v1 = q1[1:]
    v2 = q2[1:]

    # Compute the scalar component
    a = a1 * a2 - np.dot(v1, v2)

    # Calculate vector part of the product
    v = a1 * v2 + a2 * v1 + np.cross(v1, v2)

    return np.array([a, *v])


def euler2quat(euler, degrees=True, intrinsic=False):
    """
    Convert euler angles to quaternions with the order of rotation as xyz (roll, pitch, yaw)
    Assume all rotations are intrinsic
    :param euler: (3,) array of euler angles in degrees
    :return: (4,) array of quaternions in the order (w, x, y, z)
    """
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]

    if degrees:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)

    c1 = np.cos(roll / 2.)
    c2 = np.cos(pitch / 2.)
    c3 = np.cos(yaw / 2.)
    s1 = np.sin(roll / 2.)
    s2 = np.sin(pitch / 2.)
    s3 = np.sin(yaw / 2.)

    # Quaternion for rotation about z axis
    qz = np.array([c3, 0, 0, s3])
    # Quaternion for rotation about y axis
    qy = np.array([c2, 0, s2, 0])
    # Quaternion for rotation about x axis
    qx = np.array([c1, s1, 0, 0])

    if intrinsic:
        # Intrinsic rotation
        q = quaternion_multiply(qz, quaternion_multiply(qy, qx))
    else:
        # Extrinsic rotation
        q = quaternion_multiply(qx, quaternion_multiply(qy, qz))

    return q
