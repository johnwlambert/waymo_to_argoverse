"""
Unit tests on rigid body transformation utilities.

Authors: John Lambert
"""

from typing import Tuple


import numpy as np
from scipy.spatial.transform import Rotation


def test_quaternion3d_to_yaw() -> None:
    """ """
    for yaw in np.linspace(-np.pi, np.pi, 100000):
        qx, qy, qz, qw = yaw_to_quaternion3d(yaw)
        q_argo = np.array([qw, qx, qy, qz])
        new_yaw = quaternion3d_to_yaw(q_argo)
        if not np.allclose(yaw, new_yaw):
            print(yaw, new_yaw)


def se2_to_yaw(B_SE2_A) -> float:
    """Computes the pose vector v from a homogeneous transform A.
    
    Args:
        B_SE2_A
    
    Returns:
        theta:
    """
    R = B_SE2_A.rotation
    theta = np.arctan2(R[1, 0], R[0, 0])
    return theta


def quaternion3d_to_yaw(q: np.ndarray) -> float:
    """
    Args:
        q: qx,qy,qz,qw: quaternion coefficients

    Returns:
        yaw: float, rotation about the z-axis
    """
    w, x, y, z = q  # in argo format
    q_scipy = x, y, z, w
    R = Rotation.from_quat(q_scipy).as_dcm()
    # tan (yaw) = s / c
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return yaw


def yaw_to_quaternion3d(yaw: float) -> Tuple[float, float, float, float]:
    """
    Args:
        yaw: rotation about the z-axis

    Returns:
        qx,qy,qz,qw: quaternion coefficients
    """
    qx, qy, qz, qw = Rotation.from_euler("z", yaw).as_quat()
    return qx, qy, qz, qw


def rotmat2quat(R: np.ndarray) -> np.ndarray:
    """ """
    q_scipy = Rotation.from_dcm(R).as_quat()
    x, y, z, w = q_scipy
    q_argo = w, x, y, z
    return q_argo


def quat2rotmat(q: np.ndarray) -> np.ndarray:
    """Convert a unit-length quaternion into a rotation matrix.
    Note that libraries such as Scipy expect a quaternion in scalar-last [x, y, z, w] format,
    whereas at Argo we work with scalar-first [w, x, y, z] format, so we convert between the
    two formats here. We use the [w, x, y, z] order because this corresponds to the
    multidimensional complex number `w + ix + jy + kz`.
    
    Args:
        q: Array of shape (4,) representing (w, x, y, z) coordinates
    
    Returns:
        R: Array of shape (3, 3) representing a rotation matrix.
    """
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-12)
    w, x, y, z = q
    q_scipy = np.array([x, y, z, w])
    return Rotation.from_quat(q_scipy).as_dcm()


def test_cycle() -> None:
    """ """
    R = np.eye(3)
    q = rotmat2quat(R)
    R_cycle = quat2rotmat(q)


def rotMatZ_3D(yaw: float) -> np.ndarray:
    """
    Args:
        yaw: angle in radians
    
    Returns:
        rot_z
    """
    c = np.cos(yaw)
    s = np.sin(yaw)
    rot_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return rot_z


def test_transform() -> None:
    """ """
    yaw = 90
    for yaw in np.random.randn(10) * 360:
        R = rotMatZ_3D(yaw)
        w, x, y, z = rotmat2quat(R)
        qx, qy, qz, qw = yaw_to_quaternion3d(yaw)

        print(w, qw)
        print(x, qx)
        print(y, qy)
        print(z, qz)
        assert np.allclose(w, qw)
        assert np.allclose(x, qx)
        assert np.allclose(y, qy)
        assert np.allclose(z, qz)


def rotX(deg: float) -> np.ndarray:
    """Compute 3x3 rotation matrix about the X-axis.

    Args:
        deg: Euler angle in degrees
    """
    t = np.deg2rad(deg)
    return Rotation.from_euler("x", t).as_dcm()


def rotZ(deg: float) -> np.ndarray:
    """Compute 3x3 rotation matrix about the Z-axis.

    Args:
        deg: Euler angle in degrees
    """
    t = np.deg2rad(deg)
    return Rotation.from_euler("z", t).as_dcm()


def rotY(deg: float) -> np.ndarray:
    """Compute 3x3 rotation matrix about the Y-axis.

    Args:
        deg: Euler angle in degrees
    """
    t = np.deg2rad(deg)
    return Rotation.from_euler("y", t).as_dcm()
