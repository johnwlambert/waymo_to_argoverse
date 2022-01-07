"""Unit tests on coordinate transform utilities."""

import waymo2argo.transform_utils as transform_utils


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

        
def test_cycle() -> None:
    """ """
    R = np.eye(3)
    q = rotmat2quat(R)
    R_cycle = quat2rotmat(q)
    assert np.allclose(R, R_cycle)


def test_quaternion3d_to_yaw() -> None:
    """ """
    num_trials = 100000
    for yaw in np.linspace(-np.pi, np.pi, num_trials):
        qx, qy, qz, qw = yaw_to_quaternion3d(yaw)
        q_argo = np.array([qw, qx, qy, qz])
        new_yaw = quaternion3d_to_yaw(q_argo)
        assert np.isclose(yaw, new_yaw)
        if not np.allclose(yaw, new_yaw):
            print(yaw, new_yaw)
