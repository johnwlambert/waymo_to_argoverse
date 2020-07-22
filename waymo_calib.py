# changes to calibration.py for argoverse-api to support Waymo Open Dataset

if camera_name in ['ring_front_center','ring_front_left', 'ring_front_right']:
    img_width = 1920
    img_height = 1280
elif camera_name in ['ring_side_left', 'ring_side_right']:
    img_width = 1920
    img_height = 886

### FOR WAYMO ONLY
standardcam_R_waymocam = rotY(-90).dot(rotX(90))
standardcam_SE3_waymocam = SE3(rotation=standardcam_R_waymocam, translation=np.zeros(3))
waymocam_SE3_egovehicle = SE3(rotation=camera_config.extrinsic[:3,:3], translation=camera_config.extrinsic[:3,3])
standardcam_SE3_egovehicle = standardcam_SE3_waymocam.right_multiply_with_se3(waymocam_SE3_egovehicle)
camera_config.extrinsic = standardcam_SE3_egovehicle.transform_matrix
###

+from scipy.spatial.transform import Rotation
+
+def rotX(deg: float):
+    """
+    Compute rotation matrix about the X-axis.
+    Args:
+    -   deg: in degrees
+    
+    rot_z = Rotation.from_euler('z', yaw).as_dcm()
+    """
+    t = np.deg2rad(deg)
+    return Rotation.from_euler('x', t).as_dcm()
+
+def rotZ(deg: float):
+    """
+    Compute rotation matrix about the Z-axis.
+    Args
+    -   deg: in degrees
+    """
+    t = np.deg2rad(deg)
+    return Rotation.from_euler('z', t).as_dcm()
+
+def rotY(deg: float):
+    """
+    Compute rotation matrix about the Y-axis.
+    Args
+    -   deg: in degrees
+    """
+    t = np.deg2rad(deg)
+    return Rotation.from_euler('y', t).as_dcm()
+
