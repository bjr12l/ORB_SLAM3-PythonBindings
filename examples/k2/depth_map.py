import cv2
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.spatial.transform import Rotation


def depth_map_from_log_and_image(image: np.ndarray, log: pd.Series, intrinsic_matrix) -> np.ndarray:
    pitch, roll, yaw = np.radians(log[["Pitch", "Roll", "Yaw"]])
    altitude = log["Alt"]
    R = eulerAnglesToRotationMatrix(pitch, roll, yaw)
    depth_map = depth_map_from_rotation_translation(altitude, R, image.shape, intrinsic_matrix)
    display_depth_map(depth_map, pitch, roll, yaw, altitude)
    return depth_map

def eulerAnglesToRotationMatrix(pitch, roll, yaw):
    # Create a rotation object from Euler angles
    rotation_obj = Rotation.from_euler('xyz', [pitch, roll, yaw], degrees=False)
    # Convert the rotation object to a rotation matrix
    R = rotation_obj.as_matrix()
    return R

def depth_map_from_rotation_translation(
    altitude: float, 
    R: np.ndarray, 
    image_shape: Tuple[float, float],
    intrinsic_matrix: np.ndarray = None
):
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
     # Generate plane points
    plane_points = generate_flat_plane(image_shape, cx, cy)
    # Apply the rotation matrix
    rotated_points = np.dot(plane_points, R.T)
    # Translate the plane to the given distance
    translated_points = rotated_points + np.array([0, 0, altitude])
    # Compute the distance from the origin to each pixel
    distances = np.linalg.norm(translated_points, axis=-1)
    return distances


def display_depth_map(depth_map, pitch, roll, yaw, altitude):
    # Find the minimum and maximum depth values
    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)
    
    # Normalize the depth map to a range of 0-255 for visualization
    normalized_depth_map = ((depth_map - min_depth) / (max_depth - min_depth)) * 255
    normalized_depth_map = normalized_depth_map.astype(np.uint8)

    colored_depth_map = cv2.applyColorMap(normalized_depth_map, cv2.COLORMAP_VIRIDIS)

    # Overlay pitch, roll, yaw, and altitude information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)
    thickness = 2
    cv2.putText(colored_depth_map, f"Pitch: {pitch:.2f}", (10, 30), font, font_scale, font_color, thickness)
    cv2.putText(colored_depth_map, f"Roll: {roll:.2f}", (10, 60), font, font_scale, font_color, thickness)
    cv2.putText(colored_depth_map, f"Yaw: {yaw:.2f}", (10, 90), font, font_scale, font_color, thickness)
    cv2.putText(colored_depth_map, f"Altitude: {altitude:.2f}", (10, 120), font, font_scale, font_color, thickness)

    # Display the depth map
    cv2.imshow('Depth Map', colored_depth_map)
    cv2.waitKey(1)

def generate_flat_plane(image_shape, cx: float, cy: float):
    x = np.linspace(0, image_shape[1] - 1, image_shape[1])
    y = np.linspace(0, image_shape[0] - 1, image_shape[0])
    X, Y = np.meshgrid(x, y)
    X -= cx
    Y -= cy
    Z = np.zeros_like(X)
    plane_points = np.stack((X, Y, Z), axis=-1)
    return plane_points