import orbslam3
import pandas as pd
import numpy as np
import cv2
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import Tuple
from depth_map import depth_map_from_log_and_image


class ORBSLAM3:
    slam: orbslam3.System
    camera_matrix: np.ndarray
    sensor: orbslam3.Sensor

    new_frame_size: Tuple[float, float]

    def __init__(self, path_to_vocab: Path, path_to_config: Path, sensor: orbslam3.Sensor, display: bool):
        self.slam = orbslam3.System(str(path_to_vocab), str(path_to_config), sensor)
        self.slam.set_use_viewer(display)
        self.slam.initialize()
        self.sensor = sensor
        
        self.new_frame_size = self.get_new_camera_dim(path_to_config)
    
    def step(self, frame, log):
        if self.sensor == orbslam3.Sensor.MONOCULAR:
            self.slam_step_mono(frame, log)
        elif self.sensor == orbslam3.Sensor.RGBD:
            self.slam_step_rgbd(frame, log)
        else:
            raise NotImplementedError(f"Sensor {self.sensor} not supported yet")
    
    def slam_step_mono(self, frame, log):
        self.slam.process_image_mono(frame, log["seconds_from_start"], "")

    def slam_step_mono_imu(self, frame, log):
        log_array = log[
            ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'seconds_from_start']
        ].to_numpy(dtype=np.float32).reshape(1, -1)
        self.slam.process_image_imu_mono(frame, log["seconds_from_start"], "")

    def slam_step_rgbd(self, frame, log):
        frame = cv2.resize(frame, self.new_frame_size)
        if intrinsic_matrix is None:
            depth_map = np.full((frame.shape[0], frame.shape[1]), log["z"])
            intrinsic_matrix = self.slam.get_camera_matrix()
        else: 
            depth_map = depth_map_from_log_and_image(frame, log, intrinsic_matrix)
        self.slam.process_image_rgbd(frame, depth_map, log["seconds_from_start"])

    def slam_step_rgbd_imu():
        ...

    def get_trajectory_df(self) -> pd.DataFrame:
        trajectory = self.slam.get_trajectory_points()
        orb_slam3_positions = []
        timestamps = []
        for timestamp, pose in trajectory:
            position = np.array([pose[0, 3], pose[1, 3], pose[2, 3]])
            orb_slam3_positions.append(position)
            timestamps.append(timestamp)
        predicted_pos = pd.DataFrame.from_records(
            orb_slam3_positions, columns=["x", "y", "z"]
        )
        predicted_pos.index = pd.TimedeltaIndex(timestamps, unit="s")
        return predicted_pos

    @staticmethod
    def get_new_camera_dim(path_to_config: Path) -> Tuple[int, int]:
        new_width = new_height = None
        with open(path_to_config, "r") as f:
            for line in f:
                if line.startswith("Camera.newWidth"):
                    new_width = int(line.split(":")[1].strip())
                elif line.startswith("Camera.newHeight"):
                    new_height = int(line.split(":")[1].strip())
                # Break the loop once both fields are found
                if new_width is not None and new_height is not None:
                    return new_width, new_height