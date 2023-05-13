import orbslam3
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple
from depth_map import depth_map_from_log_and_image


class ORBSLAM3:
    slam: orbslam3.System
    camera_matrix: np.ndarray
    sensor: orbslam3.Sensor

    new_frame_size: Tuple[float, float]

    prev_timestamp: pd.Timedelta

    def __init__(self, path_to_vocab: Path, path_to_config: Path, sensor: orbslam3.Sensor, display: bool):
        self.slam = orbslam3.System(str(path_to_vocab), str(path_to_config), sensor)
        self.slam.set_use_viewer(display)
        self.slam.initialize()
        self.sensor = sensor
        
        self.new_frame_size = self._get_new_camera_dim(path_to_config)
    
    def step(self, frame: np.ndarray, frame_logs: pd.DataFrame, timestamp: pd.Timedelta):
        if self.sensor == orbslam3.Sensor.MONOCULAR:
            self._slam_step_mono(frame, timestamp)
        elif self.sensor == orbslam3.Sensor.IMU_MONOCULAR:
            self._slam_step_mono_imu(frame, timestamp, frame_logs)
        elif self.sensor == orbslam3.Sensor.RGBD:
            self._slam_step_rgbd(frame, timestamp, frame_logs)
        else:
            raise NotImplementedError(f"Sensor {self.sensor} not supported yet")
    
    def _slam_step_mono(self, frame: np.ndarray, timestamp: pd.Timedelta):
        self.slam.process_image_mono(frame, timestamp.total_seconds(), "")

    def _slam_step_mono_imu(self, frame: np.ndarray, timestamp: pd.Timedelta, frame_logs: pd.DataFrame):
        imu_data = frame_logs[['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', "timestamp"]].values
        self.slam.process_image_imu_mono(frame, timestamp.total_seconds(), "", imu_data)

    def _slam_step_rgbd(self, frame: np.ndarray, timestamp: pd.Timedelta, logs: pd.DataFrame):
        frame = cv2.resize(frame, self.new_frame_size)
        if intrinsic_matrix is None:
            depth_map = np.full((frame.shape[0], frame.shape[1]), logs["z"])
            intrinsic_matrix = self.slam.get_camera_matrix()
        else: 
            depth_map = depth_map_from_log_and_image(frame, logs, intrinsic_matrix)
        self.slam.process_image_rgbd(frame, depth_map, timestamp)

    def _slam_step_rgbd_imu():
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
    def _get_new_camera_dim(path_to_config: Path) -> Tuple[int, int]:
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