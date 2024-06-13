import pandas as pd
import cv2
import os
from pymap3d import geodetic2ned
from typing import Tuple, List, Generator, Optional
from pathlib import Path
from functools import reduce
from pymavlink import mavutil
from allantools import adev


SENSORS_COLUMNS = ['GyrX', 'GyrY', 'GyrZ', 'AccX', 'AccY', 'AccZ']
ANGLE_COLUMNS = ['DesRoll', 'Roll', 'DesPitch', 'Pitch', 'DesYaw', 'Yaw']
NAVIGATION_COLUMNS = ['Lat', 'Lng', 'Alt']


def read_mavlink_log(logs_path: str) -> pd.DataFrame:
    columns_manager = {
        'IMU': SENSORS_COLUMNS,
        'ATT': ANGLE_COLUMNS,
        'GPS': NAVIGATION_COLUMNS
    }

    allowed_messages = list(columns_manager.keys())
    all_columns: List[str] = reduce(lambda x, y: x + y, list(columns_manager.values()))

    reader = mavutil.mavlink_connection(logs_path)

    data = {}
    types = []
    while True:
        message = reader.recv_match()
        if message is None:
            break

        message_type = message.fmt.name
        types.append(message_type)
        if message_type in allowed_messages:
            message_data = message.to_dict()
            timestamp = message_data['TimeUS']
            if timestamp not in data:
                data[timestamp] = {column: None for column in all_columns}

            for column in columns_manager[message_type]:
                data[timestamp][column] = message_data[column]
    return pd.DataFrame(data).T

def read_logs(log_folder: Path, log_name: str, log_start: Optional[pd.Timedelta] = None, fps: Optional[int] = None):
    raw_log_file = log_folder / f"{log_name}.bin"
    csv_log_file = log_folder / f"{log_name}.csv"
    if csv_log_file.exists():
        logs = pd.read_csv(csv_log_file)
    else:
        logs = read_mavlink_log(str(raw_log_file))
        return logs
        logs.to_csv(raw_log_file, index_label='microseconds_from_start')
    logs = logs.interpolate()
    logs.index = pd.TimedeltaIndex(pd.to_timedelta(logs["microseconds_from_start"], unit="us"))
    logs["Alt"] -= (logs["Alt"].min() - 1)
    if log_start is not None:
        logs = logs.loc[log_start:]
    if fps is not None:
        logs = logs.resample(f"{(1 / fps):.3f}S").mean()
    logs = add_ned_coordinates(logs)
    return logs

def add_ned_coordinates(flight_log: pd.DataFrame) -> pd.DataFrame:
    start_lat, start_lng, start_alt = flight_log.iloc[0][["Lat", "Lng", "Alt"]]
    flight_log[["x", "y", "z"]] = flight_log.apply(
        lambda row: geodetic2ned(row["Lat"], row["Lng"], row["Alt"], start_lat, start_lng, start_alt),
        axis=1
    ).tolist()
    flight_log["z"] = -flight_log["z"]
    return flight_log

def sync_video(vid: cv2.VideoCapture, offset: pd.Timedelta, target_fps: int) -> Generator:
    fps = vid.get(cv2.CAP_PROP_FPS)
    print(f"Source FPS: {fps}")
    first_frames_skip_count = int(fps * offset.seconds) 

    vid.set(cv2.CAP_PROP_POS_FRAMES, first_frames_skip_count)
    frames_to_skip = int(fps // target_fps)

    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = int((total_frames - first_frames_skip_count) // frames_to_skip)

    for i in range(frames_to_process):
        for _ in range(frames_to_skip - 1):
            vid.grab()  # Read and discard the frames to be skipped
        _, frame = vid.read()
        timestamp = pd.Timedelta(seconds=(i * frames_to_skip) / fps)
        yield frame, timestamp


def estimate_imu_noise(logs: pd.DataFrame):
    gyr_arr = pd.concat([logs["GyrX"], logs["GyrY"], logs["GyrZ"]]).to_numpy()
    acc_arr = pd.concat([logs["AccX"], logs["AccY"], logs["AccZ"]]).to_numpy()

    (tau, gyro_allan, _, _) = adev(data=gyr_arr, data_type="freq")
    (_, accel_allan, _, _) = adev(data=acc_arr, data_type="freq")

    # Approximate the IMU noise
    IMU_NoiseGyro = gyro_allan[-1] ** 0.5  # Allan deviation at the largest tau
    IMU_NoiseAcc = accel_allan[-1] ** 0.5  # Allan deviation at the largest tau

    # Approximate the GyroWalk and AccWalk from the minimum of Allan deviation
    IMU_GyroWalk = gyro_allan.min()
    IMU_AccWalk = accel_allan.min()

    # Approximate the IMU frequency
    IMU_Frequency = 1 / logs["microseconds_from_start"].diff().mean() * 1e6  # microseconds to seconds

    print("IMU.NoiseGyro:", IMU_NoiseGyro)
    print("IMU.NoiseAcc:", IMU_NoiseAcc)
    print("IMU.GyroWalk:", IMU_GyroWalk)
    print("IMU.AccWalk:", IMU_AccWalk)
    print("IMU.Frequency:", IMU_Frequency)
