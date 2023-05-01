import pandas as pd
import cv2
import os
from pymap3d import geodetic2ned
from typing import Tuple, List
from pathlib import Path
from functools import reduce
from pymavlink import mavutil
from allantools import adev


SENSORS_COLUMNS = ['GyrX', 'GyrY', 'GyrZ', 'AccX', 'AccY', 'AccZ']
ANGLE_COLUMNS = ['DesRoll', 'Roll', 'DesPitch', 'Pitch', 'DesYaw', 'Yaw']
NAVIGATION_COLUMNS = ['Lat', 'Lng', 'Alt']


def convert_logs_to_csv(logs_path: str, output_path: str) -> pd.DataFrame:
    if os.path.exists(output_path):
        return pd.read_csv(output_path).set_index("microseconds_from_start", drop=False)
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

    df = pd.DataFrame(data).T
    df = df.fillna(method='ffill')
    if output_path:
        df.to_csv(output_path, index_label='microseconds_from_start')
    return df

def read_synced_logs(path_to_logs: Path, fps, log_start):
    logs = pd.read_csv(path_to_logs).sort_values(by="microseconds_from_start")
    logs["Alt"] -= (logs["Alt"].min() - 1)
    logs = sync_logs(logs, fps, log_start)
    logs["seconds_from_start"] = logs.index.total_seconds()
    return logs

def sync_logs(flight_log: pd.DataFrame, fps: int, log_offset: pd.Timedelta) -> pd.DataFrame:
    flight_log.index = pd.TimedeltaIndex(pd.to_timedelta(flight_log["microseconds_from_start"], unit="us"))
    flight_log = flight_log.loc[log_offset:].resample(f"{(1 / fps):.3f}S").mean()
    start_lat, start_lng, start_alt = flight_log.iloc[0][["Lat", "Lng", "Alt"]]
    flight_log[["x", "y", "z"]] = flight_log.apply(
        lambda row: geodetic2ned(row["Lat"], row["Lng"], row["Alt"], start_lat, start_lng, start_alt),
        axis=1
    ).tolist()
    flight_log["z"] = -flight_log["z"]
    return flight_log

def sync_video(vid: cv2.VideoCapture, ofset: pd.Timedelta) -> Tuple[cv2.VideoCapture, int]:
    fps = vid.get(cv2.CAP_PROP_FPS)
    vid.set(cv2.CAP_PROP_POS_FRAMES, int(fps * ofset.seconds))
    return vid, fps

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
