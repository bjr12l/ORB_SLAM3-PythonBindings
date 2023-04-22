import orbslam3
import pandas as pd
import cv2
from pathlib import Path
from typing import Tuple
from pymap3d import geodetic2ned

GENERAL_OFFSET = pd.Timedelta(minutes=7)
# FLIGHT_START_LOGS = pd.Timedelta(minutes=31, seconds=42) + GENERAL_OFFSET
FLIGHT_START_VIDEO = pd.Timedelta(minutes=3, seconds=47) + GENERAL_OFFSET
PATH_TO_VOCABLUARY = Path("Vocabluary") / "ORBvoc.txt"
PATH_TO_CONFIG = Path("examples") / "k2" / "k2.yaml"
PATH_TO_VIDEO = Path("data") / "GX010294.MP4"
# PATH_TO_LOGS = Path("data") / "log 19.12.22.csv"

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

def main(vid: cv2.VideoCapture, path_to_vocab: str, path_to_config: str):
    slam = orbslam3.System(path_to_vocab, path_to_config, orbslam3.Sensor.MONOCULAR)
    slam.set_use_viewer(True)
    slam.initialize()
    for _ in range(5000):
        _, frame = vid.read()
        # log = logs.iloc[i]
        slam.process_image_mono(frame, vid.get(cv2.CAP_PROP_POS_MSEC) / 1000, "")
    return slam.get_trajectory_points()


if __name__ == "__main__":
    # logs = pd.read_csv()
    vid = cv2.VideoCapture(str(PATH_TO_VIDEO))
    vid, fps = sync_video(vid, FLIGHT_START_VIDEO)
    # logs = sync_logs(read_logs(), fps, FLIGHT_START_LOGS)
    trajectory = main(vid, str(PATH_TO_VOCABLUARY), str(PATH_TO_CONFIG))
    # print(trajectory)
