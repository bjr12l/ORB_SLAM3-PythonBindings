from pathlib import Path
import cv2
import yaml
import pandas as pd
import orbslam3
from tqdm import tqdm

PROJECT_DIR = Path("..") / ".."
VOCABLUARY_PATH = PROJECT_DIR / "Vocabluary" / "ORBvoc.txt"

VIDEO_FOLDER = Path("cam1_lens100_alt600")
CONFIG_PATH = VIDEO_FOLDER / "config.yaml"
ORB_CONFIG_PATH = VIDEO_FOLDER / "orb_config.yaml"

DATA_DIR = PROJECT_DIR / "data" / "k2" / VIDEO_FOLDER

from orbslam_k2 import ORBSLAM3
from sensor_utils import sync_video, read_synced_logs, convert_logs_to_csv

def timedelta_from_string(timestamp_str: str) -> pd.Timedelta:
    minutes, seconds, milliseconds = timestamp_str.split(":")
    return pd.Timedelta(minutes=int(minutes), seconds=int(seconds), milliseconds=int(milliseconds))

if __name__ == "__main__":
    with open(str(CONFIG_PATH), "r") as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)
    
    offset = timedelta_from_string(config["offset"])
    log_start = timedelta_from_string(config["log_start"]) + offset
    video_start = timedelta_from_string(config["video_start"]) + offset

    print(f"Starting video from {video_start}")

    vid = cv2.VideoCapture(str(DATA_DIR / config["video_name"]))
    vid, fps = sync_video(vid, video_start)

    logs = read_synced_logs(DATA_DIR / config["csv_log_name"], fps, log_start)

    slam = ORBSLAM3(str(VOCABLUARY_PATH), str(ORB_CONFIG_PATH), orbslam3.Sensor.MONOCULAR, True)

    for i in tqdm(range(5000)):
        _, frame = vid.read()
        log = logs.iloc[i]
        slam.step(frame, log)
