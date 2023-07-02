from pathlib import Path
import cv2
import yaml
import pandas as pd
import orbslam3
from tqdm import tqdm

from orbslam_k2 import ORBSLAM3
from sensor_utils import sync_video, read_logs

PROJECT_DIR = Path("..") / ".."
VOCABLUARY_PATH = PROJECT_DIR / "Vocabluary" / "ORBvoc.txt"

CONFIG_FOLDER = Path("copter_flir") # Path("day7") / "video_cam0_13_00_34_02-03-23_flight2"
CONFIG_PATH = CONFIG_FOLDER / "config.yaml"
ORB_CONFIG_PATH = CONFIG_FOLDER / "orb_config.yaml"

DATA_DIR = PROJECT_DIR / "data" / "copter_flir" # "day7"

def timedelta_from_string(timestamp_str: str) -> pd.Timedelta:
    minutes, seconds, milliseconds = timestamp_str.split(":")
    return pd.Timedelta(minutes=int(minutes), seconds=int(seconds), milliseconds=int(milliseconds))


def display_frame_with_log(frame, log):
    # Add log information to the frame
    if log:
        log_text = f"Log info: x={log['x']:.2f}, y={log['y']:.2f}, z={log['z']:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 0, 0)  # White text
        font_thickness = 2
        x, y = 10, 30  # Position of the text on the frame
        cv2.putText(frame, log_text, (x, y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Main", frame)
    cv2.waitKey(1)


if __name__ == "__main__":
    with open(str(CONFIG_PATH), "r") as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)
    
    offset = timedelta_from_string(config["offset"])
    log_start = timedelta_from_string(config.get("log_start", "0:0:0")) + offset
    video_start = timedelta_from_string(config.get("video_start", "0:0:0")) + offset

    target_fps = int(config["target_fps"])

    print(f"Starting video from {video_start} at {target_fps} FPS")

    vid = cv2.VideoCapture(str(DATA_DIR / config["video_name"]))
    vid = sync_video(vid, video_start, target_fps)

    # logs = read_logs(DATA_DIR, config["log_name"], log_start=log_start)
    # logs.index -= logs.index[0]
    # logs["timestamp"] = (logs.index.total_seconds()).astype(int)

    slam = ORBSLAM3(str(VOCABLUARY_PATH), str(ORB_CONFIG_PATH), orbslam3.Sensor.MONOCULAR, True)

    prev_timestamp = pd.Timedelta(seconds=0)
    for i in tqdm(range(20000)):
        frame, timestamp = next(vid)
        frame = cv2.resize(frame, slam.new_frame_size)
        # frame_log = logs.loc[prev_timestamp:timestamp]
        display_frame_with_log(frame, None) # logs.loc[prev_timestamp:].iloc[0])
        slam.step(frame, None, timestamp)
        prev_timestamp = timestamp
