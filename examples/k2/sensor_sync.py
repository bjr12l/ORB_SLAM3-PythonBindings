import pandas as pd
import cv2
from pymap3d import geodetic2ned
from typing import Tuple

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