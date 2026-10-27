import pandas as pd
import numpy as np

df = pd.read_csv("delivery_data.csv")
print("raw rows:", len(df))

for col in ["trip_creation_time", "od_start_time", "od_end_time"]:
    df[col] = pd.to_datetime(df[col], format="mixed")

key = ["trip_uuid", "source_center", "destination_center"]
df = df.sort_values("od_start_time")

seg = df.groupby(key, as_index=False).agg(
    data=("data", "first"),
    route_type=("route_type", "first"),
    source_name=("source_name", "first"),
    destination_name=("destination_name", "first"),
    od_start_time=("od_start_time", "first"),
    od_end_time=("od_end_time", "last"),
    actual_time=("segment_actual_time", "sum"),
    osrm_time=("segment_osrm_time", "sum"),
    osrm_distance=("segment_osrm_distance", "sum"),
    actual_distance=("actual_distance_to_destination", "last"),
    n_scans=("trip_uuid", "size"),
)
print("segments after aggregation:", len(seg))

before = len(seg)
seg = seg[(seg.actual_time > 0) & (seg.osrm_time > 0)]
seg = seg[seg.osrm_distance > 0.5]
seg["delay_ratio"] = seg.actual_time / seg.osrm_time
seg = seg[seg.delay_ratio.between(0.2, 20)]
print("dropped", before - len(seg), "bad segments")

seg["hour"] = seg.od_start_time.dt.hour
seg["dow"] = seg.od_start_time.dt.dayofweek
seg["time_of_day"] = pd.cut(
    seg.hour, bins=[-1, 5, 11, 17, 23],
    labels=["night", "morning", "afternoon", "evening"]
)
seg["src_state"] = seg.source_name.str.extract(r"\(([^)]+)\)")
seg["dst_state"] = seg.destination_name.str.extract(r"\(([^)]+)\)")
seg["is_late"] = (seg.delay_ratio > 1.2).astype(int)

seg.to_pickle("outputs/segments.pkl")

print(seg[["actual_time", "osrm_time", "delay_ratio"]].describe().round(2))
print("share late:", round(seg.is_late.mean(), 3))
print(seg.groupby("route_type")[["delay_ratio", "is_late"]].mean().round(3))
print("facilities:", pd.concat([seg.source_center, seg.destination_center]).nunique())
print("corridors:", seg.groupby(["source_center", "destination_center"]).ngroups)
