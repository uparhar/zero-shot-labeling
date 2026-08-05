"""Convert a ROS2 bag's image topic into a playable .mp4, without needing
ROS2 installed. Reads the bag directly via the mcap format, tolerating a
missing metadata.yaml or a truncated/uncleanly-closed recording (both
common when a bag gets killed mid-recording on the robot).

Usage:
    python tools/bag_to_video.py <bag_dir_or_mcap_path> [--topic /ogopogo/front/color/image_raw] [--out out.mp4]
    python tools/bag_to_video.py <bag_dir_or_mcap_path> --list-topics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from mcap.stream_reader import StreamReader
from mcap.records import Message, Channel, Schema
from mcap.exceptions import EndOfFile
from mcap_ros2.decoder import Decoder

IMAGE_SCHEMA = "sensor_msgs/msg/Image"


def resolve_mcap_path(bag_path: Path) -> Path:
    """Accept either a direct .mcap file or a bag folder containing exactly one."""
    if bag_path.is_file():
        return bag_path
    mcap_files = list(bag_path.glob("*.mcap"))
    if not mcap_files:
        raise FileNotFoundError(f"No .mcap file found in {bag_path}")
    if len(mcap_files) > 1:
        raise ValueError(f"Multiple .mcap files in {bag_path}, pass one directly: {mcap_files}")
    return mcap_files[0]


def list_topics(mcap_path: Path) -> None:
    channels, schemas, counts = {}, {}, {}
    with mcap_path.open("rb") as f:
        reader = StreamReader(f, skip_magic=False)
        try:
            for record in reader.records:
                if isinstance(record, Schema):
                    schemas[record.id] = record
                elif isinstance(record, Channel):
                    channels[record.id] = record
                elif isinstance(record, Message):
                    ch = channels.get(record.channel_id)
                    if ch:
                        schema = schemas.get(ch.schema_id)
                        key = (ch.topic, schema.name if schema else "?")
                        counts[key] = counts.get(key, 0) + 1
        except EndOfFile:
            pass

    for (topic, schema_name), count in sorted(counts.items()):
        print(f"{topic:55s} {schema_name:45s} count={count}")


def image_msg_to_bgr(msg):
    h, w = msg.height, msg.width
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if msg.encoding == "rgb8":
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if msg.encoding == "bgr8":
        return data.reshape(h, w, 3)
    if msg.encoding == "mono8":
        return cv2.cvtColor(data.reshape(h, w), cv2.COLOR_GRAY2BGR)
    if msg.encoding == "16UC1":
        # depth: normalize + colorize so it's actually viewable as video
        arr = data.view(np.uint16).reshape(h, w)
        valid = arr[arr > 0]
        if valid.size == 0:
            return np.zeros((h, w, 3), dtype=np.uint8)
        norm = np.clip(arr, 0, np.percentile(valid, 99)).astype(np.float32)
        norm = (norm / max(norm.max(), 1) * 255).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    raise ValueError(f"unhandled encoding: {msg.encoding}")


def bag_to_video(mcap_path: Path, topic: str, out_path: Path, fps: float | None, show_frame_number: bool) -> None:
    channels, schemas = {}, {}
    decoder = Decoder()

    writer = None
    stamps: list[float] = []
    n_written = 0
    size = None

    with mcap_path.open("rb") as f:
        reader = StreamReader(f, skip_magic=False)
        try:
            for record in reader.records:
                if isinstance(record, Schema):
                    schemas[record.id] = record
                elif isinstance(record, Channel):
                    channels[record.id] = record
                elif isinstance(record, Message):
                    ch = channels.get(record.channel_id)
                    if ch is None or ch.topic != topic:
                        continue
                    schema = schemas.get(ch.schema_id)
                    try:
                        msg = decoder.decode(schema, record)
                        bgr = image_msg_to_bgr(msg)
                    except Exception:
                        continue

                    stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

                    if writer is None:
                        size = (bgr.shape[1], bgr.shape[0])
                        # placeholder fps; corrected below once we know the real rate
                        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps or 15.0, size)

                    if show_frame_number:
                        cv2.putText(bgr, f"frame {n_written}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    writer.write(bgr)
                    n_written += 1
        except EndOfFile:
            pass

    if writer is not None:
        writer.release()

    if n_written == 0:
        raise ValueError(f"No messages found on topic '{topic}'. Use --list-topics to see what's available.")

    actual_fps = None
    if len(stamps) > 1:
        duration = stamps[-1] - stamps[0]
        actual_fps = (len(stamps) - 1) / duration if duration > 0 else None

    if fps is None and actual_fps and abs(actual_fps - 15.0) > 0.5:
        # re-encode once at the correct fps (VideoWriter can't change fps mid-stream)
        bag_to_video(mcap_path, topic, out_path, fps=actual_fps, show_frame_number=show_frame_number)
        return

    print(f"wrote {n_written} frames to {out_path}" + (f" ({actual_fps:.2f} fps)" if actual_fps else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", type=Path, help="Path to a bag folder or a direct .mcap file")
    parser.add_argument("--topic", default="/ogopogo/front/color/image_raw", help="Image topic to convert")
    parser.add_argument("--out", type=Path, default=None, help="Output .mp4 path (default: <bag_name>_<topic>.mp4 next to the bag)")
    parser.add_argument("--list-topics", action="store_true", help="List topics/message counts and exit")
    parser.add_argument("--no-frame-number", action="store_true", help="Don't overlay the frame index on each frame")
    args = parser.parse_args()

    mcap_path = resolve_mcap_path(args.bag)

    if args.list_topics:
        list_topics(mcap_path)
        return

    out_path = args.out or mcap_path.with_name(f"{mcap_path.stem}_{args.topic.strip('/').replace('/', '_')}.mp4")
    bag_to_video(mcap_path, args.topic, out_path, fps=None, show_frame_number=not args.no_frame_number)


if __name__ == "__main__":
    main()
