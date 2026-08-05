"""Scan every bag folder under a directory and build labeled contact-sheet
thumbnails, so you can spot which bag has the footage you want without
converting each one to a full video. No ROS2 install required.

Usage:
    python tools/scan_bags.py "C:\\Users\\uplov\\Resilio Sync\\okmr_bags" [--topic /ogopogo/front/color/image_raw] [--out bag_scan/]
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

THUMB_W, THUMB_H = 200, 150
FRAMES_PER_BAG = 3
BAGS_PER_SHEET = 12


def image_msg_to_bgr(msg):
    h, w = msg.height, msg.width
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if msg.encoding == "rgb8":
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if msg.encoding == "bgr8":
        return data.reshape(h, w, 3)
    if msg.encoding == "mono8":
        return cv2.cvtColor(data.reshape(h, w), cv2.COLOR_GRAY2BGR)
    return None


def sample_bag(mcap_path: Path, topic: str, max_bytes: int) -> tuple[list, bool]:
    """Decode frames from `topic`, reading at most `max_bytes` of the file.

    Large bags (multi-GB) would otherwise take minutes to fully decompress
    just to pull out a handful of thumbnails, so scanning is capped to a
    byte budget. Thumbnails are then picked evenly spaced from whatever was
    found within that budget.

    Returns (thumbnails, truncated) where `truncated` is True if the byte cap
    was hit before the file ended (thumbnails only cover the scanned prefix,
    not the whole bag).
    """
    channels, schemas = {}, {}
    decoder = Decoder()
    msgs = []
    truncated = False

    with mcap_path.open("rb") as f:
        reader = StreamReader(f, skip_magic=False)
        try:
            for record in reader.records:
                if f.tell() > max_bytes:
                    truncated = True
                    break
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
                        msgs.append(decoder.decode(schema, record))
                    except Exception:
                        continue
        except EndOfFile:
            pass

    if not msgs:
        return [], truncated

    n = len(msgs)
    indices = sorted(set(int(n * f) for f in np.linspace(0, 0.99, FRAMES_PER_BAG)))
    thumbs = []
    for idx in indices:
        bgr = image_msg_to_bgr(msgs[idx])
        if bgr is not None:
            thumbs.append(cv2.resize(bgr, (THUMB_W, THUMB_H)))
    return thumbs, truncated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bags_root", type=Path, help="Directory containing bag subfolders")
    parser.add_argument("--topic", default="/ogopogo/front/color/image_raw", help="Image topic to sample")
    parser.add_argument("--out", type=Path, default=Path("bag_scan"), help="Output directory for contact sheets")
    parser.add_argument("--max-mb", type=int, default=150, help="Max MB to read per bag (caps time spent on huge bags; raise for more complete sampling)")
    args = parser.parse_args()
    max_bytes = args.max_mb * 1024 * 1024

    args.out.mkdir(parents=True, exist_ok=True)
    bag_dirs = sorted(p for p in args.bags_root.iterdir() if p.is_dir())
    print(f"found {len(bag_dirs)} bag folders")

    rows, labels = [], []
    for bag_dir in bag_dirs:
        mcap_files = list(bag_dir.glob("*.mcap"))
        if not mcap_files or mcap_files[0].stat().st_size == 0:
            continue
        size_mb = mcap_files[0].stat().st_size / (1024 * 1024)
        print(f"  [scanning] {bag_dir.name} ({size_mb:.0f} MB)...", flush=True)
        try:
            thumbs, truncated = sample_bag(mcap_files[0], args.topic, max_bytes)
        except Exception as e:
            print(f"  [skip] {bag_dir.name}: {e}")
            continue
        if not thumbs:
            print(f"  [no matching topic in first {args.max_mb}MB] {bag_dir.name}")
            continue
        while len(thumbs) < FRAMES_PER_BAG:
            thumbs.append(np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8))
        rows.append(np.hstack(thumbs[:FRAMES_PER_BAG]))
        labels.append(bag_dir.name + (" (partial)" if truncated else ""))
        print(f"  [ok] {bag_dir.name}" + (" (only scanned first chunk, raise --max-mb for full coverage)" if truncated else ""))

    label_h = 20
    row_h = THUMB_H + label_h
    sheet_w = FRAMES_PER_BAG * THUMB_W

    for page_start in range(0, len(rows), BAGS_PER_SHEET):
        page_rows = rows[page_start:page_start + BAGS_PER_SHEET]
        page_labels = labels[page_start:page_start + BAGS_PER_SHEET]
        sheet = np.zeros((row_h * len(page_rows), sheet_w, 3), dtype=np.uint8)
        for i, (row, label) in enumerate(zip(page_rows, page_labels)):
            y0 = i * row_h
            cv2.putText(sheet, label, (2, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            sheet[y0 + label_h:y0 + row_h, 0:sheet_w] = row
        out_path = args.out / f"contact_sheet_{page_start // BAGS_PER_SHEET:02d}.png"
        cv2.imwrite(str(out_path), sheet)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
