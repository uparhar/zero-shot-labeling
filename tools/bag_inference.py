"""Run the trained gate model against a bag's real RGB(+depth) footage, to
check real-world viability without needing ROS2. Produces an annotated
video (green = accepted, red = rejected by depth confirmation) and prints
summary detection stats.

The depth-confirmation logic here mirrors _depth_confirms_detection in
okmr_auv's onnx_segmentation_detector.py, reimplemented standalone so it
can be tested against real bag data outside of ROS2.

Usage:
    python tools/bag_inference.py <bag_dir_or_mcap_path> [--weights path/to/best.pt] [--conf 0.25]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from mcap.stream_reader import StreamReader
from mcap.records import Message, Channel, Schema
from mcap.exceptions import EndOfFile
from mcap_ros2.decoder import Decoder

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "yolo_train_out" / "yolo_gate_seg_real_100ep" / "weights" / "best.pt"


def resolve_mcap_path(bag_path: Path) -> Path:
    if bag_path.is_file():
        return bag_path
    mcap_files = list(bag_path.glob("*.mcap"))
    if not mcap_files:
        raise FileNotFoundError(f"No .mcap file found in {bag_path}")
    return mcap_files[0]


def image_msg_to_array(msg):
    h, w = msg.height, msg.width
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if msg.encoding == "rgb8":
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if msg.encoding == "bgr8":
        return data.reshape(h, w, 3)
    if msg.encoding == "16UC1":
        return data.view(np.uint16).reshape(h, w)
    raise ValueError(f"unhandled encoding: {msg.encoding}")


def depth_confirms(bin_mask: np.ndarray, depth: np.ndarray | None, valid_min_ratio=0.4, consistency_max_cv=0.5):
    """Same logic as onnx_segmentation_detector.py's _depth_confirms_detection."""
    if depth is None:
        return True, None, None

    eroded = cv2.erode(bin_mask.astype(np.uint8), np.ones((5, 5), np.uint8))
    sample_mask = eroded if eroded.sum() >= 10 else bin_mask.astype(np.uint8)

    region_depth = depth[sample_mask.astype(bool)]
    if region_depth.size == 0:
        return False, 0.0, None

    valid = region_depth[region_depth > 0]
    valid_ratio = valid.size / region_depth.size
    if valid_ratio < valid_min_ratio:
        return False, valid_ratio, None

    mean_depth = float(valid.mean())
    if mean_depth <= 0:
        return False, valid_ratio, None

    cv_val = float(valid.std()) / mean_depth
    return cv_val <= consistency_max_cv, valid_ratio, cv_val


def extract_pairs(mcap_path: Path, rgb_topic: str, depth_topic: str):
    channels, schemas = {}, {}
    decoder = Decoder()
    rgb_msgs, depth_msgs = [], []

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
                    if ch is None or ch.topic not in (rgb_topic, depth_topic):
                        continue
                    schema = schemas.get(ch.schema_id)
                    try:
                        msg = decoder.decode(schema, record)
                    except Exception:
                        continue
                    (rgb_msgs if ch.topic == rgb_topic else depth_msgs).append(msg)
        except EndOfFile:
            pass

    def stamp_key(m):
        return (m.header.stamp.sec, m.header.stamp.nanosec)

    rgb_msgs.sort(key=stamp_key)
    depth_msgs.sort(key=stamp_key)
    return rgb_msgs, depth_msgs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", type=Path, help="Path to a bag folder or a direct .mcap file")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to trained YOLO-seg weights")
    parser.add_argument("--rgb-topic", default="/ogopogo/front/color/image_raw")
    parser.add_argument("--depth-topic", default="/ogopogo/front/depth/image_rect_raw")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--out", type=Path, default=None, help="Output annotated .mp4 path")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed, for a quick test")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = parser.parse_args()

    mcap_path = resolve_mcap_path(args.bag)
    out_path = args.out or mcap_path.with_name(f"{mcap_path.stem}_inference.mp4")

    print(f"loading model: {args.weights}")
    model = YOLO(str(args.weights))

    print("extracting frames from bag (large bags take a while)...")
    rgb_msgs, depth_msgs = extract_pairs(mcap_path, args.rgb_topic, args.depth_topic)
    print(f"found {len(rgb_msgs)} rgb messages, {len(depth_msgs)} depth messages")

    if not rgb_msgs:
        raise ValueError(f"No messages on '{args.rgb_topic}'. Use tools/bag_to_video.py --list-topics to check topic names.")

    if args.max_frames:
        rgb_msgs = rgb_msgs[: args.max_frames]

    writer = None
    n_frames_with_detection = 0
    n_detections = 0
    n_depth_confirmed = 0
    n_depth_rejected = 0
    confidences = []

    for i, rgb_msg in enumerate(rgb_msgs):
        rgb = image_msg_to_array(rgb_msg)
        depth = image_msg_to_array(depth_msgs[i]) if i < len(depth_msgs) else None

        result = model.predict(rgb, conf=args.conf, verbose=False, device=args.device)[0]
        vis = rgb.copy()

        if result.masks is not None and len(result.masks) > 0:
            n_frames_with_detection += 1
            for box, mask_t, conf_t in zip(result.boxes, result.masks, result.boxes.conf):
                n_detections += 1
                conf = float(conf_t)
                confidences.append(conf)

                mask_resized = cv2.resize(mask_t.data[0].cpu().numpy(), (rgb.shape[1], rgb.shape[0]))
                bin_mask = mask_resized > 0.5
                ok, valid_ratio, cv_val = depth_confirms(bin_mask, depth)
                if depth is not None:
                    if ok:
                        n_depth_confirmed += 1
                    else:
                        n_depth_rejected += 1

                color = (0, 255, 0) if ok else (0, 0, 255)
                overlay = vis.copy()
                overlay[bin_mask] = color
                vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"gate {conf:.2f}"
                if depth is not None:
                    label += " ok" if ok else " REJECTED(depth)"
                cv2.putText(vis, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        cv2.putText(vis, f"frame {i}/{len(rgb_msgs)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (vis.shape[1], vis.shape[0]))
        writer.write(vis)

    if writer is not None:
        writer.release()

    print()
    print(f"processed {len(rgb_msgs)} frames")
    print(f"frames with >=1 detection: {n_frames_with_detection} ({100 * n_frames_with_detection / len(rgb_msgs):.1f}%)")
    print(f"total detections: {n_detections}")
    if confidences:
        print(f"confidence: mean={np.mean(confidences):.3f} min={min(confidences):.3f} max={max(confidences):.3f}")
    if depth_msgs:
        print(f"depth-confirmed: {n_depth_confirmed}")
        print(f"depth-rejected (likely reflection/noise): {n_depth_rejected}")
    else:
        print("(no depth topic found - depth confirmation skipped)")
    print(f"\nsaved annotated video to {out_path}")


if __name__ == "__main__":
    main()
