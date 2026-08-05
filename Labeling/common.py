from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import random
import shutil


def iter_image_paths(image_dir: Path, suffixes: Sequence[str] = (".jpg", ".jpeg", ".png")) -> list[Path]:
    """Return sorted image file paths in ``image_dir`` matching ``suffixes``.

    Args:
        image_dir: Directory containing image files.
        suffixes: Sequence of file suffixes to include (case-insensitive).

    Returns:
        Sorted list of Path objects for matching image files.
    """
    images = [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes]
    return sorted(images)


def ensure_directories(*paths: Path) -> None:
    """Create directories for each provided path if they do not exist.

    This is a convenience wrapper around ``Path.mkdir`` with ``parents=True``
    and ``exist_ok=True``.
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def read_yolo_labels(label_path: Path) -> list[dict[str, float | int | None]]:
    """Read a YOLO-format label file and return a list of box dictionaries.

    Each returned dict has keys: ``class``, ``x``, ``y``, ``w``, ``h``, and
    optional ``conf`` for confidence if present in the file.
    """
    boxes: list[dict[str, float | int | None]] = []
    if not label_path.exists():
        return boxes

    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            values = line.strip().split()
            if len(values) < 5:
                continue

            class_id, x, y, w, h = map(float, values[:5])
            confidence = float(values[5]) if len(values) > 5 else None
            boxes.append(
                {
                    "class": int(class_id),
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "conf": confidence,
                }
            )

    return boxes


def write_yolo_labels(boxes: Sequence[dict[str, float | int | None]], label_path: Path) -> None:
    """Write a sequence of YOLO-format boxes to ``label_path``.

    Args:
        boxes: Sequence of dicts with keys `class`, `x`, `y`, `w`, `h`.
        label_path: Destination file path for the labels.
    """
    ensure_directories(label_path.parent)
    with label_path.open("w", encoding="utf-8") as handle:
        for box in boxes:
            handle.write(f"{box['class']} {box['x']} {box['y']} {box['w']} {box['h']}\n")


def yolo_to_xyxy(box: dict[str, float | int | None], img_w: float, img_h: float) -> list[float]:
    """Convert a normalized YOLO box dict to absolute xyxy coordinates.

    Args:
        box: YOLO-format box dictionary with normalized ``x,y,w,h``.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        List `[x1, y1, x2, y2]` in pixel coordinates.
    """
    x = float(box["x"])
    y = float(box["y"])
    w = float(box["w"])
    h = float(box["h"])

    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return [x1, y1, x2, y2]


def iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Compute Intersection-over-Union (IoU) for two xyxy boxes.

    Args:
        box1: Sequence `[x1, y1, x2, y2]`.
        box2: Sequence `[x1, y1, x2, y2]`.

    Returns:
        IoU as a float in [0, 1]. Returns 0.0 if union is zero.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def create_yolo_yaml(
    yolo_data_dir: Path,
    output_path: Path,
    class_names: Sequence[str],
    task: str | None = None,
) -> None:
    """Write a minimal YOLOv8 `data.yaml` file for ``yolo_data_dir``.

    Args:
        yolo_data_dir: Directory containing `train/` and `valid/` subfolders.
        output_path: File path to write the YAML to.
        class_names: Iterable of class names in order.
        task: Optional Ultralytics task name (e.g. ``"segment"``) written as a
            top-level ``task:`` key. Omitted by default for detection datasets.
    """
    ensure_directories(output_path.parent)
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    task_block = f"task: {task}\n\n" if task else ""
    yaml_content = f'''path: "{yolo_data_dir.resolve().as_posix()}"\n\n{task_block}train: train/images\nval: valid/images\n\nnames:\n{names_block}\n'''

    output_path.write_text(yaml_content, encoding="utf-8")


def split_image_paths(image_paths: Sequence[Path], train_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Shuffle and split image paths into train/validation lists.

    Args:
        image_paths: Sequence of image Paths.
        train_ratio: Fraction of images to use for training.
        seed: RNG seed for reproducible shuffling.

    Returns:
        Tuple `(train_list, val_list)` of Path lists.
    """
    shuffled = list(image_paths)
    random.Random(seed).shuffle(shuffled)
    split_index = int(len(shuffled) * train_ratio)
    return shuffled[:split_index], shuffled[split_index:]


def copy_split(
    paths: Iterable[Path],
    label_dir: Path,
    images_target: Path,
    labels_target: Path,
) -> None:
    """Copy images and corresponding YOLO label files into a target split.

    Args:
        paths: Iterable of image Paths to copy.
        label_dir: Directory where original label `.txt` files live.
        images_target: Destination directory for images.
        labels_target: Destination directory for label files.
    """
    for image_path in paths:
        shutil.copy2(image_path, images_target / image_path.name)
        shutil.copy2(label_dir / f"{image_path.stem}.txt", labels_target / f"{image_path.stem}.txt")


def merge_dataset_sources(
    sources: Sequence[tuple[Path, Path]],
    dest_images: Path,
    dest_labels: Path,
) -> None:
    """Copy images+labels from multiple ``(image_dir, label_dir)`` sources into one destination.

    Later sources are skipped for any image stem already copied from an
    earlier source, so the first source in ``sources`` wins on duplicate
    filenames. Images without a matching label file are skipped.

    Args:
        sources: Ordered sequence of ``(image_dir, label_dir)`` pairs.
        dest_images: Destination directory for images.
        dest_labels: Destination directory for label files.
    """
    ensure_directories(dest_images, dest_labels)
    seen_stems: set[str] = set()

    for image_dir, label_dir in sources:
        for image_path in iter_image_paths(image_dir):
            if image_path.stem in seen_stems:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            shutil.copy2(image_path, dest_images / image_path.name)
            shutil.copy2(label_path, dest_labels / label_path.name)
            seen_stems.add(image_path.stem)


def read_yolo_seg_labels(label_path: Path) -> list[dict[str, int | list[float]]]:
    """Read a YOLO-seg label file and return a list of polygon dictionaries.

    Each line is ``class x1 y1 x2 y2 ... xn yn`` with normalized polygon
    coordinates. Each returned dict has keys ``class`` and ``polygon`` (a flat
    list of normalized ``x, y`` values).
    """
    boxes: list[dict[str, int | list[float]]] = []
    if not label_path.exists():
        return boxes

    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            values = line.strip().split()
            if len(values) < 7 or len(values) % 2 == 0:
                continue

            class_id = int(float(values[0]))
            polygon = [float(value) for value in values[1:]]
            boxes.append({"class": class_id, "polygon": polygon})

    return boxes


def write_yolo_seg_labels(boxes: Sequence[dict[str, int | list[float]]], label_path: Path) -> None:
    """Write a sequence of YOLO-seg polygon boxes to ``label_path``.

    Args:
        boxes: Sequence of dicts with keys ``class`` and ``polygon`` (flat
            list of normalized ``x, y`` values).
        label_path: Destination file path for the labels.
    """
    ensure_directories(label_path.parent)
    with label_path.open("w", encoding="utf-8") as handle:
        for box in boxes:
            coords = " ".join(f"{value:.6f}" for value in box["polygon"])
            handle.write(f"{box['class']} {coords}\n")


def mask_to_yolo_polygon(mask, epsilon_ratio: float = 0.005) -> list[float] | None:
    """Convert a binary mask into a single normalized YOLO-seg polygon.

    Finds the largest external contour, simplifies it with ``approxPolyDP``,
    and normalizes point coordinates to ``[0, 1]`` by the mask's own
    height/width.

    Args:
        mask: 2D boolean or 0/1 array.
        epsilon_ratio: Contour simplification tolerance as a fraction of the
            contour perimeter.

    Returns:
        Flat list ``[x1, y1, x2, y2, ...]`` in normalized coordinates, or
        ``None`` if the mask has no usable contour.
    """
    import cv2
    import numpy as np

    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0:
        return None

    perimeter = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon_ratio * perimeter, True)
    if len(approx) < 3:
        approx = largest

    height, width = mask.shape[:2]
    polygon: list[float] = []
    for point in approx.reshape(-1, 2):
        polygon.append(float(point[0]) / width)
        polygon.append(float(point[1]) / height)

    return polygon


def yolo_polygon_to_mask(polygon: Sequence[float], img_w: int, img_h: int):
    """Rasterize a normalized YOLO-seg polygon into a boolean mask.

    Args:
        polygon: Flat list of normalized ``x, y`` values.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        Boolean array of shape ``(img_h, img_w)``.
    """
    import cv2
    import numpy as np

    points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    points[:, 0] *= img_w
    points[:, 1] *= img_h

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [points.astype(np.int32)], 1)
    return mask.astype(bool)


def mask_iou(mask1, mask2) -> float:
    """Compute Intersection-over-Union between two boolean masks of the same shape.

    Returns:
        IoU as a float in ``[0, 1]``. Returns ``0.0`` if the union is empty.
    """
    import numpy as np

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0
