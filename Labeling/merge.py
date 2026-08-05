from __future__ import annotations

from pathlib import Path

from Labeling.common import (
    ensure_directories,
    iou,
    iter_image_paths,
    mask_iou,
    read_yolo_labels,
    read_yolo_seg_labels,
    write_yolo_labels,
    write_yolo_seg_labels,
    yolo_polygon_to_mask,
    yolo_to_xyxy,
)


def merge_label_sets(
    image_dir: Path,
    yolo_label_dir: Path,
    dino_label_dir: Path,
    merged_label_dir: Path,
    iou_threshold: float = 0.5,
    ) -> Path:
    """Merge YOLO and DINO label files for all images in ``image_dir``.

    For each image, the function prefers YOLO boxes and adds DINO boxes that
    do not overlap sufficiently with existing YOLO boxes, based on ``iou_threshold``.

    Args:
        image_dir: Directory with source images.
        yolo_label_dir: Directory with YOLO prediction labels.
        dino_label_dir: Directory with DINO prediction labels.
        merged_label_dir: Output directory to write merged label files.
        iou_threshold: Minimum IoU to consider two boxes the same.

    Returns:
        The ``merged_label_dir`` Path.
    """
    ensure_directories(merged_label_dir)

    for image_path in iter_image_paths(image_dir):
        yolo_boxes = read_yolo_labels(yolo_label_dir / f"{image_path.stem}.txt")
        dino_boxes = read_yolo_labels(dino_label_dir / f"{image_path.stem}.txt")

        yolo_xy = [yolo_to_xyxy(box, 1, 1) for box in yolo_boxes]
        dino_xy = [yolo_to_xyxy(box, 1, 1) for box in dino_boxes]

        matched_dino: set[int] = set()
        merged: list[dict[str, float | int | None]] = []

        for yolo_index, yolo_box in enumerate(yolo_xy):
            best_iou = 0.0
            best_index = -1

            for dino_index, dino_box in enumerate(dino_xy):
                if dino_index in matched_dino:
                    continue

                overlap = iou(yolo_box, dino_box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = dino_index

            merged.append(yolo_boxes[yolo_index])
            if best_iou > iou_threshold and best_index >= 0:
                matched_dino.add(best_index)

        for dino_index, dino_box in enumerate(dino_boxes):
            if dino_index not in matched_dino:
                merged.append(dino_box)

        write_yolo_labels(merged, merged_label_dir / f"{image_path.stem}.txt")

    return merged_label_dir


def merge_seg_label_sets(
    image_dir: Path,
    yolo_label_dir: Path,
    dino_label_dir: Path,
    merged_label_dir: Path,
    iou_threshold: float = 0.5,
    ) -> Path:
    """Merge YOLO-seg and DINO-seg polygon label files for all images in ``image_dir``.

    Same fusion strategy as `merge_label_sets` (prefer YOLO polygons, add any
    DINO polygon that does not sufficiently overlap an existing YOLO
    polygon), but overlap is computed on rasterized masks since both label
    sets are polygons rather than boxes.

    Args:
        image_dir: Directory with source images (used for pixel dimensions).
        yolo_label_dir: Directory with YOLO-seg prediction labels.
        dino_label_dir: Directory with DINO-seg (SAM2-promoted) labels.
        merged_label_dir: Output directory to write merged label files.
        iou_threshold: Minimum mask IoU to consider two polygons the same.

    Returns:
        The ``merged_label_dir`` Path.
    """
    import cv2

    ensure_directories(merged_label_dir)

    for image_path in iter_image_paths(image_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]

        yolo_boxes = read_yolo_seg_labels(yolo_label_dir / f"{image_path.stem}.txt")
        dino_boxes = read_yolo_seg_labels(dino_label_dir / f"{image_path.stem}.txt")

        yolo_masks = [yolo_polygon_to_mask(box["polygon"], width, height) for box in yolo_boxes]
        dino_masks = [yolo_polygon_to_mask(box["polygon"], width, height) for box in dino_boxes]

        matched_dino: set[int] = set()
        merged: list[dict[str, int | list[float]]] = []

        for yolo_index, yolo_mask in enumerate(yolo_masks):
            best_iou = 0.0
            best_index = -1

            for dino_index, dino_mask in enumerate(dino_masks):
                if dino_index in matched_dino:
                    continue

                overlap = mask_iou(yolo_mask, dino_mask)
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = dino_index

            merged.append(yolo_boxes[yolo_index])
            if best_iou > iou_threshold and best_index >= 0:
                matched_dino.add(best_index)

        for dino_index, dino_box in enumerate(dino_boxes):
            if dino_index not in matched_dino:
                merged.append(dino_box)

        write_yolo_seg_labels(merged, merged_label_dir / f"{image_path.stem}.txt")

    return merged_label_dir
