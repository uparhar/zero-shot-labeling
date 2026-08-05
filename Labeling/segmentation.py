from __future__ import annotations

from pathlib import Path
import sys

from Labeling.common import (
    ensure_directories,
    iter_image_paths,
    mask_to_yolo_polygon,
    read_yolo_labels,
    write_yolo_seg_labels,
    yolo_to_xyxy,
)
from Labeling.config import REPO_ROOT


GROUNDED_SAM2_ROOT = REPO_ROOT / "Grounded-SAM-2"


def load_sam2_predictor(checkpoint: Path, model_config: str, device: str | None = None):
    """Build and return a SAM2ImagePredictor from the Grounded-SAM-2 submodule.

    Args:
        checkpoint: Path to a SAM2 checkpoint (e.g. ``sam2.1_hiera_large.pt``).
        model_config: SAM2 model config name relative to the submodule
            (e.g. ``"configs/sam2.1/sam2.1_hiera_l.yaml"``).
        device: Optional device string (e.g. 'cpu', 'cuda', 'mps').

    Returns:
        A loaded ``SAM2ImagePredictor`` ready for ``set_image``/``predict``.
    """
    import torch

    if str(GROUNDED_SAM2_ROOT) not in sys.path:
        sys.path.insert(0, str(GROUNDED_SAM2_ROOT))

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    resolved_device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    sam2_model = build_sam2(model_config, str(checkpoint), device=resolved_device)
    return SAM2ImagePredictor(sam2_model)


def boxes_to_masks(predictor, image_rgb, boxes_xyxy: list[list[float]]) -> list:
    """Promote a list of xyxy boxes into binary segmentation masks using SAM2.

    Args:
        predictor: A loaded ``SAM2ImagePredictor``.
        image_rgb: RGB image array the boxes were detected on.
        boxes_xyxy: List of ``[x1, y1, x2, y2]`` pixel boxes.

    Returns:
        List of boolean masks, one per input box, in the same order.
    """
    import numpy as np

    if not boxes_xyxy:
        return []

    predictor.set_image(image_rgb)

    masks = []
    for box in boxes_xyxy:
        box_array = np.array(box, dtype=np.float32)
        box_masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_array[None, :],
            multimask_output=False,
        )
        masks.append(box_masks[0].astype(bool))

    return masks


def run_sam2_on_directory(
    image_dir: Path,
    box_label_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    model_config: str,
    device: str | None = None,
) -> Path:
    """Promote existing YOLO box labels to YOLO-seg polygon labels using SAM2.

    Reads YOLO box labels from ``box_label_dir`` for every image in
    ``image_dir``, runs SAM2 to get a mask per box, converts each mask to a
    normalized polygon, and writes YOLO-seg label files (plus mask-overlay
    visualizations) to ``output_dir``.

    Args:
        image_dir: Directory containing source images.
        box_label_dir: Directory with existing YOLO box `.txt` labels.
        output_dir: Directory to write seg labels and visualizations.
        checkpoint: Path to a SAM2 checkpoint.
        model_config: SAM2 model config name.
        device: Optional device string.

    Returns:
        The ``output_dir`` path passed in.
    """
    import cv2
    import numpy as np

    labels_dir = output_dir / "labels"
    viz_dir = output_dir / "viz"
    ensure_directories(output_dir, labels_dir, viz_dir)

    predictor = load_sam2_predictor(checkpoint, model_config, device)

    for image_path in iter_image_paths(image_dir):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_bgr.shape[:2]

        box_labels = read_yolo_labels(box_label_dir / f"{image_path.stem}.txt")
        boxes_xyxy = [yolo_to_xyxy(box, width, height) for box in box_labels]

        masks = boxes_to_masks(predictor, image_rgb, boxes_xyxy)

        seg_labels: list[dict[str, int | list[float]]] = []
        overlay = image_bgr.copy()
        for box, mask in zip(box_labels, masks):
            polygon = mask_to_yolo_polygon(mask)
            if polygon is None:
                continue
            seg_labels.append({"class": box["class"], "polygon": polygon})
            overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 255, 0])).astype(overlay.dtype)

        write_yolo_seg_labels(seg_labels, labels_dir / f"{image_path.stem}.txt")
        cv2.imwrite(str(viz_dir / image_path.name), overlay)

    return output_dir
