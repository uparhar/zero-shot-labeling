from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class GateSegWorkflowConfig:
    """Config for the gate segmentation bootstrap workflow (DINO box -> SAM2 mask -> YOLO-seg)."""

    # raw, not-yet-labeled footage to bootstrap
    image_dir: Path = REPO_ROOT / "dataset" / "gate_dataset" / "raw_unlabeled" / "images"

    # pre-existing box-labeled data (ODST + okmr_auv/ml/gate_ml), promoted to seg once
    seed_image_dir: Path = REPO_ROOT / "dataset" / "gate_dataset" / "seed" / "images"
    seed_label_dir: Path = REPO_ROOT / "dataset" / "gate_dataset" / "seed" / "labels"
    seed_seg_output_dir: Path = REPO_ROOT / "dataset" / "gate_dataset" / "seed_seg_out"

    dino_output_dir: Path = REPO_ROOT / "dataset" / "gate_dataset" / "dino_out"
    sam2_output_dir: Path = REPO_ROOT / "dataset" / "gate_dataset" / "sam2_out"
    combined_seg_dir: Path = REPO_ROOT / "dataset" / "gate_dataset" / "combined_seg"
    final_seg_dir: Path = REPO_ROOT / "dataset" / "gate_dataset" / "final_seg"

    yolo_dataset_dir: Path = REPO_ROOT / "dataset" / "gate_yolo_data_set"
    yolo_project_dir: Path = REPO_ROOT / "yolo_train_out"
    yolo_prediction_dir: Path = REPO_ROOT / "dataset" / "gate_yolo_predictions"
    merged_label_dir: Path = REPO_ROOT / "dataset" / "gate_autolabels" / "merged_labels"

    text_prompt: str = "underwater gate"
    box_threshold: float = 0.25
    text_threshold: float = 0.20
    confidence_threshold: float = 0.0
    nms_iou_threshold: float = 0.5
    merge_iou_threshold: float = 0.5

    train_run_name: str = "yolo_gate_seg_bootstrap"
    prediction_run_name: str = "yolo_gate_seg_bootstrap_pred"
    class_names: tuple[str, ...] = ("gate",)
    train_ratio: float = 0.8
    random_seed: int = 42
    train_epochs: int = 100
    train_batch_size: int = 32
    train_imgsz: int = 640
    train_workers: int = 8
    train_device: str | None = None
    inference_conf: float = 0.25
    inference_iou: float = 0.30

    grounding_dino_config: Path = REPO_ROOT / "Grounded-SAM-2" / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    grounding_dino_checkpoint: Path = REPO_ROOT / "Grounded-SAM-2" / "gdino_checkpoints" / "groundingdino_swint_ogc.pth"
    sam2_checkpoint: Path = REPO_ROOT / "Grounded-SAM-2" / "checkpoints" / "sam2.1_hiera_large.pt"
    sam2_model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device: str | None = None
