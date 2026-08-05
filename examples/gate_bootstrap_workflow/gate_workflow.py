"""Run the underwater-gate segmentation bootstrap workflow step by step.

Pipeline: DINO boxes -> SAM2 masks -> YOLO-seg training -> re-infer -> merge
-> retrain -> export ONNX. Existing box-labeled seed data (ODST +
okmr_auv/ml/gate_ml, consolidated by consolidate_seed_data.py) is promoted to
segmentation once and combined with auto-labeled footage so everything trains
in one consistent YOLO-seg format. The exported ONNX model is meant to be
dropped into okmr_auv's ros2_ws/src/okmr_object_detection/models/gate.onnx.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Labeling.common import merge_dataset_sources
from Labeling.config import GateSegWorkflowConfig
from Labeling.grounding_dino import run_grounding_dino_directory
from Labeling.merge import merge_seg_label_sets
from Labeling.segmentation import run_sam2_on_directory
from Labeling.yolo import export_to_onnx, prepare_yolo_seg_dataset, run_yolo_seg_inference, train_yolo_seg


WORKFLOW_CONFIG = GateSegWorkflowConfig()


def promote_seed_step(config: GateSegWorkflowConfig) -> Path:
    """Promote the existing box-labeled seed dataset to YOLO-seg using SAM2."""
    return run_sam2_on_directory(
        image_dir=config.seed_image_dir,
        box_label_dir=config.seed_label_dir,
        output_dir=config.seed_seg_output_dir,
        checkpoint=config.sam2_checkpoint,
        model_config=config.sam2_model_config,
        device=config.device,
    )


def run_grounding_dino_step(config: GateSegWorkflowConfig) -> Path:
    """Run Grounding DINO on the raw unlabeled footage and return the output directory."""
    return run_grounding_dino_directory(
        image_dir=config.image_dir,
        output_dir=config.dino_output_dir,
        text_prompt=config.text_prompt,
        box_threshold=config.box_threshold,
        text_threshold=config.text_threshold,
        confidence_threshold=config.confidence_threshold,
        nms_iou_threshold=config.nms_iou_threshold,
        device=config.device,
        grounding_dino_config=config.grounding_dino_config,
        grounding_dino_checkpoint=config.grounding_dino_checkpoint,
    )


def promote_dino_step(config: GateSegWorkflowConfig, dino_output_dir: Path) -> Path:
    """Promote the DINO box labels on the raw footage to YOLO-seg using SAM2."""
    return run_sam2_on_directory(
        image_dir=config.image_dir,
        box_label_dir=dino_output_dir / "labels",
        output_dir=config.sam2_output_dir,
        checkpoint=config.sam2_checkpoint,
        model_config=config.sam2_model_config,
        device=config.device,
    )


def prepare_bootstrap_dataset_step(config: GateSegWorkflowConfig, dino_seg_dir: Path) -> Path:
    """Combine seed + DINO/SAM2 seg labels and build the bootstrap YOLO-seg dataset."""
    combined_images = config.combined_seg_dir / "images"
    combined_labels = config.combined_seg_dir / "labels"
    merge_dataset_sources(
        sources=[
            (config.seed_image_dir, config.seed_seg_output_dir / "labels"),
            (config.image_dir, dino_seg_dir / "labels"),
        ],
        dest_images=combined_images,
        dest_labels=combined_labels,
    )
    return prepare_yolo_seg_dataset(
        image_dir=combined_images,
        label_dir=combined_labels,
        yolo_data_dir=config.yolo_dataset_dir,
        class_names=config.class_names,
        train_ratio=config.train_ratio,
        seed=config.random_seed,
    )


def train_yolo_step(config: GateSegWorkflowConfig, data_yaml: Path, run_name: str) -> Path:
    """Train the YOLO-seg model and return the best weights path."""
    return train_yolo_seg(
        data_yaml=data_yaml,
        project_dir=config.yolo_project_dir,
        run_name=run_name,
        epochs=config.train_epochs,
        imgsz=config.train_imgsz,
        batch_size=config.train_batch_size,
        resume=True,
        workers=config.train_workers,
        device=config.train_device or config.device,
    )


def run_yolo_inference_step(config: GateSegWorkflowConfig, best_weights: Path) -> Path:
    """Run YOLO-seg inference on the raw footage and return the prediction directory."""
    return run_yolo_seg_inference(
        weights_path=best_weights,
        source_dir=config.image_dir,
        project_dir=config.yolo_prediction_dir,
        run_name=config.prediction_run_name,
        conf=config.inference_conf,
        iou=config.inference_iou,
        device=config.device,
    )


def merge_labels_step(config: GateSegWorkflowConfig, dino_seg_dir: Path, prediction_dir: Path) -> Path:
    """Merge YOLO-seg and DINO/SAM2-seg labels for the raw footage only.

    The seed set is already hand-labeled ground truth and is left untouched;
    only the auto-labeled footage benefits from this refinement pass.
    """
    return merge_seg_label_sets(
        image_dir=config.image_dir,
        yolo_label_dir=prediction_dir / "labels",
        dino_label_dir=dino_seg_dir / "labels",
        merged_label_dir=config.merged_label_dir,
        iou_threshold=config.merge_iou_threshold,
    )


def prepare_final_dataset_step(config: GateSegWorkflowConfig, merged_dir: Path) -> Path:
    """Combine seed + merged (refined) labels and build the final YOLO-seg dataset."""
    final_images = config.final_seg_dir / "images"
    final_labels = config.final_seg_dir / "labels"
    merge_dataset_sources(
        sources=[
            (config.seed_image_dir, config.seed_seg_output_dir / "labels"),
            (config.image_dir, merged_dir),
        ],
        dest_images=final_images,
        dest_labels=final_labels,
    )
    return prepare_yolo_seg_dataset(
        image_dir=final_images,
        label_dir=final_labels,
        yolo_data_dir=config.yolo_dataset_dir.parent / f"{config.yolo_dataset_dir.name}_final",
        class_names=config.class_names,
        train_ratio=config.train_ratio,
        seed=config.random_seed,
    )


def main() -> None:
    """Execute the gate segmentation bootstrap workflow one stage at a time."""
    config = WORKFLOW_CONFIG

    seed_seg_dir = promote_seed_step(config)
    print(f"Seed labels promoted to seg: {seed_seg_dir}")

    dino_output_dir = run_grounding_dino_step(config)
    print(f"Grounding DINO outputs: {dino_output_dir}")

    dino_seg_dir = promote_dino_step(config, dino_output_dir)
    print(f"DINO labels promoted to seg: {dino_seg_dir}")

    bootstrap_data_yaml = prepare_bootstrap_dataset_step(config, dino_seg_dir)
    print(f"Bootstrap YOLO-seg dataset YAML: {bootstrap_data_yaml}")

    bootstrap_weights = train_yolo_step(config, bootstrap_data_yaml, config.train_run_name)
    print(f"Bootstrap YOLO-seg weights: {bootstrap_weights}")

    prediction_dir = run_yolo_inference_step(config, bootstrap_weights)
    print(f"YOLO-seg predictions: {prediction_dir}")

    merged_dir = merge_labels_step(config, dino_seg_dir, prediction_dir)
    print(f"Merged labels: {merged_dir}")

    final_data_yaml = prepare_final_dataset_step(config, merged_dir)
    print(f"Final YOLO-seg dataset YAML: {final_data_yaml}")

    final_weights = train_yolo_step(config, final_data_yaml, f"{config.train_run_name}_final")
    print(f"Final YOLO-seg weights: {final_weights}")

    onnx_path = export_to_onnx(final_weights, imgsz=config.train_imgsz)
    print(f"Exported ONNX model: {onnx_path}")
    print("Copy this file to okmr_auv/ros2_ws/src/okmr_object_detection/models/gate.onnx "
          "and verify its output shape with onnxruntime before deploying.")


if __name__ == "__main__":
    main()
