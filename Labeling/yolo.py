from __future__ import annotations

from pathlib import Path

from Labeling.common import copy_split, create_yolo_yaml, ensure_directories, iter_image_paths, resolve_device, split_image_paths


def prepare_yolo_dataset(
    image_dir: Path,
    label_dir: Path,
    yolo_data_dir: Path,
    class_names: tuple[str, ...] = ("gate",),
    train_ratio: float = 0.8,
    seed: int = 42,
    ) -> Path:
    """Create YOLOv8 dataset structure and `data.yaml` from images and labels.

    This copies images and label files into `train/` and `valid/` subfolders and
    writes a `data.yaml` file suitable for Ultralytics YOLO training.

    Returns:
        Path to the generated `data.yaml` file.
    """
    images_train_dir = yolo_data_dir / "train" / "images"
    images_val_dir = yolo_data_dir / "valid" / "images"
    labels_train_dir = yolo_data_dir / "train" / "labels"
    labels_val_dir = yolo_data_dir / "valid" / "labels"

    ensure_directories(images_train_dir, images_val_dir, labels_train_dir, labels_val_dir)
    create_yolo_yaml(yolo_data_dir, yolo_data_dir / "data.yaml", class_names)

    train_images, val_images = split_image_paths(iter_image_paths(image_dir), train_ratio, seed)
    copy_split(train_images, label_dir, images_train_dir, labels_train_dir)
    copy_split(val_images, label_dir, images_val_dir, labels_val_dir)
    return yolo_data_dir / "data.yaml"


def train_yolo(
    data_yaml: Path,
    project_dir: Path,
    run_name: str,
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 32,
    resume: bool = True,
    workers: int = 8,
    device: str | None = None,
    ) -> Path:
    """Train a YOLOv8 model using the Ultralytics API and return best weights.

    Args:
        data_yaml: Path to YOLO `data.yaml`.
        project_dir: Directory where training outputs are saved.
        run_name: Name of the training run.
        epochs: Number of training epochs.
        imgsz: Training image size.
        batch_size: Batch size.
        resume: Whether to resume from last checkpoint if present.
        workers: Number of dataloader workers.
        device: Optional device string.

    Returns:
        Path to the `best.pt` weights file produced by training.
    """
    from ultralytics import YOLO

    weights_path = project_dir / run_name / "weights" / "last.pt"
    if weights_path.exists():
        model = YOLO(str(weights_path))
        resume_flag = True
    else:
        model = YOLO("yolov8n.pt")
        resume_flag = False

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=str(project_dir),
        name=run_name,
        device=resolve_device(device),
        exist_ok=True,
        resume=resume and resume_flag,
        workers=workers,
    )

    return project_dir / run_name / "weights" / "best.pt"


def run_yolo_inference(
    weights_path: Path,
    source_dir: Path,
    project_dir: Path,
    run_name: str,
    conf: float = 0.03,
    iou: float = 0.30,
    device: str | None = None,
    ) -> Path:
    """Run YOLO inference on a source directory and save predictions.

    Args:
        weights_path: Path to model weights to use for inference.
        source_dir: Directory containing images to run inference on.
        project_dir: Directory where inference outputs are saved.
        run_name: Name for the inference run folder.
        conf: Confidence threshold for predictions.
        iou: IoU threshold for NMS during prediction.
        device: Optional device string.

    Returns:
        Path to the inference run folder containing predictions and `labels/`.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    model.predict(
        source=str(source_dir),
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        conf=conf,
        iou=iou,
        device=resolve_device(device),
        save=True,
        save_txt=True,
        save_conf=True,
    )

    return project_dir / run_name


def prepare_yolo_seg_dataset(
    image_dir: Path,
    label_dir: Path,
    yolo_data_dir: Path,
    class_names: tuple[str, ...] = ("gate",),
    train_ratio: float = 0.8,
    seed: int = 42,
    ) -> Path:
    """Create YOLOv8-seg dataset structure and `data.yaml` from images and seg labels.

    Identical layout to `prepare_yolo_dataset`; label files are YOLO-seg
    polygon `.txt` files instead of box `.txt` files, which `copy_split`
    handles transparently since it copies whichever label file matches an
    image's stem regardless of its content format.

    Returns:
        Path to the generated `data.yaml` file.
    """
    images_train_dir = yolo_data_dir / "train" / "images"
    images_val_dir = yolo_data_dir / "valid" / "images"
    labels_train_dir = yolo_data_dir / "train" / "labels"
    labels_val_dir = yolo_data_dir / "valid" / "labels"

    ensure_directories(images_train_dir, images_val_dir, labels_train_dir, labels_val_dir)
    create_yolo_yaml(yolo_data_dir, yolo_data_dir / "data.yaml", class_names, task="segment")

    train_images, val_images = split_image_paths(iter_image_paths(image_dir), train_ratio, seed)
    copy_split(train_images, label_dir, images_train_dir, labels_train_dir)
    copy_split(val_images, label_dir, images_val_dir, labels_val_dir)
    return yolo_data_dir / "data.yaml"


def train_yolo_seg(
    data_yaml: Path,
    project_dir: Path,
    run_name: str,
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 32,
    resume: bool = True,
    workers: int = 8,
    device: str | None = None,
    ) -> Path:
    """Train a YOLOv8 segmentation model using the Ultralytics API and return best weights.

    Same pattern as `train_yolo`, starting from the `yolov8n-seg.pt`
    checkpoint instead of the detection checkpoint.

    Returns:
        Path to the `best.pt` weights file produced by training.
    """
    from ultralytics import YOLO

    weights_path = project_dir / run_name / "weights" / "last.pt"
    if weights_path.exists():
        model = YOLO(str(weights_path))
        resume_flag = True
    else:
        model = YOLO("yolov8n-seg.pt")
        resume_flag = False

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=str(project_dir),
        name=run_name,
        device=resolve_device(device),
        exist_ok=True,
        resume=resume and resume_flag,
        workers=workers,
    )

    return project_dir / run_name / "weights" / "best.pt"


def run_yolo_seg_inference(
    weights_path: Path,
    source_dir: Path,
    project_dir: Path,
    run_name: str,
    conf: float = 0.03,
    iou: float = 0.30,
    device: str | None = None,
    ) -> Path:
    """Run YOLO-seg inference on a source directory and save polygon predictions.

    Ultralytics writes `save_txt` labels in seg-polygon format automatically
    when the loaded model is a segmentation model, so this mirrors
    `run_yolo_inference` exactly.

    Returns:
        Path to the inference run folder containing predictions and `labels/`.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    model.predict(
        source=str(source_dir),
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        conf=conf,
        iou=iou,
        device=resolve_device(device),
        save=True,
        save_txt=True,
        save_conf=True,
    )

    return project_dir / run_name


def export_to_onnx(weights_path: Path, imgsz: int = 640) -> Path:
    """Export a trained YOLO model (detection or segmentation) to ONNX format.

    Args:
        weights_path: Path to a trained `.pt` checkpoint (e.g. `best.pt`).
        imgsz: Export image size.

    Returns:
        Path to the generated `.onnx` file.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    exported_path = model.export(format="onnx", imgsz=imgsz)
    return Path(exported_path)
