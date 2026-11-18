"""Interactive manual review for the new bag-derived gate labels.

Compares Grounding DINO's boxes (tuned "black vertical pole." prompt)
against the existing trained model's own predictions on the same 203 new
frames, and lets you accept/reject/fix each one by hand.

Run this yourself — it opens a live OpenCV window with keyboard controls:
  a = accept current box       d = reject/delete
  b = back                     h/l = move selection (final review)
  s = save and move to next    f = accept all remaining, save
  k = skip image (no save)     q = quit
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Labeling.manual_review import ManualReviewConfig, ManualReviewSession

config = ManualReviewConfig(
    image_source=REPO_ROOT / "dataset" / "gate_dataset" / "raw_unlabeled" / "images",
    dino_labels=REPO_ROOT / "dataset" / "gate_dataset" / "dino_out" / "labels",
    yolo_labels=REPO_ROOT / "dataset" / "gate_yolo_predictions" / "existing_model_on_new_bag" / "labels",
    output_labels=REPO_ROOT / "dataset" / "gate_dataset" / "manual_review_labels",
)

if __name__ == "__main__":
    ManualReviewSession(config).run()
