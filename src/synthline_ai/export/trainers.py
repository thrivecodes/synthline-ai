"""Export boilerplate training scripts and data loaders for ML frameworks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synthline_ai.config.models import GenerationConfig


def export_pytorch_dataset(
    output_dir: Path,
    config: GenerationConfig | None = None,
) -> Path:
    """Generate a zero-boilerplate PyTorch Dataset class for the generated data.

    Creates `output_dir/dataset_pytorch.py` which can be imported directly into
    torchvision, PyTorch Lightning, or custom segmentation architectures.
    """
    output_file = output_dir / "dataset_pytorch.py"

    code = '''"""PyTorch Dataset loader for SynthLine AI generated synthetic datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SynthLineDataset(Dataset):
    """PyTorch Dataset loading paired synthetic images and binary defect masks.

    Args:
        root_dir: Path to generated SynthLine dataset root.
        split: Partition to load ('train', 'val', 'test', or 'all').
        transform: Optional callable transform applied to image and mask.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "all",
        transform: (
            Callable[[np.ndarray, np.ndarray], tuple[torch.Tensor, torch.Tensor]] | None
        ) = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        self.images_dir = self.root_dir / "images"
        self.masks_dir = self.root_dir / "masks"

        # Check for VOC partition list first, otherwise scan images directory
        split_file = self.root_dir / "voc" / "ImageSets" / "Main" / f"{split}.txt"
        if split_file.exists():
            stems = [
                line.strip() for line in split_file.read_text().splitlines() if line.strip()
            ]
            self.samples = [
                (
                    self.images_dir / f"{s}.png",
                    self.masks_dir / f"{s.replace('image_', 'mask_')}.png",
                )
                for s in stems
            ]
        elif (self.root_dir / "yolo" / "labels" / split).exists():
            img_split_dir = self.root_dir / "yolo" / "images" / split
            self.samples = [
                (p, self.masks_dir / f"{p.stem.replace('image_', 'mask_')}.png")
                for p in sorted(img_split_dir.glob("*.png"))
            ]
        else:
            img_files = (
                sorted(self.images_dir.glob("*.png")) if self.images_dir.exists() else []
            )
            self.samples = [
                (img_p, self.masks_dir / f"{img_p.stem.replace('image_', 'mask_')}.png")
                for img_p in img_files
            ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        img_path, mask_path = self.samples[idx]

        # Load image (BGR -> RGB)
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Load binary mask (0 or 1)
        if mask_path.exists():
            mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask_binary = (mask_raw > 127).astype(np.float32)
        else:
            mask_binary = np.zeros(rgb.shape[:2], dtype=np.float32)

        if self.transform is not None:
            img_tensor, mask_tensor = self.transform(rgb, mask_binary)
        else:
            # Default to Normalized Tensor: (C, H, W) in [0, 1]
            img_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).float()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "filename": img_path.name,
        }


def get_dataloader(
    root_dir: str | Path,
    split: str = "all",
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a PyTorch DataLoader for the dataset."""
    ds = SynthLineDataset(root_dir=root_dir, split=split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    import sys
    base_dir = Path(__file__).parent
    ds = SynthLineDataset(base_dir)
    print(f"SynthLineDataset successfully verified: {len(ds)} samples found.")
    if len(ds) > 0:
        sample = ds[0]
        print(f"Sample 0 shape: image={sample['image'].shape}, mask={sample['mask'].shape}")
'''

    output_file.write_text(code, encoding="utf-8")
    return output_file


def export_yolo_train_script(
    output_dir: Path,
    config: GenerationConfig | None = None,
) -> Path:
    """Generate an executable training script for Ultralytics YOLO.

    Creates `output_dir/train_yolo.py` configured with the dataset yaml.
    """
    output_file = output_dir / "train_yolo.py"

    code = '''"""Train Ultralytics YOLO object detector on SynthLine AI synthetic dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO on SynthLine AI dataset")
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt", help="Base model weights (e.g. yolov8n.pt)"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="0", help="GPU device index or 'cpu'")
    parser.add_argument(
        "--project", type=str, default="synthline_runs", help="Output project folder"
    )
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics not installed. Install with: pip install ultralytics")
        return

    dataset_yaml = Path(__file__).parent / "yolo" / "data.yaml"
    if not dataset_yaml.exists():
        # Fallback to local data.yaml
        dataset_yaml = Path(__file__).parent / "data.yaml"

    if not dataset_yaml.exists():
        print(f"Error: YOLO configuration file '{dataset_yaml}' not found.")
        return

    print(f"Loading base model weights: {args.model}...")
    model = YOLO(args.model)

    print(f"Starting training on {dataset_yaml} for {args.epochs} epochs...")
    results = model.train(
        data=str(dataset_yaml.resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        save=True,
    )
    print("Training finished successfully. Model weights saved in project directory.")


if __name__ == "__main__":
    main()
'''

    output_file.write_text(code, encoding="utf-8")
    return output_file


def export_training_boilerplates(
    output_dir: Path,
    config: GenerationConfig | None = None,
) -> dict[str, Path]:
    """Export both PyTorch and YOLO training scaffolding files."""
    pt_path = export_pytorch_dataset(output_dir, config)
    yolo_path = export_yolo_train_script(output_dir, config)
    return {
        "pytorch": pt_path,
        "yolo": yolo_path,
    }
