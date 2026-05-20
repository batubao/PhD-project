"""
Dataset preparation utilities for the SMPG pipeline.

The repository supports five datasets used in the thesis. Each dataset has its own
Kaggle URL and folder preparation rule. After preparation, the returned root
folder must contain one subfolder per class.
"""

import glob
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import opendatasets as od
import torch
from torch.utils.data import Dataset

from graph_utils import load_full_tensor


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass(frozen=True)
class DatasetInfo:
    dataset_id: int
    short_name: str
    kaggle_url: str
    raw_root: str
    prepared_root: str


DATASETS: Dict[int, DatasetInfo] = {
    1: DatasetInfo(
        dataset_id=1,
        short_name="Paediatric Chest X-ray Pneumonia dataset",
        kaggle_url="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
        raw_root="/content/chest-xray-pneumonia/chest_xray",
        prepared_root="/content/chest-xray-pneumonia/chest_xray/lung",
    ),
    2: DatasetInfo(
        dataset_id=2,
        short_name="Chest X-ray PA dataset",
        kaggle_url="https://www.kaggle.com/datasets/amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset",
        raw_root="/content/covid19-pneumonia-normal-chest-xray-pa-dataset",
        prepared_root="/content/covid19-pneumonia-normal-chest-xray-pa-dataset",
    ),
    3: DatasetInfo(
        dataset_id=3,
        short_name="SARS-CoV-2 CT-scan dataset",
        kaggle_url="https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset",
        raw_root="/content/sarscov2-ctscan-dataset",
        prepared_root="/content/sarscov2-ctscan-dataset",
    ),
    4: DatasetInfo(
        dataset_id=4,
        short_name="COVID-19 Radiography Database",
        kaggle_url="https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database",
        raw_root="/content/covid19-radiography-database/COVID-19_Radiography_Dataset",
        prepared_root="/content/covid19-radiography-database/dataset_clean",
    ),
    5: DatasetInfo(
        dataset_id=5,
        short_name="Chest X-ray (COVID-19 & Pneumonia) dataset",
        kaggle_url="https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia/data",
        raw_root="/content/chest-xray-covid19-pneumonia/Data",
        prepared_root="/content/chest-xray-covid19-pneumonia/Data/lung",
    ),
}


def download_kaggle_dataset(dataset_id: int) -> None:
    """Download one of the thesis datasets from Kaggle."""
    info = DATASETS[dataset_id]
    print(f"Downloading Dataset {dataset_id}: {info.short_name}")
    od.download(info.kaggle_url)


def merge_split_folders(
    root_dir: str,
    split_names=("train", "test", "val"),
    destination_name: str = "lung",
    move_files: bool = True,
) -> str:
    """
    Merge images from split folders into one class-based folder.

    Example input:
        root/train/NORMAL/*.jpg
        root/test/NORMAL/*.jpg
        root/val/NORMAL/*.jpg

    Example output:
        root/lung/NORMAL/*.jpg
    """
    root = Path(root_dir)
    destination_root = root / destination_name
    destination_root.mkdir(parents=True, exist_ok=True)

    for split in split_names:
        split_path = root / split
        if not split_path.exists():
            print(f"Skipped missing split folder: {split_path}")
            continue

        for class_dir in split_path.iterdir():
            if not class_dir.is_dir():
                continue

            dest_class_dir = destination_root / class_dir.name
            dest_class_dir.mkdir(parents=True, exist_ok=True)

            for src_file in class_dir.iterdir():
                if not src_file.is_file():
                    continue
                if src_file.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                dest_file = dest_class_dir / src_file.name
                if dest_file.exists():
                    dest_file = dest_class_dir / f"{split}_{src_file.name}"

                if move_files:
                    shutil.move(str(src_file), str(dest_file))
                else:
                    shutil.copy2(str(src_file), str(dest_file))

    print("Merge complete. All images are now in:", destination_root)
    return str(destination_root)


def prepare_covid19_radiography_database() -> str:
    """
    Prepare Dataset 4 by copying images from each class Images folder into a clean root.

    Input:
        /content/covid19-radiography-database/COVID-19_Radiography_Dataset/<class>/images

    Output:
        /content/covid19-radiography-database/dataset_clean/<class_name>/*.png
    """
    src_root = Path(DATASETS[4].raw_root)
    dst_root = Path(DATASETS[4].prepared_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    for cls in os.listdir(src_root):
        cls_path = src_root / cls
        if not cls_path.is_dir():
            continue

        img_folder = None
        for sub in os.listdir(cls_path):
            if sub.lower() == "images":
                img_folder = cls_path / sub
                break

        if img_folder is None:
            print(f"No images folder in {cls}")
            continue

        clean_name = cls.replace(" ", "_").replace("-", "_")
        dst_cls_dir = dst_root / clean_name
        dst_cls_dir.mkdir(parents=True, exist_ok=True)

        for fname in os.listdir(img_folder):
            if fname.lower().endswith(IMAGE_EXTENSIONS):
                shutil.copy2(str(img_folder / fname), str(dst_cls_dir / fname))

        print(f"Processed: {cls} -> {clean_name}")

    print("Done. Clean dataset directory:", dst_root)
    return str(dst_root)


def prepare_dataset(dataset_id: int, download: bool = True) -> str:
    """
    Download and prepare a dataset, then return the final root directory.

    The returned root directory is the path that should be passed to
    discover_classes_and_paths().
    """
    if dataset_id not in DATASETS:
        raise ValueError(f"Unknown dataset_id={dataset_id}. Use one of: {sorted(DATASETS)}")

    info = DATASETS[dataset_id]

    if download:
        download_kaggle_dataset(dataset_id)

    if dataset_id == 1:
        return merge_split_folders(
            root_dir=info.raw_root,
            split_names=("train", "test", "val"),
            destination_name="lung",
            move_files=True,
        )

    if dataset_id == 4:
        return prepare_covid19_radiography_database()

    if dataset_id == 5:
        return merge_split_folders(
            root_dir=info.raw_root,
            split_names=("train", "test"),
            destination_name="lung",
            move_files=True,
        )

    print(f"Dataset {dataset_id} does not require extra folder merging.")
    print("Dataset root:", info.prepared_root)
    return info.prepared_root


def get_dataset_root(dataset_id: int) -> str:
    """Return the expected prepared root path without downloading or reorganising."""
    return DATASETS[dataset_id].prepared_root


def discover_classes_and_paths(root: str) -> Tuple[List[str], List[str], np.ndarray]:
    """Discover class folders and image paths from a dataset root directory."""
    class_names = sorted(
        [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    )
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    paths, labels = [], []
    for class_name in class_names:
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            files += glob.glob(os.path.join(root, class_name, ext))

        files = sorted(files)
        paths.extend(files)
        labels.extend([class_to_idx[class_name]] * len(files))

    return class_names, paths, np.array(labels, dtype=np.int64)


class CachedGraphDataset(Dataset):
    """Dataset that loads cached PyG graph files."""

    def __init__(self, graph_files: List[str]):
        self.graph_files = graph_files

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        return torch.load(self.graph_files[idx], map_location="cpu", weights_only=False)


class GraphWithSalAndImageDataset(Dataset):
    """Dataset that returns cached graph, saliency vector, and full image tensor."""

    def __init__(self, graph_files: List[str], saliency_files: List[str], img_paths: List[str]):
        self.graph_files = graph_files
        self.saliency_files = saliency_files
        self.img_paths = img_paths

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        graph = torch.load(self.graph_files[idx], map_location="cpu", weights_only=False)
        saliency = torch.load(self.saliency_files[idx], map_location="cpu", weights_only=False)
        full_img = load_full_tensor(self.img_paths[idx])

        graph.node_sal = saliency.float()
        graph.full_img = full_img
        return graph


if __name__ == "__main__":
    # Change this number to prepare another dataset.
    final_root = prepare_dataset(dataset_id=1, download=True)
    print("Final root_dir:", final_root)
