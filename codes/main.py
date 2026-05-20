"""
Main entry point for running the Saliency-Guided Multiscale Patch Graph pipeline.
"""

import argparse
import os

import torch

from config import CFG
from graph_utils import (
    build_graph_templates,
    cache_full_graphs,
    cache_saliency_scores,
    get_saliency_files,
)
from models import make_densenet_encoder
from prepare_dataset import DATASETS, discover_classes_and_paths, get_dataset_root, prepare_dataset
from train import run_cross_validation, train_baseline_model


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    return device


def parse_args():
    parser = argparse.ArgumentParser(description="Run the SMPG medical image classification pipeline.")
    parser.add_argument("--dataset", type=int, default=CFG.dataset_id, choices=sorted(DATASETS.keys()))
    parser.add_argument("--no-download", action="store_true", help="Skip Kaggle download.")
    parser.add_argument("--no-prepare", action="store_true", help="Skip dataset organisation and use the expected prepared root.")
    parser.add_argument("--root-dir", type=str, default=None, help="Manually provide a class-folder dataset root.")
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(CFG.seed)

    dataset_info = DATASETS[args.dataset]
    print(f"Selected Dataset {args.dataset}: {dataset_info.short_name}")

    if args.root_dir is not None:
        root_dir = args.root_dir
        print("Using manual root_dir:", root_dir)
    elif args.no_prepare:
        root_dir = get_dataset_root(args.dataset)
        print("Using expected prepared root_dir:", root_dir)
    elif CFG.auto_prepare_dataset:
        root_dir = prepare_dataset(dataset_id=args.dataset, download=not args.no_download)
    else:
        root_dir = CFG.root_dir
        print("Using CFG.root_dir:", root_dir)

    os.makedirs(CFG.cache_full_dir, exist_ok=True)
    os.makedirs(CFG.cache_sal_dir, exist_ok=True)

    device = prepare_device()

    class_names, img_paths, labels = discover_classes_and_paths(root_dir)
    num_classes = len(class_names)

    print("Root directory:", root_dir)
    print("Classes:", class_names)
    print("Total images:", len(img_paths))
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {int((labels == i).sum())}")

    templates = build_graph_templates(CFG.img_size, CFG.scales)
    print(f"Nodes per graph: {templates['nodes_per_graph']} (4 coarse + 16 medium + 64 fine)")

    backbone, feat_dim = make_densenet_encoder()
    backbone = backbone.to(device)

    print("\nStep 1: Cache full multiscale graphs")
    graph_files = cache_full_graphs(
        img_paths=img_paths,
        labels=labels,
        cache_full_dir=CFG.cache_full_dir,
        templates=templates,
        backbone=backbone,
        device=device,
    )

    saliency_files = get_saliency_files(img_paths, CFG.cache_sal_dir)

    print("\nStep 2: Train baseline GCN for saliency estimation")
    sample_graph = torch.load(graph_files[0], map_location="cpu", weights_only=False)
    in_dim = sample_graph.x.size(1)

    baseline_model = train_baseline_model(
        graph_files=graph_files,
        in_dim=in_dim,
        num_classes=num_classes,
        device=device,
    )

    print("\nStep 3: Cache perturbation-based saliency")
    cache_saliency_scores(
        graph_files=graph_files,
        saliency_files=saliency_files,
        model=baseline_model,
        device=device,
    )

    print("\nStep 4: Train and evaluate final SMPG model")
    run_cross_validation(
        graph_files=graph_files,
        saliency_files=saliency_files,
        img_paths=img_paths,
        labels=labels,
        templates=templates,
        backbone=backbone,
        in_dim=in_dim,
        num_classes=num_classes,
        device=device,
    )


if __name__ == "__main__":
    main()
