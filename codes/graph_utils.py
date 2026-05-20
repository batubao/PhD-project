"""
Graph construction, cache handling, image transforms, and perturbation-based saliency.
"""

from contextlib import nullcontext
from typing import Dict, List, Tuple
import glob
import hashlib
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch_geometric.data import Data, Batch

from config import CFG



# graph/patch_metadata.py
from typing import Dict, List, Tuple

import numpy as np
import torch


def build_patch_meta(img_size: int, scales: List[Dict]) -> List[Dict]:
    """Build patch metadata for all scales."""
    meta_all = []

    for scale in scales:
        patch = scale["patch"]
        grid = img_size // patch

        for row in range(grid):
            for col in range(grid):
                x0, y0 = col * patch, row * patch
                x1, y1 = x0 + patch, y0 + patch
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0

                meta_all.append(
                    {
                        "patch": patch,
                        "grid": grid,
                        "row": row,
                        "col": col,
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "cx_norm": cx / img_size,
                        "cy_norm": cy / img_size,
                    }
                )

    return meta_all


def build_intrascale_edges(meta_all: List[Dict]) -> torch.Tensor:
    """Build 4-neighbour edges within each patch scale."""
    edges = []
    by_patch = {}

    for idx, meta in enumerate(meta_all):
        by_patch.setdefault(meta["patch"], []).append((idx, meta))

    for _, items in by_patch.items():
        grid = items[0][1]["grid"]
        coord_to_id = {(m["row"], m["col"]): idx for idx, m in items}

        for idx, meta in items:
            row, col = meta["row"], meta["col"]

            if col + 1 < grid:
                neighbour = coord_to_id[(row, col + 1)]
                edges.extend([(idx, neighbour), (neighbour, idx)])

            if row + 1 < grid:
                neighbour = coord_to_id[(row + 1, col)]
                edges.extend([(idx, neighbour), (neighbour, idx)])

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_interscale_edges(meta_all: List[Dict], scales: List[Dict]) -> torch.Tensor:
    """Build parent-child edges between adjacent patch scales."""
    patches = [scale["patch"] for scale in scales]
    idx_by_patch = {patch: [] for patch in patches}

    for idx, meta in enumerate(meta_all):
        idx_by_patch[meta["patch"]].append((idx, meta))

    edges = []

    for scale_idx in range(len(patches) - 1):
        coarse_patch = patches[scale_idx]
        fine_patch = patches[scale_idx + 1]

        coarse_items = idx_by_patch[coarse_patch]
        fine_items = idx_by_patch[fine_patch]

        if not coarse_items or not fine_items:
            continue

        coarse_grid = coarse_items[0][1]["grid"]
        fine_grid = fine_items[0][1]["grid"]
        coord_to_coarse_id = {(m["row"], m["col"]): idx for idx, m in coarse_items}

        for fine_idx, fine_meta in fine_items:
            parent_row = int(np.floor(fine_meta["row"] * coarse_grid / fine_grid))
            parent_col = int(np.floor(fine_meta["col"] * coarse_grid / fine_grid))

            parent_row = min(coarse_grid - 1, max(0, parent_row))
            parent_col = min(coarse_grid - 1, max(0, parent_col))

            coarse_idx = coord_to_coarse_id[(parent_row, parent_col)]
            edges.extend([(fine_idx, coarse_idx), (coarse_idx, fine_idx)])

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_medium_crossscale_map(meta_all: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build neighbour map for cross-scale attention.

    Medium nodes act as queries. Coarse and fine nodes whose centres fall inside
    each medium patch are used as keys and values.
    """
    coarse_idx = [i for i, m in enumerate(meta_all) if m["patch"] == 128]
    medium_idx = [i for i, m in enumerate(meta_all) if m["patch"] == 64]
    fine_idx = [i for i, m in enumerate(meta_all) if m["patch"] == 32]

    boxes = np.array(
        [[m["x0"], m["y0"], m["x1"], m["y1"]] for m in meta_all],
        dtype=np.float32,
    )
    cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
    cy = 0.5 * (boxes[:, 1] + boxes[:, 3])

    neighbour_map = []
    max_len = 0

    for medium_node in medium_idx:
        x0, y0, x1, y1 = boxes[medium_node]
        neighbours = []

        for node_idx in coarse_idx + fine_idx:
            inside_x = cx[node_idx] >= x0 and cx[node_idx] <= x1
            inside_y = cy[node_idx] >= y0 and cy[node_idx] <= y1

            if inside_x and inside_y:
                neighbours.append(node_idx)

        neighbour_map.append(neighbours)
        max_len = max(max_len, len(neighbours))

    neighbour_tensor = torch.full((len(medium_idx), max_len), -1, dtype=torch.long)
    mask_tensor = torch.zeros((len(medium_idx), max_len), dtype=torch.bool)

    for row, neighbours in enumerate(neighbour_map):
        neighbour_tensor[row, : len(neighbours)] = torch.tensor(neighbours, dtype=torch.long)
        mask_tensor[row, : len(neighbours)] = True

    return torch.tensor(medium_idx, dtype=torch.long), neighbour_tensor, mask_tensor


def build_single_medium_edge_template(grid: int = 4) -> torch.Tensor:
    """Build 4-neighbour edges for the retained 4 x 4 medium graph."""
    edges = []

    def node_id(row, col):
        return row * grid + col

    for row in range(grid):
        for col in range(grid):
            if col + 1 < grid:
                a, b = node_id(row, col), node_id(row, col + 1)
                edges.extend([(a, b), (b, a)])

            if row + 1 < grid:
                a, b = node_id(row, col), node_id(row + 1, col)
                edges.extend([(a, b), (b, a)])

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_graph_templates(img_size: int, scales: List[Dict]):
    """Create reusable graph templates used by the full pipeline."""
    meta = build_patch_meta(img_size, scales)

    edge_index = torch.cat(
        [
            build_intrascale_edges(meta),
            build_interscale_edges(meta, scales),
        ],
        dim=1,
    ).contiguous()

    pos = torch.tensor([[m["cx_norm"], m["cy_norm"]] for m in meta], dtype=torch.float)
    rowcol = torch.tensor([[m["row"], m["col"]] for m in meta], dtype=torch.long)
    patch_size = torch.tensor([m["patch"] for m in meta], dtype=torch.long)
    box = torch.tensor([[m["x0"], m["y0"], m["x1"], m["y1"]] for m in meta], dtype=torch.float)

    medium_local_idx, csa_neigh_idx, csa_neigh_mask = build_medium_crossscale_map(meta)
    medium_edge_template = build_single_medium_edge_template(grid=4)

    return {
        "meta": meta,
        "edge_index": edge_index,
        "pos": pos,
        "rowcol": rowcol,
        "patch_size": patch_size,
        "box": box,
        "medium_local_idx": medium_local_idx,
        "csa_neigh_idx": csa_neigh_idx,
        "csa_neigh_mask": csa_neigh_mask,
        "medium_edge_template": medium_edge_template,
        "nodes_per_graph": len(meta),
        "medium_start": 4,
        "medium_end": 20,
        "medium_nodes_per_graph": 16,
    }


# utils/cache_utils.py
import hashlib
import os


def cache_key(path: str) -> str:
    """Create a stable hash key for an image path."""
    return hashlib.md5(path.encode("utf-8")).hexdigest()


def full_graph_path(img_path: str, cache_full_dir: str) -> str:
    return os.path.join(cache_full_dir, f"{cache_key(img_path)}.pt")


def saliency_path(img_path: str, cache_sal_dir: str) -> str:
    return os.path.join(cache_sal_dir, f"{cache_key(img_path)}.pt")


# graph/graph_builder.py
from contextlib import nullcontext
from typing import Dict

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch_geometric.data import Data



crop_tf = transforms.Compose(
    [
        transforms.Resize((CFG.crop_resize, CFG.crop_resize)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

full_img_tf = transforms.Compose(
    [
        transforms.Resize((CFG.img_size, CFG.img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def autocast_context(device: str):
    """Return automatic mixed precision context when running on CUDA."""
    if device == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return nullcontext()


def load_raw_pil(path: str) -> Image.Image:
    """Load image as resized RGB PIL image."""
    image = Image.open(path).convert("L")
    image = image.resize((CFG.img_size, CFG.img_size), Image.BILINEAR)
    return image.convert("RGB")


def load_full_tensor(path: str) -> torch.Tensor:
    """Load full image tensor for the dense contextual branch."""
    image = Image.open(path).convert("L")
    return full_img_tf(image)


def crop_patch(pil_img: Image.Image, x0, y0, x1, y1) -> torch.Tensor:
    """Crop and transform one image patch."""
    crop = pil_img.crop((x0, y0, x1, y1))
    return crop_tf(crop)


@torch.no_grad()
def build_full_graph(
    img_path: str,
    label: int,
    templates: Dict,
    backbone,
    device: str,
    mini_batch: int = None,
) -> Data:
    """Build one full multiscale patch graph from an image."""
    if mini_batch is None:
        mini_batch = CFG.patch_encode_batch

    pil_img = load_raw_pil(img_path)

    crops = [
        crop_patch(pil_img, m["x0"], m["y0"], m["x1"], m["y1"])
        for m in templates["meta"]
    ]

    feat_list = []

    for start in range(0, len(crops), mini_batch):
        batch = torch.stack(crops[start : start + mini_batch], dim=0)
        batch = batch.to(device, non_blocking=True)

        with autocast_context(device):
            fmap = backbone(batch)
            fmap = F.relu(fmap)
            vec = F.adaptive_avg_pool2d(fmap, 1)

        feat_list.append(vec.view(vec.size(0), -1).float().cpu())

    feats = torch.cat(feat_list, dim=0)
    feats = (feats - feats.mean(dim=0, keepdim=True)) / (
        feats.std(dim=0, keepdim=True) + 1e-6
    )

    data = Data(
        x=feats,
        edge_index=templates["edge_index"],
        y=torch.tensor([int(label)], dtype=torch.long),
        pos=templates["pos"],
    )

    data.rowcol = templates["rowcol"]
    data.patch_size = templates["patch_size"]
    data.box = templates["box"]

    return data


def build_medium_only_graph_batched(data, x_updated, templates: Dict):
    """Collapse the multiscale graph into a refined medium-only graph."""
    device = x_updated.device
    batch_size = int(data.y.view(-1).numel())
    nodes_per_graph = templates["nodes_per_graph"]
    medium_nodes = templates["medium_nodes_per_graph"]
    feature_dim = x_updated.size(1)

    x_b = x_updated.view(batch_size, nodes_per_graph, feature_dim)
    pos_b = data.pos.view(batch_size, nodes_per_graph, 2)

    medium_start = templates["medium_start"]
    medium_end = templates["medium_end"]

    x_m = x_b[:, medium_start:medium_end, :].reshape(batch_size * medium_nodes, feature_dim)
    pos_m = pos_b[:, medium_start:medium_end, :].reshape(batch_size * medium_nodes, 2)

    batch_m = torch.arange(batch_size, device=device).repeat_interleave(medium_nodes)

    base_edge = templates["medium_edge_template"].to(device)
    edge_blocks = [base_edge + b * medium_nodes for b in range(batch_size)]
    edge_index = torch.cat(edge_blocks, dim=1)

    out = Data(
        x=x_m,
        edge_index=edge_index,
        y=data.y.view(-1).to(device),
        pos=pos_m,
    )
    out.batch = batch_m

    return out


# graph/graph_cache.py
import os
from typing import List

import torch



def cache_full_graphs(
    img_paths: List[str],
    labels,
    cache_full_dir: str,
    templates,
    backbone,
    device: str,
) -> List[str]:
    """Build and cache full multiscale graphs."""
    os.makedirs(cache_full_dir, exist_ok=True)

    output_files = [full_graph_path(path, cache_full_dir) for path in img_paths]
    missing = sum(0 if os.path.exists(path) else 1 for path in output_files)

    print(f"Missing full graphs: {missing} / {len(output_files)}")

    built = 0
    for img_path, label, out_path in zip(img_paths, labels, output_files):
        if os.path.exists(out_path):
            continue

        graph = build_full_graph(
            img_path=img_path,
            label=int(label),
            templates=templates,
            backbone=backbone,
            device=device,
        )

        tmp_path = out_path + ".tmp"
        torch.save(graph, tmp_path)
        os.replace(tmp_path, out_path)

        built += 1
        if built % 50 == 0:
            print(f"  built {built}/{missing}")

    print("Full graph caching done.")
    return output_files


def get_saliency_files(img_paths: List[str], cache_sal_dir: str) -> List[str]:
    """Return expected saliency cache paths."""
    os.makedirs(cache_sal_dir, exist_ok=True)
    return [saliency_path(path, cache_sal_dir) for path in img_paths]


# graph/saliency.py
import os
from typing import List

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch



@torch.no_grad()
def perturb_scores_single_fast(
    model,
    data,
    device: str,
    target_class=None,
    mask_value: float = 0.0,
    chunk_size: int = None,
):
    """
    Estimate node importance by measuring probability drop after node perturbation.
    """
    if chunk_size is None:
        chunk_size = CFG.perturb_chunk

    model.eval()

    base_data = data.clone().to(device)
    base_data.batch = torch.zeros(base_data.num_nodes, dtype=torch.long, device=device)

    base_probs = F.softmax(model(base_data), dim=1)[0]
    pred_class = int(base_probs.argmax().item())

    if target_class is None:
        target_class = pred_class

    base_score = float(base_probs[target_class].item())
    num_nodes = base_data.x.size(0)
    scores = torch.zeros(num_nodes, dtype=torch.float32)

    for start in range(0, num_nodes, chunk_size):
        end = min(start + chunk_size, num_nodes)
        masked_graphs = []

        for node_idx in range(start, end):
            graph = Data(
                x=base_data.x.clone(),
                edge_index=base_data.edge_index,
                y=base_data.y,
                pos=base_data.pos,
            )
            graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
            graph.x[node_idx].fill_(mask_value)
            masked_graphs.append(graph)

        batch_obj = Batch.from_data_list(masked_graphs).to(device)

        with autocast_context(device):
            probs = F.softmax(model(batch_obj), dim=1)[:, target_class]

        scores[start:end] = (base_score - probs).detach().cpu()

    return scores


def cache_saliency_scores(
    graph_files: List[str],
    saliency_files: List[str],
    model,
    device: str,
) -> None:
    """Compute and cache perturbation-based node saliency for each graph."""
    if len(saliency_files) == 0:
        return

    os.makedirs(os.path.dirname(saliency_files[0]), exist_ok=True)

    missing = sum(0 if os.path.exists(path) else 1 for path in saliency_files)
    print(f"Missing saliency files: {missing} / {len(saliency_files)}")

    done = 0
    for graph_file, saliency_file in zip(graph_files, saliency_files):
        if os.path.exists(saliency_file):
            continue

        graph = torch.load(graph_file, map_location="cpu", weights_only=False)
        scores = perturb_scores_single_fast(model, graph, device=device)

        tmp_path = saliency_file + ".tmp"
        torch.save(scores, tmp_path)
        os.replace(tmp_path, saliency_file)

        done += 1
        if done % 100 == 0:
            print(f"  scored {done}/{missing}")

    print("Saliency caching done.")
