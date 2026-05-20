"""
Training, evaluation, metrics, and cross-validation utilities.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

from config import CFG
from prepare_dataset import CachedGraphDataset, GraphWithSalAndImageDataset
from graph_utils import autocast_context
from models import BaselineGCN, SaliencyGuidedMultiscalePatchGraph



# utils/metrics.py
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def compute_classification_metrics(y_true, y_pred, y_prob, num_classes: int):
    if len(y_true) == 0:
        return {
            "acc": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "auc": float("nan"),
        }

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    acc = float(np.mean(y_true == y_pred))
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return {
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision": precision,
        "recall": recall,
        "auc": auc,
    }


def summarise_fold_metrics(fold_metrics):
    keys = fold_metrics[0].keys()
    summary = {}
    for key in keys:
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        summary[key] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
        }
    return summary


# training/train_baseline.py
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader



def train_one_epoch_baseline(model, loader, optimizer, criterion, device: str, scaler=None):
    """Train baseline GCN for one epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device):
            logits = model(batch)
            y = batch.y.view(-1)
            loss = criterion(logits, y)

        if scaler is not None and device == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()

    return total_loss / max(total, 1), correct / max(total, 1)


def train_baseline_model(graph_files, in_dim: int, num_classes: int, device: str):
    """Train baseline GCN on full multiscale graphs."""
    dataset = CachedGraphDataset(graph_files)

    loader = DataLoader(
        dataset,
        batch_size=CFG.batch_size_baseline,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=CFG.pin_memory,
        persistent_workers=CFG.persistent_workers,
    )

    model = BaselineGCN(
        in_dim=in_dim,
        hidden=CFG.hidden_dim,
        num_classes=num_classes,
        dropout=CFG.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG.lr_base,
        weight_decay=CFG.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    print("=" * 80)
    print("Training baseline GCN for saliency estimation")
    print("=" * 80)

    for epoch in range(1, CFG.epochs_baseline + 1):
        train_loss, train_acc = train_one_epoch_baseline(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
        )

        print(
            f"Epoch {epoch:02d}/{CFG.epochs_baseline} | "
            f"train loss {train_loss:.4f} | train acc {train_acc:.4f}"
        )

    return model


# training/train_final.py
def train_one_epoch_final(model, loader, optimizer, criterion, device: str, scaler=None):
    """Train final SMPG model for one epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        sal = batch.node_sal.to(device, non_blocking=True)
        full_img = batch.full_img.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device):
            logits = model(batch, sal, full_img)
            y = batch.y.view(-1).to(device)
            loss = criterion(logits, y)

        if scaler is not None and device == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()

    return total_loss / max(total, 1), correct / max(total, 1)


# training/evaluate.py
import numpy as np
import torch



@torch.no_grad()
def evaluate_model(model, loader, num_classes: int, device: str):
    """Evaluate SMPG model."""
    model.eval()

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        sal = batch.node_sal.to(device, non_blocking=True)
        full_img = batch.full_img.to(device, non_blocking=True)

        with autocast_context(device):
            logits = model(batch, sal, full_img)

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(1)
        y = batch.y.view(-1).to(device)

        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(preds.detach().cpu().numpy())
        y_prob_all.append(probs.detach().cpu().numpy())

    if not y_true_all:
        return compute_classification_metrics([], [], [], num_classes)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_prob = np.concatenate(y_prob_all, axis=0)

    return compute_classification_metrics(y_true, y_pred, y_prob, num_classes)


# training/cross_validation.py
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader



def run_cross_validation(
    graph_files,
    saliency_files,
    img_paths,
    labels,
    templates,
    backbone,
    in_dim: int,
    num_classes: int,
    device: str,
):
    """Run stratified k-fold cross-validation for the final SMPG model."""
    skf = StratifiedKFold(
        n_splits=CFG.k_folds,
        shuffle=True,
        random_state=CFG.seed,
    )

    fold_best_metrics = []

    print("=" * 90)
    print("5-fold CV: Saliency-guided multiscale patch graph model")
    print("=" * 90)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels),
        start=1,
    ):
        print("=" * 90)
        print(f"FOLD {fold}/{CFG.k_folds}")
        print("=" * 90)

        train_graphs = [graph_files[i] for i in train_idx]
        train_sals = [saliency_files[i] for i in train_idx]
        train_imgs = [img_paths[i] for i in train_idx]

        val_graphs = [graph_files[i] for i in val_idx]
        val_sals = [saliency_files[i] for i in val_idx]
        val_imgs = [img_paths[i] for i in val_idx]

        train_dataset = GraphWithSalAndImageDataset(train_graphs, train_sals, train_imgs)
        val_dataset = GraphWithSalAndImageDataset(val_graphs, val_sals, val_imgs)

        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.batch_size_final,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=CFG.pin_memory,
            persistent_workers=CFG.persistent_workers,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=CFG.batch_size_final,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=CFG.pin_memory,
            persistent_workers=CFG.persistent_workers,
        )

        model = SaliencyGuidedMultiscalePatchGraph(
            feat_dim=in_dim,
            templates=templates,
            backbone=backbone,
            hidden=CFG.hidden_dim,
            num_classes=num_classes,
            dropout=CFG.dropout,
            attn_dim=CFG.attn_dim,
            edge_ctx_dim=CFG.edge_ctx_dim,
            use_triplet_attention=True,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=CFG.lr_final,
            weight_decay=CFG.weight_decay,
        )

        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

        best_acc = -1.0
        best_metrics = None

        for epoch in range(1, CFG.epochs_final + 1):
            train_loss, train_acc = train_one_epoch_final(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=scaler,
            )

            val_metrics = evaluate_model(
                model=model,
                loader=val_loader,
                num_classes=num_classes,
                device=device,
            )

            print(
                f"Epoch {epoch:03d}/{CFG.epochs_final} | "
                f"train loss {train_loss:.4f} | train acc {train_acc:.4f} | "
                f"val acc {val_metrics['acc']:.4f} | "
                f"val f1_macro {val_metrics['f1_macro']:.4f} | "
                f"val f1_weighted {val_metrics['f1_weighted']:.4f} | "
                f"val precision {val_metrics['precision']:.4f} | "
                f"val recall {val_metrics['recall']:.4f} | "
                f"val auc {val_metrics['auc']:.4f}"
            )

            if val_metrics["acc"] > best_acc:
                best_acc = val_metrics["acc"]
                best_metrics = val_metrics.copy()

        fold_best_metrics.append(best_metrics)

        print(
            f"Best fold {fold}: "
            f"ACC {best_metrics['acc']:.4f} | "
            f"F1-macro {best_metrics['f1_macro']:.4f} | "
            f"F1-weighted {best_metrics['f1_weighted']:.4f} | "
            f"Precision {best_metrics['precision']:.4f} | "
            f"Recall {best_metrics['recall']:.4f} | "
            f"AUC {best_metrics['auc']:.4f}"
        )

    print("=" * 100)
    print("5-FOLD SUMMARY")
    print("=" * 100)

    for i, metrics in enumerate(fold_best_metrics, start=1):
        print(
            f"Fold {i}: "
            f"ACC {metrics['acc']:.4f} | "
            f"F1-macro {metrics['f1_macro']:.4f} | "
            f"F1-weighted {metrics['f1_weighted']:.4f} | "
            f"Precision {metrics['precision']:.4f} | "
            f"Recall {metrics['recall']:.4f} | "
            f"AUC {metrics['auc']:.4f}"
        )

    summary = summarise_fold_metrics(fold_best_metrics)

    print("-" * 100)
    for metric_name, values in summary.items():
        print(f"Mean {metric_name:12s}: {values['mean']:.4f} ± {values['std']:.4f}")

    return fold_best_metrics, summary
