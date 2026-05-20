"""
Model definitions for the SMPG pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch_geometric.nn import GCNConv, global_mean_pool

from config import CFG
from graph_utils import autocast_context, build_medium_only_graph_batched



# models/densenet_encoder.py
from torchvision import models


def make_densenet_encoder():
    """Create DenseNet201 feature extractor."""
    model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
    backbone = model.features
    backbone.eval()
    feature_dim = 1920
    return backbone, feature_dim


# models/baseline_gcn.py
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class BaselineGCN(nn.Module):
    """Baseline GCN used for initial graph classification and saliency estimation."""

    def __init__(self, in_dim: int, hidden: int = 128, num_classes: int = 2, dropout: float = 0.25):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.conv2(x, edge_index))

        pooled = global_mean_pool(x, batch)
        pooled = F.relu(self.lin1(pooled))
        pooled = F.dropout(pooled, p=self.dropout, training=self.training)

        return self.lin2(pooled)


# models/cross_scale_attention.py
import torch
import torch.nn as nn


class SaliencyGuidedCrossScaleAttention(nn.Module):
    """
    Saliency-guided cross-scale attention.

    Medium-scale nodes are updated using information from their corresponding
    coarse-scale and fine-scale neighbouring nodes.
    """

    def __init__(self, feat_dim: int, templates, attn_dim: int = 128, pos_dim: int = 2):
        super().__init__()
        self.templates = templates
        self.wq = nn.Linear(feat_dim + pos_dim, attn_dim)
        self.wk = nn.Linear(feat_dim + pos_dim, attn_dim)
        self.wv = nn.Linear(feat_dim, feat_dim)

    @staticmethod
    def l2norm(tensor, eps: float = 1e-8):
        return tensor / (tensor.norm(p=2, dim=-1, keepdim=True) + eps)

    def forward(self, x, pos, batch_vec, sal):
        device = x.device
        batch_size = int(batch_vec.max().item()) + 1
        nodes_per_graph = self.templates["nodes_per_graph"]
        feature_dim = x.size(1)

        x_b = x.view(batch_size, nodes_per_graph, feature_dim)
        pos_b = pos.view(batch_size, nodes_per_graph, 2)
        sal_b = sal.view(batch_size, nodes_per_graph, 1)

        x_gated = x_b * sal_b
        qk_input = torch.cat([x_gated, pos_b], dim=-1)

        q = self.l2norm(self.wq(qk_input))
        k = self.l2norm(self.wk(qk_input))
        v = self.wv(x_b)

        medium_idx = self.templates["medium_local_idx"].to(device)
        neigh_idx = self.templates["csa_neigh_idx"].to(device)
        neigh_mask = self.templates["csa_neigh_mask"].to(device)

        q_medium = q[:, medium_idx, :]
        sal_medium = sal_b[:, medium_idx, :]

        neigh_idx_safe = neigh_idx.clamp(min=0)

        k_neigh = k[:, neigh_idx_safe, :]
        v_neigh = v[:, neigh_idx_safe, :]
        sal_neigh = sal_b[:, neigh_idx_safe, :]

        similarity = (q_medium.unsqueeze(2) * k_neigh).sum(dim=-1)
        saliency_bias = sal_medium.unsqueeze(2) + sal_neigh

        logits = similarity.unsqueeze(-1) + saliency_bias
        logits = logits.masked_fill(~neigh_mask.unsqueeze(0).unsqueeze(-1), -1e9)

        attention = torch.softmax(logits, dim=2)
        message = (attention * v_neigh).sum(dim=2)

        x_out = x_b.clone()
        x_out[:, medium_idx, :] = x_out[:, medium_idx, :] + message

        return x_out.view(batch_size * nodes_per_graph, feature_dim)


# models/dense_edge_enhancer.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    """Convolution, batch normalisation, and ReLU block."""

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SpatialGate(nn.Module):
    """Spatial attention gate used by Triplet Attention."""

    def __init__(self):
        super().__init__()
        self.compress_conv = BasicConv(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_cat = torch.cat([x_max, x_avg], dim=1)
        scale = torch.sigmoid(self.compress_conv(x_cat))
        return x * scale


class TripletAttention(nn.Module):
    """
    Triplet attention module.

    If Triplet Attention is not part of the final thesis version, set
    use_triplet_attention=False in DenseEdgeEnhancer.
    """

    def __init__(self):
        super().__init__()
        self.cw = SpatialGate()
        self.hc = SpatialGate()
        self.hw = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out1 = x_out1.permute(0, 2, 1, 3).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out2 = x_out2.permute(0, 3, 2, 1).contiguous()

        x_out3 = self.hw(x)
        return (x_out1 + x_out2 + x_out3) / 3.0


class DenseEdgeEnhancer(nn.Module):
    """
    Dense contextual edge enhancement branch.

    DenseNet full-image feature maps are compressed into a contextual map, which is
    later used to estimate adaptive edge weights for the retained medium graph.
    """

    def __init__(self, in_ch: int = 1920, ctx_dim: int = 256, use_triplet_attention: bool = True):
        super().__init__()
        self.triplet = TripletAttention() if use_triplet_attention else nn.Identity()

        self.reduce = nn.Conv2d(in_ch, ctx_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(ctx_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, fmap):
        fmap = F.adaptive_avg_pool2d(fmap, (8, 8))
        fmap = self.triplet(fmap)
        fmap = self.act(self.bn(self.reduce(fmap)))
        return fmap


# models/smpg_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool



class SaliencyGuidedMultiscalePatchGraph(nn.Module):
    """
    Main SMPG model.

    The model performs:
    1. saliency-guided cross-scale attention,
    2. medium-only graph construction,
    3. dense contextual edge enhancement,
    4. GCN-based graph classification.
    """

    def __init__(
        self,
        feat_dim: int,
        templates,
        backbone,
        hidden: int = 128,
        num_classes: int = 2,
        dropout: float = 0.25,
        attn_dim: int = 128,
        edge_ctx_dim: int = 256,
        use_triplet_attention: bool = True,
    ):
        super().__init__()
        self.templates = templates
        self.backbone = backbone

        self.csa = SaliencyGuidedCrossScaleAttention(
            feat_dim=feat_dim,
            templates=templates,
            attn_dim=attn_dim,
        )

        self.edge_branch = DenseEdgeEnhancer(
            in_ch=1920,
            ctx_dim=edge_ctx_dim,
            use_triplet_attention=use_triplet_attention,
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_ctx_dim * 3, edge_ctx_dim),
            nn.ReLU(inplace=True),
            nn.Linear(edge_ctx_dim, 1),
        )

        self.conv1 = GCNConv(feat_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, num_classes)
        self.dropout = dropout

    def build_edge_weights(self, dense_ctx_map, batch_size: int, device):
        """Estimate adaptive edge weights for medium-only graphs."""
        _, channels, height, width = dense_ctx_map.shape
        assert height == 8 and width == 8

        node_ctx = F.avg_pool2d(dense_ctx_map, kernel_size=2, stride=2)
        node_ctx = node_ctx.permute(0, 2, 3, 1).contiguous().view(batch_size, 16, channels)

        edge_blocks = []
        edge_weight_blocks = []

        base_edge = self.templates["medium_edge_template"].to(device)

        for batch_id in range(batch_size):
            edge_blocks.append(base_edge + batch_id * 16)

            src = base_edge[0]
            dst = base_edge[1]

            c_src = node_ctx[batch_id, src, :]
            c_dst = node_ctx[batch_id, dst, :]
            c_diff = torch.abs(c_src - c_dst)

            edge_feat = torch.cat([c_src, c_dst, c_diff], dim=1)
            edge_weight = torch.sigmoid(self.edge_mlp(edge_feat)).view(-1)
            edge_weight_blocks.append(edge_weight)

        edge_index = torch.cat(edge_blocks, dim=1)
        edge_weight = torch.cat(edge_weight_blocks, dim=0)

        return edge_index, edge_weight

    def forward(self, data, sal, full_img):
        device = data.x.device

        x_updated = self.csa(
            x=data.x,
            pos=data.pos,
            batch_vec=data.batch,
            sal=sal,
        )

        medium_graph = build_medium_only_graph_batched(data, x_updated, self.templates)

        batch_size = int(data.y.view(-1).numel())

        if full_img.dim() == 3:
            full_img = full_img.view(batch_size, 3, CFG.img_size, CFG.img_size)
        elif full_img.dim() == 4 and full_img.size(1) != 3:
            full_img = full_img.view(batch_size, 3, CFG.img_size, CFG.img_size)

        with autocast_context(device):
            fmap = self.backbone(full_img)
            fmap = F.relu(fmap)

        dense_ctx_map = self.edge_branch(fmap)
        edge_index, edge_weight = self.build_edge_weights(
            dense_ctx_map=dense_ctx_map,
            batch_size=batch_size,
            device=medium_graph.x.device,
        )

        x = F.relu(self.conv1(medium_graph.x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))

        pooled = global_mean_pool(x, medium_graph.batch)
        pooled = F.relu(self.lin1(pooled))
        pooled = F.dropout(pooled, p=self.dropout, training=self.training)

        return self.lin2(pooled)
