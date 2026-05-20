from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Config:
    # Dataset selection:
    # 1 = Paediatric Chest X-ray Pneumonia dataset
    # 2 = Chest X-ray PA dataset
    # 3 = SARS-CoV-2 CT-scan dataset
    # 4 = COVID-19 Radiography Database
    # 5 = Chest X-ray (COVID-19 & Pneumonia) dataset
    dataset_id: int = 1

    # If auto_prepare_dataset=True, main.py will download and organise the selected dataset.
    # If False, main.py will use root_dir directly.
    auto_prepare_dataset: bool = True
    download_dataset: bool = True

    # Used only when auto_prepare_dataset=False.
    root_dir: str = "/content/chest-xray-pneumonia/chest_xray/lung"

    cache_full_dir: str = "/content/ms_rawcrop_densenet_graphs"
    cache_sal_dir: str = "/content/ms_rawcrop_densenet_sal"

    seed: int = 42

    img_size: int = 256
    crop_resize: int = 224
    scales: List[Dict] = None

    patch_encode_batch: int = 84
    feature_dim: int = 1920

    epochs_baseline: int = 5
    batch_size_baseline: int = 32
    lr_base: float = 1e-3
    weight_decay: float = 1e-4

    k_folds: int = 5
    epochs_final: int = 100
    batch_size_final: int = 64
    lr_final: float = 1e-3

    num_workers: int = 4
    pin_memory: bool = True

    perturb_chunk: int = 32

    hidden_dim: int = 128
    dropout: float = 0.25
    attn_dim: int = 128
    edge_ctx_dim: int = 8

    def __post_init__(self):
        if self.scales is None:
            self.scales = [
                {"name": "P128", "patch": 128},
                {"name": "P64", "patch": 64},
                {"name": "P32", "patch": 32},
            ]

    @property
    def persistent_workers(self) -> bool:
        return self.num_workers > 0


CFG = Config()
