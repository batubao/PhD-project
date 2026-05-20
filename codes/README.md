This folder contains source code for the PhD project.
Datasets, cached graph files, saliency files, and model checkpoints are not stored in this repository due to file size

The code is intentionally kept compact for the thesis GitHub documentation. It is divided into a small number of files rather than many subfolders.

## Repository structure

```text
smpg_simple_repository/
├── README.md
├── requirements.txt
├── config.py
├── prepare_dataset.py
├── graph_utils.py
├── models.py
├── train.py
└── main.py
```

## Files

- `config.py`: experiment settings, dataset selection, training parameters, cache paths, and patch settings.
- `prepare_dataset.py`: Kaggle download links and dataset-specific folder organisation for the five datasets.
- `graph_utils.py`: multiscale patch metadata, graph construction, graph caching, image transforms, and perturbation saliency.
- `models.py`: DenseNet201 feature extractor, baseline GCN, cross-scale attention, dense-context edge enhancement, and final SMPG model.
- `train.py`: training, evaluation, and 5-fold cross-validation.
- `main.py`: main entry point for running the full pipeline.

## Installation

```bash
pip install -r requirements.txt
```

For Google Colab, you may also run:

```bash
pip install opendatasets torch-geometric torchvision
```

## Dataset selection

The five datasets are configured in `prepare_dataset.py`.

| ID | Dataset name | Kaggle URL | Final root used by the pipeline |
|---:|---|---|---|
| 1 | Paediatric Chest X-ray Pneumonia dataset | `paultimothymooney/chest-xray-pneumonia` | `/content/chest-xray-pneumonia/chest_xray/lung` |
| 2 | Chest X-ray PA dataset | `amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset` | `/content/covid19-pneumonia-normal-chest-xray-pa-dataset` |
| 3 | SARS-CoV-2 CT-scan dataset | `plameneduardo/sarscov2-ctscan-dataset` | `/content/sarscov2-ctscan-dataset` |
| 4 | COVID-19 Radiography Database | `tawsifurrahman/covid19-radiography-database` | `/content/covid19-radiography-database/dataset_clean` |
| 5 | Chest X-ray (COVID-19 & Pneumonia) dataset | `prashant268/chest-xray-covid19-pneumonia/data` | `/content/chest-xray-covid19-pneumonia/Data/lung` |

For Dataset 5, the raw root is `/content/chest-xray-covid19-pneumonia/Data`, but the code merges `train` and `test` into `/content/chest-xray-covid19-pneumonia/Data/lung`. The final class-folder root used for training is therefore the `lung` folder.

## How to run

Run Dataset 1:

```bash
python main.py --dataset 1
```

Run Dataset 2:

```bash
python main.py --dataset 2
```

Run Dataset 3:

```bash
python main.py --dataset 3
```

Run Dataset 4:

```bash
python main.py --dataset 4
```

Run Dataset 5:

```bash
python main.py --dataset 5
```

If the dataset has already been downloaded and organised, skip download and preparation:

```bash
python main.py --dataset 1 --no-download --no-prepare
```

If you want to manually provide a root directory that already contains one folder per class:

```bash
python main.py --root-dir /content/my_dataset_root
```

## Expected dataset format

After preparation, the dataset root must contain one folder per class:

```text
root_dir/
├── class_1/
│   ├── image_001.png
│   └── image_002.png
├── class_2/
│   ├── image_003.png
│   └── image_004.png
└── class_3/
    ├── image_005.png
    └── image_006.png
```

## Pipeline overview

1. Download and organise the selected dataset.
2. Resize each image to 256 × 256 pixels.
3. Divide each image into multiscale patches:
   - 128 × 128 coarse patches
   - 64 × 64 medium patches
   - 32 × 32 fine patches
4. Extract patch-level features using DenseNet201.
5. Construct a multiscale patch graph with intra-scale and inter-scale edges.
6. Estimate perturbation-based node saliency using a baseline GCN.
7. Apply saliency-guided cross-scale attention.
8. Collapse the graph into a refined medium-scale graph.
9. Enhance medium-scale edges using dense contextual features.
10. Perform graph classification using GCN.
11. Evaluate the model using 5-fold cross-validation.

## Note

This repository is designed as a clear and readable thesis implementation. For large experiments, it is recommended to run the code in Google Colab or another GPU environment.

