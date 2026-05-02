"""engine/setup.py — Funciones de inicialización y setup.

Extraído de train.py para modularizar la construcción de componentes.
"""

import os
import copy
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms as T

from models.moco import build_index, MoCoDataset, ModelBase, MoCoQueue


def resolve_kaggle_paths(paths_config, rank=0):
    """Auto-descubre la ubicación real del dataset en Kaggle."""
    logger = logging.getLogger("AranduSSL")
    dataset_root = paths_config.get("dataset_root", "")

    if os.path.isdir(dataset_root):
        return paths_config
    if not os.path.isdir("/kaggle/input"):
        return paths_config

    dataset_folder_name = os.path.basename(dataset_root)
    if not dataset_folder_name:
        return paths_config

    found = None
    for dirpath, dirnames, _ in os.walk("/kaggle/input"):
        if dataset_folder_name in dirnames:
            found = os.path.join(dirpath, dataset_folder_name)
            break

    if found is None:
        path_parts = dataset_root.rstrip("/").split("/")
        if len(path_parts) >= 2:
            dataset_slug = path_parts[-2]
            for dirpath, dirnames, _ in os.walk("/kaggle/input"):
                if dataset_slug in dirnames:
                    candidate = os.path.join(dirpath, dataset_slug, dataset_folder_name)
                    if os.path.isdir(candidate):
                        found = candidate
                        break

    if found is None:
        if rank == 0:
            logger.warning(f"⚠️ No se encontró '{dataset_folder_name}' bajo /kaggle/input/. "
                           f"Path original: {dataset_root}")
        return paths_config

    old_root = dataset_root
    new_root = found
    if rank == 0:
        logger.info(f"🔍 Auto-discovery: dataset encontrado en {new_root}")
        logger.info(f"   (path original del config: {old_root})")

    patched = {}
    for key, value in paths_config.items():
        if isinstance(value, str) and old_root in value:
            patched[key] = value.replace(old_root, new_root)
        else:
            patched[key] = value

    return patched


def make_eval_subset_loader(eval_ds, subset_size: int, num_workers: int) -> DataLoader:
    """Crea un DataLoader con un subconjunto aleatorio del dataset de evaluación.

    Cada llamada genera un nuevo subconjunto independiente, permitiendo
    rerandomizar periódicamente y obtener estimaciones no sesgadas de KNN.
    """
    indices = torch.randperm(len(eval_ds))[:min(subset_size, len(eval_ds))].tolist()
    return DataLoader(
        Subset(eval_ds, indices), batch_size=128,
        num_workers=num_workers, pin_memory=True
    )


def build_dataloaders(CONFIG, is_distributed, rank):
    paths = build_index(CONFIG["paths"]["dataset_root"], rank, CONFIG["paths"]["index_cache_path"])
    dataset = MoCoDataset(paths, moco_config=CONFIG["moco"])
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if is_distributed else None

    n_workers = CONFIG["training"]["num_workers"]
    train_loader = DataLoader(
        dataset, batch_size=CONFIG["training"]["batch_size"], shuffle=(sampler is None),
        sampler=sampler, num_workers=n_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=(n_workers > 0),
        prefetch_factor=2 if n_workers > 0 else None
    )

    eval_transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    eval_ds = ImageFolder(CONFIG["paths"]["eval_train_root"], transform=eval_transform)
    val_ds = ImageFolder(CONFIG["paths"]["eval_val_root"], transform=eval_transform)

    eval_workers = min(2, n_workers)
    # C3 FIX: Usar make_eval_subset_loader() para permitir rerandomización periódica.
    eval_train_loader = make_eval_subset_loader(eval_ds, CONFIG["eval"]["subset_size"], eval_workers)
    eval_val_loader = DataLoader(val_ds, batch_size=128, num_workers=eval_workers, pin_memory=True)

    return train_loader, eval_train_loader, eval_val_loader, eval_ds, val_ds


def build_model(CONFIG, is_distributed, device, rank, local_rank):
    logger = logging.getLogger("AranduSSL")
    model_base_raw = ModelBase(
        dim=CONFIG["moco"]["dim"],
        predictor_hidden_dim=CONFIG["moco"].get("predictor_hidden_dim", 4096)
    ).to(device, memory_format=torch.channels_last)

    if is_distributed:
        model_base_raw = nn.SyncBatchNorm.convert_sync_batchnorm(model_base_raw)

    model_q = copy.deepcopy(model_base_raw)
    model_k = copy.deepcopy(model_base_raw).to(device, memory_format=torch.channels_last)

    is_compiled = False
    if hasattr(torch, "compile"):
        try:
            model_q = torch.compile(model_q, dynamic=True)
            is_compiled = True
        except Exception as e:
            if rank == 0: logger.warning(f"No se pudo compilar el modelo: {e}")

    model_k.eval()
    for p in model_k.parameters(): p.requires_grad = False
    for m in model_k.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
            m.track_running_stats = False

    if is_distributed: model_q = nn.parallel.DistributedDataParallel(model_q, device_ids=[local_rank])
    queue = MoCoQueue(dim=CONFIG["moco"]["dim"], K=CONFIG["moco"]["queue"]).to(device)

    return model_q, model_k, queue, is_compiled
