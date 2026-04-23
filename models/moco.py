import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import models, transforms as T
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
from pathlib import Path
import logging

from utils.distributed import concat_all_gather

# Eliminado RandomRotate90 para evitar transformaciones físicamente imposibles en cultivos.

def get_transforms():
    """Devuelve transformaciones asimétricas para las 2 vistas globales (224x224).
    La asimetría (MoCo v3 / BYOL style) ayuda a evitar el colapso y mejora el aprendizaje.
    """
    # Vista 1: Más fuerte en color, con Blur moderado, sin Solarize.
    t_q = T.Compose([
        T.RandomResizedCrop(224, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=15), # Simula viento/ángulo cámara real
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Vista 2: Con Solarize probabilístico y Blur más fuerte.
    t_k = T.Compose([
        T.RandomResizedCrop(224, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=15),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomGrayscale(p=0.2),
        T.RandomSolarize(threshold=128, p=0.2), # Asimetría: Solarize solo aquí
        T.GaussianBlur(kernel_size=9, sigma=(0.5, 2.0)), # Blur base más alto
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return t_q, t_k

def get_local_transforms(n_crops=4, scale=(0.05, 0.4), size=96):
    """
    Devuelve una lista de N transformaciones para vistas locales (Multi-Crop DINO-style).
    Las vistas locales son recortes pequeños (96x96 por defecto) de baja escala
    que fuerzan al modelo a aprender invarianzas finas del dominio (ej. manchas en hojas).
    """
    return [
        T.Compose([
            T.RandomResizedCrop(size, scale=scale),
            T.RandomHorizontalFlip(),
            T.RandomSolarize(threshold=128, p=0.1),
            T.ColorJitter(0.4, 0.4, 0.2, 0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        for _ in range(n_crops)
    ]

class MoCoDataset(Dataset):
    def __init__(self, paths, moco_config=None):
        self.paths = paths
        self.t_q, self.t_k = get_transforms()
        
        # Multi-Crop: Configuración de vistas locales
        if moco_config is not None:
            n_local = moco_config.get('num_local_crops', 0)
            scale_min = moco_config.get('local_crop_scale_min', 0.05)
            scale_max = moco_config.get('local_crop_scale_max', 0.4)
            size = moco_config.get('local_crop_size', 96)
        else:
            n_local = 0
            scale_min, scale_max, size = 0.05, 0.4, 96
        
        self.local_transforms = get_local_transforms(n_local, (scale_min, scale_max), size) if n_local > 0 else []
        self.load_errors = 0
        self.logger = logging.getLogger("AranduSSL")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # M4 FIX: Límite de reintentos para evitar loop infinito si muchas imágenes son corruptas
        max_retries = min(100, len(self.paths))
        for attempt in range(max_retries):
            try:
                img = Image.open(self.paths[idx]).convert("RGB")
                v_q = self.t_q(img)
                v_k = self.t_k(img)
                
                if self.local_transforms:
                    # Apilar N vistas locales: [N_local, C, H, W]
                    locals_ = torch.stack([t(img) for t in self.local_transforms])
                else:
                    # Placeholder vacío para mantener collation consistente [0, C, H, W]
                    locals_ = torch.empty(0, *v_q.shape)
                
                return v_q, v_k, locals_
            except Exception as e:
                # B12 FIX: Contabilizar errores de carga para monitoreo
                self.load_errors += 1
                if len(self.paths) == 0:
                    raise RuntimeError("MoCoDataset está vacío, no hay imágenes disponibles.")
                idx = random.randint(0, len(self.paths) - 1)
        raise RuntimeError(f"MoCoDataset: {max_retries} imágenes consecutivas fallaron al cargarse. Último error: {e}")

def build_index(root, rank, cache_path):
    """Construye o carga el índice de imágenes del dataset con sincronización DDP.
    
    C2 FIX: Solo el Rank 0 decide si hay que reconstruir el índice.
    Se usa una barrera para asegurar que los Ranks >= 1 no intenten leer
    un archivo parcial mientras Rank 0 escribe.
    """
    is_dist = dist.is_available() and dist.is_initialized()
    
    # 1. Rank 0 verifica y construye si es necesario
    if rank == 0:
        if not os.path.exists(cache_path):
            files = sorted([str(f) for ext in ["*.jpg", "*.png", "*.jpeg"] for f in Path(root).rglob(ext)])
            if len(files) == 0:
                raise RuntimeError(f"No se encontraron imágenes en {root}")
            np.save(cache_path, files)
            logging.getLogger("AranduSSL").info(f"📁 Índice creado con {len(files)} imágenes.")
    
    # 2. Barrera crítica: todos esperan a que Rank 0 termine de escribir en disco
    if is_dist:
        dist.barrier()
    
    # 3. Todos cargan el mismo archivo (ahora garantizado que existe y está completo)
    if not os.path.exists(cache_path):
        raise RuntimeError(f"Fallo crítico: El cache {cache_path} no existe tras la barrera DDP.")
        
    return np.load(cache_path, allow_pickle=True).tolist()

class ModelBase(nn.Module):
    """
    Backbone ResNet50 + Projector MLP 3-capas + Predictor MLP (MoCo v3).
    
    Durante entrenamiento:
      - Queries: forward(x, use_predictor=True)  → predictor(projector(encoder(x)))
      - Keys:    forward(x, use_predictor=False) → projector(encoder(x))
    Durante evaluación/export:
      - forward(x) → projector(encoder(x))  [use_predictor=False por defecto]
    """
    def __init__(self, dim=256, predictor_hidden_dim=4096):
        super().__init__()
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.encoder.fc = nn.Identity()
        
        # Projector: 2048 → 2048 → 2048 → dim
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, dim),
            nn.BatchNorm1d(dim, affine=False)
        )
        
        # Predictor MLP (MoCo v3): dim → predictor_hidden_dim → dim
        # Solo se aplica al modelo Query durante el entrenamiento.
        # Separa la dinámica de aprendizaje del query vs el key, mejorando la estabilidad.
        self.predictor = nn.Sequential(
            nn.Linear(dim, predictor_hidden_dim),
            nn.BatchNorm1d(predictor_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(predictor_hidden_dim, dim)
        )

    def forward(self, x, use_predictor=False):
        h = self.encoder(x)
        z = self.projector(h)
        # Normalizar en float32 para evitar overflow de FP16
        z = F.normalize(z.float(), dim=1).to(z.dtype)
        
        if use_predictor:
            p = self.predictor(z)
            p = F.normalize(p.float(), dim=1).to(p.dtype)
            return p
        
        return z

class MoCoQueue(nn.Module):
    def __init__(self, dim=256, K=32768):
        super().__init__()
        self.K = K
        self.register_buffer("queue", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue_dequeue(self, keys, step=None):
        keys = concat_all_gather(keys.detach())
        keys = F.normalize(keys, dim=1)
        batch_size = keys.shape[0]
        
        if batch_size > self.K:
            keys = keys[:self.K]
            batch_size = self.K
            
        ptr = int(self.queue_ptr)
        end_ptr = ptr + batch_size
        
        if end_ptr <= self.K:
            self.queue[:, ptr:end_ptr].copy_(keys.T)
        else:
            first_part = self.K - ptr
            self.queue[:, ptr:].copy_(keys[:first_part].T)
            self.queue[:, :batch_size - first_part].copy_(keys[first_part:].T)
            
        self.queue_ptr[0] = (ptr + batch_size) % self.K
        
        if step is not None and step % 500 == 0:
            self.queue.copy_(F.normalize(self.queue, dim=0))