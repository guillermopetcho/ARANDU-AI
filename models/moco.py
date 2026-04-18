import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms as T
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
from pathlib import Path
import logging

from utils.distributed import concat_all_gather

class RandomRotate90:
    def __init__(self):
        self.angles = [0, 90, 180, 270]
    def __call__(self, img):
        return T.functional.rotate(img, random.choice(self.angles))

def get_transforms():
    t_q = T.Compose([
        T.RandomResizedCrop(224, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        RandomRotate90(),
        T.RandomSolarize(threshold=128, p=0.2), 
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    t_k = T.Compose([
        T.RandomResizedCrop(224, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        RandomRotate90(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return t_q, t_k

class MoCoDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.t_q, self.t_k = get_transforms()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        for _ in range(10): 
            try:
                img = Image.open(self.paths[idx]).convert("RGB")
                return self.t_q(img), self.t_k(img)
            except Exception:
                idx = random.randint(0, len(self.paths) - 1)
        
        dummy = Image.new("RGB", (224, 224))
        return self.t_q(dummy), self.t_k(dummy)

def build_index(root, rank, cache_path):
    if os.path.exists(cache_path):
        return np.load(cache_path).tolist()
    files = sorted([str(f) for ext in ["*.jpg", "*.png", "*.jpeg"] for f in Path(root).rglob(ext)])
    if len(files) == 0: raise RuntimeError(f"No imágenes en {root}")
    if rank == 0: np.save(cache_path, files)
    return files

class ModelBase(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.encoder.fc = nn.Identity()
        
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

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        # Force float32 for L2 norm to prevent FP16 overflow (inf), which causes NaNs.
        return F.normalize(z.float(), dim=1).to(z.dtype)

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