import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
import json
import logging

def run_linear_probe(encoder, train_ds, val_ds, num_classes, config, device):
    logger = logging.getLogger("LinearProbe")
    logger.info("Iniciando Linear Probing...")

    encoder = encoder.to(device).eval().to(memory_format=torch.channels_last)
    for p in encoder.parameters(): p.requires_grad = False

    classifier = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, num_classes)).to(device)
    param_groups = [{'params': [p], 'weight_decay': 0.0 if p.ndim <= 1 or n.endswith(".bias") else 1e-4} 
                    for n, p in classifier.named_parameters() if p.requires_grad]
            
    optimizer = torch.optim.AdamW(param_groups, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    n_workers = config['training']['num_workers']
    # B11 FIX: persistent_workers solo es válido cuando num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                              num_workers=n_workers, pin_memory=True,
                              persistent_workers=(n_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                            num_workers=n_workers, pin_memory=True)
    
    for epoch in range(25):
        classifier.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True, memory_format=torch.channels_last), y.to(device, non_blocking=True)
            with torch.no_grad(): feats = F.normalize(encoder(x), dim=1).detach()
            loss = criterion(classifier(feats), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
            
    classifier.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            preds = torch.argmax(classifier(F.normalize(encoder(x), dim=1)), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    logger.info(f"Finalizado -> ACC: {acc:.4f} | F1: {f1:.4f}")
    return classifier.state_dict(), acc, f1