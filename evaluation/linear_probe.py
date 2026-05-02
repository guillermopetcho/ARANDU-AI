import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
import json
import logging


def run_linear_probe(encoder, train_ds, val_ds, num_classes, config, device):
    """Entrena una sonda lineal sobre las representaciones del projector.

    NOTA DE DISEÑO: Esta función espera un `encoder` de tipo `ModelBase` (o compatible)
    que acepte `forward(x, use_predictor=False)` y retorne un vector de 256-dim.
    Esto mantiene COHERENCIA con el evaluador KNN, que también evalúa sobre 256-dim.
    Si se quisiera una Linear Probe sobre el backbone puro (transfer learning clásico,
    2048-dim), se debe pasar `models.resnet50` con `fc=Identity()` como `encoder`
    y ajustar `proj_dim=2048` en el clasificador de abajo.
    """
    logger = logging.getLogger("LinearProbe")
    logger.info("Iniciando Linear Probe...")

    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Extraer una muestra de features para inferir la dimensión de salida automáticamente.
    # Esto hace la función robusta a cambios de `dim` en moco.yaml sin hardcodear 256.
    with torch.no_grad():
        sample_x, _ = train_ds[0]
        sample_out = encoder(sample_x.unsqueeze(0).to(device), use_predictor=False)
        proj_dim = sample_out.shape[-1]
    logger.info(f"Linear Probe: dim de features = {proj_dim}")

    classifier = nn.Sequential(
        nn.LayerNorm(proj_dim),
        nn.Linear(proj_dim, num_classes)
    ).to(device)
    param_groups = [
        {'params': [p], 'weight_decay': 0.0 if p.ndim <= 1 or n.endswith(".bias") else 1e-4}
        for n, p in classifier.named_parameters() if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=1e-3)
    epochs = config.get('eval', {}).get('linear_probe_epochs', 25)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    n_workers = config['training']['num_workers']
    # B11 FIX: persistent_workers solo es válido cuando num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                              num_workers=n_workers, pin_memory=True,
                              persistent_workers=(n_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                            num_workers=n_workers, pin_memory=True)

    device_type = device.type if hasattr(device, 'type') else str(device).split(':')[0]
    use_amp = config.get('training', {}).get('use_amp', False)
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0
        
        # Añadir tqdm para ver el progreso del batch
        pbar = tqdm(train_loader, desc=f"Linear Probe Ep {epoch+1}/{epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                with torch.amp.autocast(device_type, enabled=use_amp):
                    feats = F.normalize(encoder(x, use_predictor=False), dim=1)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type, enabled=use_amp):
                out = classifier(feats)
                loss = criterion(out, y)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        logger.info(f"Linear Probe Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    classifier.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            feats = F.normalize(encoder(x, use_predictor=False), dim=1)
            preds = torch.argmax(classifier(feats), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    logger.info(f"Finalizado -> ACC: {acc:.4f} | F1: {f1:.4f}")
    return classifier.state_dict(), acc, f1