"""engine/loop.py — Lógica del bucle de entrenamiento principal.

Extraído de train.py para simplificar el flujo principal y delegar
la lógica de epochs, evaluación, y rollback.
"""

import os
import csv
import logging
import warnings
import torch
import torch.distributed as dist

from engine.checkpoint import (
    build_checkpoint_dict, save_checkpoint, load_weights_for_rollback
)
from engine.setup import make_eval_subset_loader
from evaluation.knn import extract_features_fast, fast_knn
from engine.controller import Action
from utils.metrics import get_module_stats


def get_model_module(model, is_distributed):
    return model.module if is_distributed else model


def handle_evaluation(
    epoch, model_q, eval_train_loader, eval_val_loader,
    device, is_distributed, CONFIG, logger, controller, metrics
):
    """Ejecuta la evaluación KNN y el análisis del espacio latente."""
    eval_model = get_model_module(model_q, is_distributed)
    eval_model.eval()
    X_t, y_t = extract_features_fast(eval_model, eval_train_loader, device)
    X_v, y_v = extract_features_fast(eval_model, eval_val_loader, device)
    eval_model.train()
    
    curr_acc = fast_knn(X_t, y_t, X_v, y_v, k=CONFIG["eval"]["knn_k"])
    logger.info(f"KNN ACC: {curr_acc:.4f}")

    SVD_MAX_SAMPLES = 2000
    X_v_t = torch.tensor(X_v)
    if len(X_v_t) > SVD_MAX_SAMPLES:
        perm = torch.randperm(len(X_v_t))[:SVD_MAX_SAMPLES]
        X_svd = X_v_t[perm]
    else:
        X_svd = X_v_t

    mu = X_svd.mean(dim=0)
    X_centered = X_svd - mu.unsqueeze(0)
    s = torch.linalg.svdvals(X_centered)
    p = (s**2) / ((s**2).sum() + 1e-8)
    p = torch.clamp(p, min=1e-6)
    p = p / p.sum()
    eff_rank = torch.exp(-(p * torch.log(p)).sum()).item()

    metrics['mu'] = mu
    metrics['eff_rank'] = eff_rank

    return controller.step_epoch(epoch, curr_acc, metrics), curr_acc


def handle_rollback(
    CONFIG, rank, use_wandb, global_step, model_q, model_k,
    optimizer, scaler, queue, is_compiled, is_distributed,
    warmup_steps, total_steps, final_lr_ratio, build_scheduler, trainer, controller, logger
):
    """Ejecuta la lógica de rollback a un checkpoint previo."""
    if rank == 0: logger.info("🔄 Iniciando proceso de Rollback...")
    if rank == 0 and use_wandb:
        try:
            import wandb
            wandb.log({"event/rollback": 1}, step=global_step)
        except Exception:
            pass

    rollback_ckpt_path = CONFIG["paths"]["best_checkpoint_path"]
    if not os.path.exists(rollback_ckpt_path):
        rollback_ckpt_path = CONFIG["paths"].get("checkpoint_path", "")
    if not rollback_ckpt_path or not os.path.exists(rollback_ckpt_path):
        if rank == 0: logger.warning("⚠️ Rollback solicitado pero no hay checkpoint. Continuando.")
        return global_step

    global_step = load_weights_for_rollback(
        path=rollback_ckpt_path,
        model_q=model_q, model_k=model_k,
        optimizer=optimizer, scaler=scaler, queue=queue,
        is_compiled=is_compiled, is_distributed=is_distributed,
    )

    for param_group in optimizer.param_groups:
        new_lr = param_group['initial_lr'] * 0.5
        param_group['initial_lr'] = new_lr
        param_group['lr'] = new_lr

    scheduler = build_scheduler(optimizer, warmup_steps, total_steps, final_lr_ratio=final_lr_ratio)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(global_step):
            scheduler.step()

    trainer.scheduler = scheduler
    controller.warmup_aborted = True
    if rank == 0: logger.info("✅ Rollback completado. Iniciando fase de Decaimiento Cosenoidal.")
    return global_step
