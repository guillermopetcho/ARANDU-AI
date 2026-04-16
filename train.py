import os
import math
import copy
import logging
import csv
import yaml
import json
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torchvision import models

from models.moco import ModelBase, MoCoQueue, build_index, MoCoDataset
from engine.trainer import MoCoTrainer
from evaluation.knn import extract_features_fast, fast_knn
from evaluation.linear_probe import run_linear_probe

def get_model_module(model, is_distributed):
    return model.module if is_distributed else model

def main():
    torch.set_float32_matmul_precision('high')

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    
    if is_distributed:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("config/moco.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)

    eff_batch = CONFIG["training"]["batch_size"] * CONFIG["training"]["grad_accum_steps"] * world_size
    lr = min(CONFIG["training"]["lr_base"] * (eff_batch / 256.0), 0.3)
    
    logger = logging.getLogger("AranduSSL")
    logger.setLevel(logging.INFO)
    if rank == 0 and not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(ch)

    if rank == 0: logger.info(f"Iniciando. EffBatch: {eff_batch}, LR: {lr:.6f}")

    seed = CONFIG["training"]["seed"] + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    cudnn.benchmark = True

    paths = build_index(CONFIG["paths"]["dataset_root"], rank, CONFIG["paths"]["index_cache_path"])
    dataset = MoCoDataset(paths)
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if is_distributed else None
    
    train_loader = DataLoader(
        dataset, batch_size=CONFIG["training"]["batch_size"], shuffle=(sampler is None),
        sampler=sampler, num_workers=CONFIG["training"]["num_workers"], 
        pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=2
    )

    eval_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    eval_ds = ImageFolder(CONFIG["paths"]["eval_train_root"], transform=eval_transform)
    val_ds = ImageFolder(CONFIG["paths"]["eval_val_root"], transform=eval_transform)
    
    # 🔥 FIX: Eval Dataloaders optimizados
    indices = torch.randperm(len(eval_ds))[:min(CONFIG["eval"]["subset_size"], len(eval_ds))].tolist()
    eval_train_loader = DataLoader(Subset(eval_ds, indices), batch_size=128, num_workers=2, pin_memory=True)
    eval_val_loader = DataLoader(val_ds, batch_size=128, num_workers=2, pin_memory=True)

    model_base = ModelBase(dim=CONFIG["moco"]["dim"]).to(device, memory_format=torch.channels_last)
    if is_distributed: model_base = nn.SyncBatchNorm.convert_sync_batchnorm(model_base)
    
    model_q = copy.deepcopy(model_base)
    model_k = copy.deepcopy(model_base).to(device, memory_format=torch.channels_last)
    model_k.eval()
    for p in model_k.parameters(): p.requires_grad = False
    for m in model_k.modules():
        if isinstance(m, nn.BatchNorm2d): m.eval(); m.track_running_stats = False
            
    if is_distributed: model_q = nn.parallel.DistributedDataParallel(model_q, device_ids=[local_rank])
    queue = MoCoQueue(dim=CONFIG["moco"]["dim"], K=CONFIG["moco"]["queue"]).to(device)

    optimizer = torch.optim.SGD(model_q.parameters(), lr=lr, momentum=0.9, weight_decay=float(CONFIG["training"]["weight_decay"]))
    scaler = GradScaler(enabled=CONFIG["training"]["use_amp"])
    
    total_steps = CONFIG["training"]["epochs"] * math.ceil(len(train_loader) / CONFIG["training"]["grad_accum_steps"])
    warmup_steps = max(1, CONFIG["training"]["warmup_epochs"] * math.ceil(len(train_loader) / CONFIG["training"]["grad_accum_steps"]))
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
        ], milestones=[warmup_steps]
    )

    trainer = MoCoTrainer(model_q, model_k, queue, optimizer, scheduler, scaler, CONFIG, device, is_distributed)

    start_epoch, global_step = 0, 0
    best_acc, patience = 0.0, 0
    log_buffer = []
    stop_signal = torch.tensor(0, device=device)
    
    ckpt_path = CONFIG["paths"].get("checkpoint_path", "")
    if ckpt_path and os.path.exists(ckpt_path):
        if rank == 0: logger.info(f"🔄 Reanudando entrenamiento desde {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_q.load_state_dict(ckpt["model_q"])
        model_k.load_state_dict(ckpt["model_k"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        queue.load_state_dict(ckpt["queue"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_acc = ckpt.get("best_acc", 0.0)
    log_file = CONFIG["paths"]["metrics_path"].replace('.json', '_log.csv')

    if rank == 0 and not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","loss","lr","knn_acc","pos","neg","margin","align","unif","std","gn","tput"])

    for epoch in range(start_epoch, CONFIG["training"]["epochs"]):
        if is_distributed: train_loader.sampler.set_epoch(epoch)
        
        metrics, global_step = trainer.train_epoch(train_loader, epoch, global_step, total_steps, rank)
        
        if rank == 0:
            curr_acc = -1
            if (epoch + 1) % 5 == 0:
                # Sincrónico rápido
                eval_model = get_model_module(model_q, is_distributed)
                X_t, y_t = extract_features_fast(eval_model, eval_train_loader, device)
                X_v, y_v = extract_features_fast(eval_model, eval_val_loader, device)
                curr_acc = fast_knn(X_t, y_t, X_v, y_v, k=CONFIG["eval"]["knn_k"])
                
                logger.info(f"KNN ACC: {curr_acc:.4f}")
                
            ckpt = {
                "model_q": model_q.state_dict(), "model_k": model_k.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "queue": queue.state_dict(),
                "epoch": epoch, "global_step": global_step, "best_acc": best_acc
            }
            torch.save(ckpt, CONFIG["paths"]["checkpoint_path"])
            
            if (epoch + 1) % 5 == 0:
                if curr_acc > best_acc:
                    best_acc = curr_acc
                    patience = 0
                    torch.save(ckpt, CONFIG["paths"]["best_checkpoint_path"])
                    logger.info("🏆 Best model guardado")
                else:
                    patience += 1
                    if patience >= CONFIG["training"]["early_stopping_patience"]:
                        logger.warning("⛔ Early Stopping")
                        stop_signal.fill_(1)

            log_buffer.append([
                epoch+1, metrics['loss'], optimizer.param_groups[0]['lr'], curr_acc,
                metrics['pos'], metrics['neg'], metrics['margin'], metrics['align'], 
                metrics['unif'], metrics['std'], metrics['gn'], metrics['tput']
            ])
            
            if (epoch + 1) % 5 == 0:
                with open(log_file, "a", newline="") as f: csv.writer(f).writerows(log_buffer)
                log_buffer.clear()
                
                
            logger.info(f"Ep {epoch+1} | Loss {metrics['loss']:.3f} | U: {metrics['unif']:.2f} | LR: {optimizer.param_groups[0]['lr']:.4f}")

        if is_distributed:
            dist.broadcast(stop_signal, src=0)
        
        if stop_signal.item() == 1:
            break

    if is_distributed: dist.destroy_process_group()

    if rank == 0:
        if log_buffer:
            with open(log_file, "a", newline="") as f: csv.writer(f).writerows(log_buffer)
        
        logger.info("Iniciando Linear Probe...")
        best_ckpt = torch.load(CONFIG["paths"]["best_checkpoint_path"], map_location=device)
        clean_enc = {}
        for k, v in best_ckpt["model_q"].items():
            if k.startswith("module.encoder."):
                clean_enc[k.replace("module.encoder.", "")] = v
            elif k.startswith("encoder."):
                clean_enc[k.replace("encoder.", "")] = v
        
        eval_enc = models.resnet50(weights=None); eval_enc.fc = nn.Identity()
        eval_enc.load_state_dict(clean_enc, strict=False)
        
        num_classes = len(eval_ds.classes)
        head, acc, f1 = run_linear_probe(eval_enc, eval_ds, val_ds, num_classes, CONFIG, device)
        
        torch.save(head, CONFIG["paths"]["encoder_export_path"].replace(".pth", "_head.pth"))
        torch.save(clean_enc, CONFIG["paths"]["encoder_export_path"])
        with open(CONFIG["paths"]["metrics_path"], "w") as f: json.dump({"acc": acc, "f1": f1}, f, indent=4)
        logger.info("✅ Listo.")

if __name__ == "__main__":
    main()