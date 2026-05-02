import os
import datetime
import math
import logging
import csv
import yaml
import json
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

# Silenciar warnings molestos de sympy/inductor durante torch.compile
logging.getLogger("torch.utils._sympy").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
from torch.amp import GradScaler

from engine.trainer import MoCoTrainer
from engine.scheduler import build_scheduler
from engine.controller import TrainingController, Action
# C1 FIX: Funciones de inicialización extraídas a engine/setup.py
from engine.setup import (
    resolve_kaggle_paths, make_eval_subset_loader, 
    build_dataloaders, build_model
)
# C1 FIX: Funciones de control extraídas a engine/loop.py
from engine.loop import handle_evaluation, handle_rollback
from engine.checkpoint import (
    get_latest_valid_checkpoint, build_checkpoint_dict, 
    save_checkpoint, load_checkpoint
)
from evaluation.linear_probe import run_linear_probe
from utils.metrics import get_module_stats
from models.moco import ModelBase


def main():
    torch.set_float32_matmul_precision('high')

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=7200),
            device_id=device
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(BASE_DIR, "config", "moco.yaml")

    with open(config_path, "r") as f:
        CONFIG = yaml.safe_load(f)

    if rank == 0:
        CONFIG["paths"] = resolve_kaggle_paths(CONFIG["paths"], rank)

    if is_distributed:
        broadcast_buf = [CONFIG["paths"]]
        dist.broadcast_object_list(broadcast_buf, src=0)
        CONFIG["paths"] = broadcast_buf[0]

    controller = TrainingController(CONFIG)
    CONFIG['_controller'] = controller

    eff_batch = CONFIG["training"]["batch_size"] * CONFIG["training"]["grad_accum_steps"] * world_size
    lr = min(CONFIG["training"]["lr_base"] * (eff_batch / 256.0), 0.15)

    logger = logging.getLogger("AranduSSL")
    logger.setLevel(logging.INFO)
    if rank == 0 and not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(ch)

    use_wandb = CONFIG.get("wandb", {}).get("enabled", False)
    if rank == 0 and use_wandb:
        try:
            import wandb
            if not os.environ.get("WANDB_API_KEY"):
                os.environ["WANDB_MODE"] = "offline"
            wandb_config = {k: v for k, v in CONFIG.items() if k != '_controller'}
            wandb.init(
                project=CONFIG.get("wandb", {}).get("project", "MoCo-ENCODER"),
                config=wandb_config,
                name=f"run_effbatch_{eff_batch}_lr_{lr:.4f}"
            )
        except ImportError:
            logger.warning("WandB no está instalado. Usando logger estándar.")
            use_wandb = False

    if rank == 0: logger.info(f"Iniciando. EffBatch: {eff_batch}, LR: {lr:.6f}")

    seed = CONFIG["training"]["seed"] + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

    # C1 FIX: Inicialización modularizada con engine/setup.py
    train_loader, eval_train_loader, eval_val_loader, eval_ds, val_ds = build_dataloaders(CONFIG, is_distributed, rank)
    model_q, model_k, queue, is_compiled = build_model(CONFIG, is_distributed, device, rank, local_rank)

    optimizer = torch.optim.SGD(model_q.parameters(), lr=lr, momentum=0.9, weight_decay=float(CONFIG["training"]["weight_decay"]))
    scaler = GradScaler(device.type, enabled=CONFIG["training"]["use_amp"])

    total_steps = CONFIG["training"]["epochs"] * math.ceil(len(train_loader) / CONFIG["training"]["grad_accum_steps"])
    warmup_steps = max(1, CONFIG["training"]["warmup_epochs"] * math.ceil(len(train_loader) / CONFIG["training"]["grad_accum_steps"]))
    final_lr_ratio = CONFIG["training"].get("final_lr_ratio", 0.0)

    scheduler = build_scheduler(optimizer, warmup_steps, total_steps, final_lr_ratio=final_lr_ratio)

    trainer = MoCoTrainer(model_q, model_k, queue, optimizer, scheduler, scaler, CONFIG, device, is_distributed)

    start_epoch, global_step = 0, 0
    log_buffer = []
    stop_signal = torch.tensor(0, device=device)
    sync_data = [None]

    ckpt_path = CONFIG["paths"].get("checkpoint_path", "")
    best_ckpt_path = CONFIG["paths"].get("best_checkpoint_path", "")
    is_exploitation = CONFIG.get("training", {}).get("exploitation_mode", False)

    # C1 FIX: Carga de checkpoint usando engine/checkpoint.py
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt_to_load = get_latest_valid_checkpoint(CONFIG["paths"])
    elif best_ckpt_path and os.path.exists(best_ckpt_path):
        ckpt_to_load = best_ckpt_path
    else:
        ckpt_to_load = get_latest_valid_checkpoint(CONFIG["paths"])

    if ckpt_to_load:
        if rank == 0:
            logger.info(f"🔄 Reanudando desde {ckpt_to_load} "
                        f"{'(MODO EXPLOTACIÓN)' if is_exploitation else ''}")
        # C2 FIX: load_checkpoint usa weights_only=True
        start_epoch, global_step, scheduler = load_checkpoint(
            path=ckpt_to_load,
            model_q=model_q, model_k=model_k,
            optimizer=optimizer, scaler=scaler,
            queue=queue, controller=controller,
            lr=lr, is_compiled=is_compiled, is_distributed=is_distributed,
            build_scheduler_fn=build_scheduler,
            warmup_steps=warmup_steps, total_steps=total_steps,
            final_lr_ratio=final_lr_ratio, trainer=trainer,
        )

    log_file = CONFIG["paths"]["metrics_path"].replace('.json', '_log.csv')
    proj_log_file = CONFIG["paths"]["metrics_path"].replace('.json', '_projector.csv')

    if rank == 0:
        if not os.path.exists(log_file):
            with open(log_file, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "loss", "lr", "knn_acc", "pos", "neg", "margin",
                                        "align", "unif", "psim", "nsim", "rnorm", "std", "gn",
                                        "tput", "data_err"])
        if not os.path.exists(proj_log_file):
            with open(proj_log_file, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "total_mean", "total_std", "total_norm"])

    EVAL_SUBSET_REFRESH_FREQ = 5
    eval_workers = min(2, CONFIG["training"]["num_workers"])

    for epoch in range(start_epoch, CONFIG["training"]["epochs"]):
        if is_distributed and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)

        # C3 FIX: Rerandomizar subconjunto de evaluación periódicamente
        if rank == 0 and epoch > 0 and epoch % EVAL_SUBSET_REFRESH_FREQ == 0:
            eval_train_loader = make_eval_subset_loader(
                eval_ds, CONFIG["eval"]["subset_size"], eval_workers
            )
            logger.info(f"🔀 Subconjunto KNN rerandomizado (epoch {epoch})")

        metrics, global_step = trainer.train_epoch(train_loader, epoch, global_step, total_steps, rank)

        if rank == 0:
            curr_acc = -1
            eval_freq = 1

            if (epoch + 1) % eval_freq == 0:
                # C1 FIX: Lógica de evaluación delegada a handle_evaluation en engine/loop.py
                action, curr_acc = handle_evaluation(
                    epoch, model_q, eval_train_loader, eval_val_loader,
                    device, is_distributed, CONFIG, logger, controller, metrics
                )

                if action == Action.EARLY_STOP:
                    stop_signal.fill_(1)
                elif action == Action.ROLLBACK:
                    stop_signal.fill_(2)

            ckpt_dict = build_checkpoint_dict(
                model_q, model_k, optimizer, scheduler, scaler, queue,
                epoch, global_step, controller
            )
            save_checkpoint(CONFIG["paths"]["checkpoint_path"], ckpt_dict)

            if (epoch + 1) % eval_freq == 0:
                if curr_acc >= controller.best_acc and curr_acc > 0:
                    save_checkpoint(CONFIG["paths"]["best_checkpoint_path"], ckpt_dict)
                    logger.info("🏆 Best model guardado")

                if getattr(controller, 'is_best_geom', False):
                    geom_ckpt_path = CONFIG["paths"]["checkpoint_path"].replace('.pth', '_best_geom.pth')
                    save_checkpoint(geom_ckpt_path, ckpt_dict)
                    logger.info("💎 Best Geometric model guardado")

            log_buffer.append([
                epoch+1, metrics['loss'], optimizer.param_groups[0]['lr'], curr_acc,
                metrics['pos'], metrics['neg'], metrics['margin'], metrics['align'],
                metrics['unif'], metrics['pos_sim'], metrics['neg_sim'],
                controller.history['ratio_norm'][-1] if controller.history['ratio_norm'] else 1.0,
                metrics['std'], metrics['gn'], metrics['tput'],
                metrics.get('data_err', 0)
            ])

            if (epoch + 1) % eval_freq == 0:
                with open(log_file, "a", newline="") as f: csv.writer(f).writerows(log_buffer)
                log_buffer.clear()

                model_to_save = model_q.module if is_distributed else model_q
                if hasattr(model_to_save, 'projector'):
                    p_stats = get_module_stats(model_to_save.projector)
                    p_stats['epoch'] = epoch + 1
                    with open(proj_log_file, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=sorted(p_stats.keys()))
                        if f.tell() == 0:
                            writer.writeheader()
                        writer.writerow(p_stats)

            logger.info(f"Ep {epoch+1} | Loss {metrics['loss']:.3f} | U: {metrics['unif']:.2f} | LR: {optimizer.param_groups[0]['lr']:.4f} | Err: {metrics.get('data_err', 0):.2f}%")

            if use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "train/loss": metrics['loss'],
                        "train/lr": optimizer.param_groups[0]['lr'],
                        "train/tau": controller.tau,
                        "train/momentum": controller.current_m,
                        "train/lr_scale": controller.lr_scale,
                        "train/unif": metrics['unif'],
                        "train/align": metrics['align'],
                        "train/pos_sim": metrics['pos_sim'],
                        "train/neg_sim": metrics['neg_sim'],
                        "train/ratio_norm": controller.history['ratio_norm'][-1] if controller.history['ratio_norm'] else 1.0,
                        "train/grad_norm": metrics['gn'],
                        "train/pos_loss": metrics['pos'],
                        "train/neg_loss": metrics['neg'],
                        "train/tput": metrics['tput'],
                        "train/data_error_rate": metrics.get('data_err', 0),
                        "pid/eU_ema": controller.eU_ema,
                        "pid/eD_ema": controller.eD_ema,
                        "pid/eR_ema": controller.eR_ema,
                        "pid/I_U": controller.I_U,
                        "pid/tau_rank_coef": controller.tau_rank_coef,
                        "pid/lr_step_factor": controller.lr_step_factor,
                        "eval/knn_acc": curr_acc if curr_acc >= 0 else None,
                        "epoch": epoch + 1
                    }, step=global_step)
                except Exception as e:
                    logger.warning(f"Error logging to wandb: {e}")

        if is_distributed:
            if rank == 0:
                sync_data[0] = {
                    'action': stop_signal.item(),
                    'tau': controller.tau,
                    'current_m': controller.current_m,
                    'lr_step_factor': controller.lr_step_factor,
                    'lr_scale': controller.lr_scale
                }
            dist.barrier(device_ids=[local_rank])
            dist.broadcast_object_list(sync_data, src=0)

            if rank != 0:
                stop_signal.fill_(sync_data[0]['action'])
                controller.tau = sync_data[0]['tau']
                controller.current_m = sync_data[0]['current_m']
                controller.lr_step_factor = sync_data[0]['lr_step_factor']
                controller.lr_scale = sync_data[0]['lr_scale']

        if controller.lr_step_factor != 1.0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= controller.lr_step_factor
                param_group['initial_lr'] *= controller.lr_step_factor
            controller.lr_step_factor = 1.0

        if stop_signal.item() == 2:
            # C1 FIX: Rollback delegado a handle_rollback en engine/loop.py
            global_step = handle_rollback(
                CONFIG, rank, use_wandb, global_step, model_q, model_k,
                optimizer, scaler, queue, is_compiled, is_distributed,
                warmup_steps, total_steps, final_lr_ratio, build_scheduler, 
                trainer, controller, logger
            )
            stop_signal.fill_(0)
            continue

        elif stop_signal.item() == 1:
            break

    if is_distributed: dist.destroy_process_group()

    if rank == 0 and use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    if rank == 0:
        if log_buffer:
            with open(log_file, "a", newline="") as f: csv.writer(f).writerows(log_buffer)

        logger.info("Iniciando Linear Probe...")
        best_ckpt_file = CONFIG["paths"]["best_checkpoint_path"]
        if not os.path.exists(best_ckpt_file):
            best_ckpt_file = CONFIG["paths"].get("checkpoint_path", "")
        if not best_ckpt_file or not os.path.exists(best_ckpt_file):
            logger.warning("⚠️ No se encontró checkpoint para Linear Probe. Saltando.")
        else:
            # C2 FIX: Linear probe usa weights_only=True
            best_ckpt = torch.load(best_ckpt_file, map_location=device, weights_only=True)

            clean_model_q = {
                k.replace("_orig_mod.", "").replace("module.", ""): v
                for k, v in best_ckpt["model_q"].items()
            }
            eval_model_base = ModelBase(
                dim=CONFIG["moco"]["dim"],
                predictor_hidden_dim=CONFIG["moco"].get("predictor_hidden_dim", 4096)
            ).to(device)
            eval_model_base.load_state_dict(clean_model_q, strict=False)

            num_classes = len(eval_ds.classes)
            head, acc, f1 = run_linear_probe(eval_model_base, eval_ds, val_ds, num_classes, CONFIG, device)

            torch.save(head, CONFIG["paths"]["encoder_export_path"].replace(".pth", "_head.pth"))
            torch.save(clean_model_q, CONFIG["paths"]["encoder_export_path"])
            with open(CONFIG["paths"]["metrics_path"], "w") as f:
                json.dump({"acc": acc, "f1": f1}, f, indent=4)
            logger.info("✅ Listo.")


if __name__ == "__main__":
    main()