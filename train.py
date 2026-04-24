import os
import datetime
import math
import copy
import logging
import csv
import yaml
import json
import warnings
import glob
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


from engine.trainer import MoCoTrainer
from engine.scheduler import build_scheduler
from engine.controller import TrainingController, Action
from evaluation.knn import extract_features_fast, fast_knn
from evaluation.linear_probe import run_linear_probe
from models.moco import build_index, MoCoDataset, ModelBase, MoCoQueue
from utils.metrics import get_module_stats


def get_model_module(model, is_distributed):
    return model.module if is_distributed else model


def resolve_kaggle_paths(paths_config, rank=0):
    """Auto-descubre la ubicación real del dataset en Kaggle.
    
    En Kaggle, los datasets se montan en /kaggle/input/ con una estructura que
    incluye el username del dueño del dataset. Cuando el notebook se forkea,
    el username puede cambiar. Esta función busca el dataset por su nombre
    de carpeta (slug) independientemente de la ruta completa del usuario.
    
    Estrategia:
      1. Si el path configurado ya existe, usarlo directamente (zero-cost path).
      2. Si no existe, extraer el nombre de la carpeta final del path configurado
         y buscarlo recursivamente bajo /kaggle/input/.
      3. Si se encuentra, parchear TODOS los paths del config que dependan del
         dataset root para mantener consistencia.
    """
    logger = logging.getLogger("AranduSSL")
    dataset_root = paths_config.get("dataset_root", "")
    
    # Si el path ya existe, no hay nada que resolver
    if os.path.isdir(dataset_root):
        return paths_config
    
    # Solo aplicar auto-discovery en entorno Kaggle
    if not os.path.isdir("/kaggle/input"):
        return paths_config
    
    # Extraer el nombre de la carpeta del dataset (última parte del path)
    # Ejemplo: de "/kaggle/input/datasets/user/base-soja-encoder-full/BASE-SOJA-ENCODER-FULL"
    #          extraemos "BASE-SOJA-ENCODER-FULL"
    dataset_folder_name = os.path.basename(dataset_root)
    if not dataset_folder_name:
        return paths_config
    
    # Buscar la carpeta en /kaggle/input/ (puede estar a cualquier profundidad)
    found = None
    for dirpath, dirnames, _ in os.walk("/kaggle/input"):
        if dataset_folder_name in dirnames:
            found = os.path.join(dirpath, dataset_folder_name)
            break
    
    if found is None:
        # Fallback: intentar buscar por el slug del dataset (penúltimo componente)
        # Ejemplo: "base-soja-encoder-full"
        path_parts = dataset_root.rstrip("/").split("/")
        if len(path_parts) >= 2:
            dataset_slug = path_parts[-2]  # ej. "base-soja-encoder-full"
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
    
    # Parchear todos los paths que usen el dataset_root antiguo
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

# B1 FIX: Eliminada la función local duplicada. Se usa únicamente la importada de engine/scheduler.
# La función local sobreescribía la importada sin querer.

def adapt_keys(state_dict, is_compiled, is_ddp=False):
    """Adapta las claves del state_dict para que coincidan con el modelo actual.

    Combinaciones soportadas:
      - plain:           key                    → key
      - compiled:        _orig_mod.key          → key  (pero guardamos sin _orig_mod via clean_state_dict_for_save)
      - ddp:             module.key             → key
      - ddp+compiled:    module._orig_mod.key   → key

    Al cargar: limpiar prefijos y re-añadir los que correspondan al modelo vivo.
    """
    new_dict = {}
    for k, v in state_dict.items():
        # 1. Limpiar cualquier prefijo del checkpoint guardado
        clean_k = k
        clean_k = clean_k.replace("_orig_mod.", "")
        clean_k = clean_k.replace("module.", "")

        # 2. Reconstruir el prefijo según el modelo vivo
        if is_ddp and is_compiled:
            new_key = "module._orig_mod." + clean_k
        elif is_ddp:
            new_key = "module." + clean_k
        elif is_compiled:
            new_key = "_orig_mod." + clean_k
        else:
            new_key = clean_k

        new_dict[new_key] = v
    return new_dict


def clean_state_dict_for_save(state_dict):
    """B6 FIX: Movida a scope de módulo para no redefinirse en cada época."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

def main():
    torch.set_float32_matmul_precision('high')

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    
    if is_distributed:
        torch.cuda.set_device(local_rank)
        # T6 FIX: Aumentar timeout a 2 horas (7200s) para permitir escaneo de datasets grandes
        # sin que los ranks secundarios expiren en la barrera de build_index.
        dist.init_process_group(
            backend="nccl", 
            rank=rank, 
            world_size=world_size,
            timeout=datetime.timedelta(seconds=7200)
        )
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix: Get the directory where train.py is located to dynamically find config/moco.yaml
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(BASE_DIR, "config", "moco.yaml")
    
    with open(config_path, "r") as f:
        CONFIG = yaml.safe_load(f)

    # 🔥 FIX: Auto-discovery de paths en Kaggle independiente del usuario.
    # El dataset slug (ej. "base-soja-encoder-full") y el nombre de la carpeta raíz
    # (ej. "BASE-SOJA-ENCODER-FULL") son constantes, pero el username en la ruta cambia.
    # Buscamos recursivamente bajo /kaggle/input/ para resolver el path real.
    CONFIG["paths"] = resolve_kaggle_paths(CONFIG["paths"], rank)

    # Inicializar el Controlador Adaptativo
    controller = TrainingController(CONFIG)
    CONFIG['_controller'] = controller

    eff_batch = CONFIG["training"]["batch_size"] * CONFIG["training"]["grad_accum_steps"] * world_size
    lr = min(CONFIG["training"]["lr_base"] * (eff_batch / 256.0), 0.15) # 🔥 Fix Pro: Cap bajado a 0.15 para mayor estabilidad SSL
    
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
            # E7 FIX: Excluir '_controller' (objeto Python no serializable) del config de W&B
            wandb_config = {k: v for k, v in CONFIG.items() if k != '_controller'}
            wandb.init(
                project=CONFIG.get("wandb", {}).get("project", "MoCo-ENCODER"),
                config=wandb_config,
                name=f"run_effbatch_{eff_batch}_lr_{lr:.4f}"
            )
        except ImportError:
            logger.warning("WandB está habilitado en config pero no está instalado. Usando logger estándar.")
            use_wandb = False

    if rank == 0: logger.info(f"Iniciando. EffBatch: {eff_batch}, LR: {lr:.6f}")

    seed = CONFIG["training"]["seed"] + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    cudnn.benchmark = True

    paths = build_index(CONFIG["paths"]["dataset_root"], rank, CONFIG["paths"]["index_cache_path"])
    dataset = MoCoDataset(paths, moco_config=CONFIG["moco"])
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if is_distributed else None
    
    n_workers = CONFIG["training"]["num_workers"]
    train_loader = DataLoader(
        dataset, batch_size=CONFIG["training"]["batch_size"], shuffle=(sampler is None),
        sampler=sampler, num_workers=n_workers,
        pin_memory=True, drop_last=True,
        # E1/W2 FIX: persistent_workers y prefetch_factor solo son válidos cuando num_workers > 0
        persistent_workers=(n_workers > 0),
        prefetch_factor=2 if n_workers > 0 else None
    )

    eval_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    eval_ds = ImageFolder(CONFIG["paths"]["eval_train_root"], transform=eval_transform)
    val_ds = ImageFolder(CONFIG["paths"]["eval_val_root"], transform=eval_transform)
    
    # D1 FIX: usar min(2, n_workers) para respetar la config de workers
    indices = torch.randperm(len(eval_ds))[:min(CONFIG["eval"]["subset_size"], len(eval_ds))].tolist()
    eval_workers = min(2, n_workers)
    eval_train_loader = DataLoader(Subset(eval_ds, indices), batch_size=128,
                                   num_workers=eval_workers, pin_memory=True)
    eval_val_loader   = DataLoader(val_ds, batch_size=128,
                                   num_workers=eval_workers, pin_memory=True)

    # C4 FIX: deepcopy ANTES de torch.compile para evitar que model_q y model_k
    # compartan buffers internos del compilador con model_base.
    model_base_raw = ModelBase(
        dim=CONFIG["moco"]["dim"],
        predictor_hidden_dim=CONFIG["moco"].get("predictor_hidden_dim", 4096)
    ).to(device, memory_format=torch.channels_last)

    # B3 FIX: SyncBatchNorm se aplica ANTES del deepcopy para que
    # model_q y model_k hereden los SyncBatchNorm correctamente.
    if is_distributed:
        model_base_raw = nn.SyncBatchNorm.convert_sync_batchnorm(model_base_raw)

    # Deepcopy de la arquitectura limpia (antes de compilar)
    model_q = copy.deepcopy(model_base_raw)
    model_k = copy.deepcopy(model_base_raw).to(device, memory_format=torch.channels_last)

    # Compilar solo model_q (el entrenado) para maximizar rendimiento
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

    optimizer = torch.optim.SGD(model_q.parameters(), lr=lr, momentum=0.9, weight_decay=float(CONFIG["training"]["weight_decay"]))
    # M5 FIX: Pasar device_type explícitamente para evitar deprecation warning en PyTorch 2.x+
    scaler = GradScaler(device.type, enabled=CONFIG["training"]["use_amp"])
    
    total_steps = CONFIG["training"]["epochs"] * math.ceil(len(train_loader) / CONFIG["training"]["grad_accum_steps"])
    warmup_steps = max(1, CONFIG["training"]["warmup_epochs"] * math.ceil(len(train_loader) / CONFIG["training"]["grad_accum_steps"]))
    
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    trainer = MoCoTrainer(model_q, model_k, queue, optimizer, scheduler, scaler, CONFIG, device, is_distributed)

    start_epoch, global_step = 0, 0
    log_buffer = []
    stop_signal = torch.tensor(0, device=device)
    # Lista para sincronización de objetos DDP (Action, temp_adj, etc.)
    sync_data = [None] 
    
    ckpt_path = CONFIG["paths"].get("checkpoint_path", "")
    best_ckpt_path = CONFIG["paths"].get("best_checkpoint_path", "")
    
    ckpt_to_load = None
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt_to_load = ckpt_path
    elif best_ckpt_path and os.path.exists(best_ckpt_path):
        ckpt_to_load = best_ckpt_path
        
    if ckpt_to_load:
        if rank == 0: logger.info(f"🔄 Reanudando entrenamiento desde {ckpt_to_load}")
        ckpt = torch.load(ckpt_to_load, map_location="cpu", weights_only=False)
        
        model_q.load_state_dict(adapt_keys(ckpt["model_q"], is_compiled, is_distributed), strict=False)
        model_k.load_state_dict(adapt_keys(ckpt["model_k"], is_compiled, False), strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        
        # Cargar estado del controlador o fallback de retrocompatibilidad
        if "controller" in ckpt:
            controller.load_state_dict(ckpt["controller"])
        else:
            controller.best_acc = ckpt.get("best_acc", 0.0)
            controller.warmup_aborted = ckpt.get("warmup_aborted", False)
            
        global_step = ckpt.get("global_step", 0)
        
        # 🔥 FIX CRÍTICO: Sanitizar el optimizer state dict que viene del checkpoint
        # Esto previene que inicializaciones corruptas de schedulers previos o rollbacks 
        # provoquen que el LR se dispare al instanciar CosineAnnealingLR.
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = lr
            param_group['lr'] = lr  # Reset para que el scheduler tome control matemáticamente
            
        # Reconstruir scheduler siempre desde cero y hacer fast-forward determinista
        # Esto evita incompatibilidades de PyTorch con SequentialLR.load_state_dict()
        scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(global_step):
                scheduler.step()

        # Actualizar la referencia del scheduler en el trainer tras reconstruirlo
        trainer.scheduler = scheduler

        scaler.load_state_dict(ckpt["scaler"])
        queue.load_state_dict(ckpt["queue"])
        start_epoch = ckpt["epoch"] + 1
    log_file = CONFIG["paths"]["metrics_path"].replace('.json', '_log.csv')
    proj_log_file = CONFIG["paths"]["metrics_path"].replace('.json', '_projector.csv')

    if rank == 0:
        if not os.path.exists(log_file):
            with open(log_file, "w", newline="") as f:
                csv.writer(f).writerow(["epoch","loss","lr","knn_acc","pos","neg","margin","align","unif","psim","nsim","rnorm","std","gn","tput","data_err"])
        
        # Inicializar log de projector si no existe
        if not os.path.exists(proj_log_file):
            with open(proj_log_file, "w", newline="") as f:
                # Columnas base, el resto se autogeneran en el primer log
                csv.writer(f).writerow(["epoch", "total_mean", "total_std", "total_norm"])

    for epoch in range(start_epoch, CONFIG["training"]["epochs"]):
        # B4 FIX: Guard explícito para evitar AttributeError cuando sampler es None
        if is_distributed and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)
        
        metrics, global_step = trainer.train_epoch(train_loader, epoch, global_step, total_steps, rank)
        
        if rank == 0:
            curr_acc = -1
            eval_freq = 1

            if (epoch + 1) % eval_freq == 0:
                eval_model = get_model_module(model_q, is_distributed)
                # Poner model_q en eval para la extracción de features
                eval_model.eval()
                X_t, y_t = extract_features_fast(eval_model, eval_train_loader, device)
                X_v, y_v = extract_features_fast(eval_model, eval_val_loader, device)
                # Restaurar a train mode inmediatamente
                eval_model.train()
                curr_acc = fast_knn(X_t, y_t, X_v, y_v, k=CONFIG["eval"]["knn_k"])
                logger.info(f"KNN ACC: {curr_acc:.4f}")
                
                # Delegar decisiones lógicas al Controlador Adaptativo
                action = controller.step_epoch(epoch, curr_acc, metrics)
                
                if action == Action.EARLY_STOP:
                    stop_signal.fill_(1)
                elif action == Action.ROLLBACK:
                    stop_signal.fill_(2)
                    
            ckpt = {
                "model_q": clean_state_dict_for_save(model_q.state_dict()),
                "model_k": clean_state_dict_for_save(model_k.state_dict()),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "queue": queue.state_dict(),
                "epoch": epoch, "global_step": global_step,
                "controller": controller.state_dict()
            }
            
            tmp_ckpt_path = CONFIG["paths"]["checkpoint_path"] + ".tmp"
            torch.save(ckpt, tmp_ckpt_path)
            os.replace(tmp_ckpt_path, CONFIG["paths"]["checkpoint_path"])
            
            if (epoch + 1) % eval_freq == 0:
                # T3 FIX: Comparar con >= en vez de == para evitar problemas de precisión float.
                # best_acc se actualiza dentro de step_epoch, así curr_acc >= best_acc
                # equivale a "este epoch igualó o superó el máximo histórico".
                if curr_acc >= controller.best_acc and curr_acc > 0:
                    tmp_best_path = CONFIG["paths"]["best_checkpoint_path"] + ".tmp"
                    torch.save(ckpt, tmp_best_path)
                    os.replace(tmp_best_path, CONFIG["paths"]["best_checkpoint_path"])
                    logger.info("🏆 Best model guardado")

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
                
                # S1 FIX: Guardar estadísticas de parámetros del Projector
                model_to_save = get_model_module(model_q, is_distributed)
                if hasattr(model_to_save, 'projector'):
                    p_stats = get_module_stats(model_to_save.projector)
                    p_stats['epoch'] = epoch + 1
                    
                    file_exists = os.path.isfile(proj_log_file)
                    with open(proj_log_file, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=sorted(p_stats.keys()))
                        if f.tell() == 0: # Caso borde si el archivo se vació
                            writer.writeheader()
                        writer.writerow(p_stats)
                
            logger.info(f"Ep {epoch+1} | Loss {metrics['loss']:.3f} | U: {metrics['unif']:.2f} | LR: {optimizer.param_groups[0]['lr']:.4f} (x{controller.lr_multiplier:.2f}) | Err: {metrics.get('data_err', 0):.2f}%")
            
            if use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "train/loss": metrics['loss'],
                        "train/lr": optimizer.param_groups[0]['lr'],
                        "train/lr_multiplier": controller.lr_multiplier,
                        "train/momentum_boost": controller.momentum_boost,
                        "train/temp_adj": controller.temp_adj,
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
                        "eval/knn_acc": curr_acc if curr_acc >= 0 else None,
                        "epoch": epoch + 1
                    }, step=global_step)
                except Exception:
                    pass

        # R4 FIX: Sincronización total del estado del controlador vía DDP.
        # Rank 0 difunde la acción y el ajuste de temperatura a los demás.
        if is_distributed:
            if rank == 0:
                sync_data[0] = {
                    'action': stop_signal.item(),
                    'temp_adj': controller.temp_adj
                }
            dist.barrier()
            dist.broadcast_object_list(sync_data, src=0)
            
            if rank != 0:
                stop_signal.fill_(sync_data[0]['action'])
                controller.temp_adj = sync_data[0]['temp_adj']
        
        if stop_signal.item() == 2:
            if rank == 0: logger.info("🔄 Iniciando proceso de Rollback...")
            if rank == 0 and use_wandb:
                try:
                    import wandb
                    wandb.log({"event/rollback": 1}, step=global_step)
                except Exception:
                    pass
            # E2 FIX: Verificar que el best checkpoint exista antes del Rollback.
            # Si el modelo aún no guardó un best (primeras épocas), usar el checkpoint regular.
            rollback_ckpt_path = CONFIG["paths"]["best_checkpoint_path"]
            if not os.path.exists(rollback_ckpt_path):
                rollback_ckpt_path = CONFIG["paths"].get("checkpoint_path", "")
            if not rollback_ckpt_path or not os.path.exists(rollback_ckpt_path):
                if rank == 0: logger.warning("⚠️ Rollback solicitado pero no hay checkpoint disponible. Continuando.")
                stop_signal.fill_(0)
                continue
            ckpt = torch.load(rollback_ckpt_path, map_location="cpu", weights_only=False)
            model_q.load_state_dict(adapt_keys(ckpt["model_q"], is_compiled, is_distributed), strict=False)
            model_k.load_state_dict(adapt_keys(ckpt["model_k"], is_compiled, False), strict=False)
            optimizer.load_state_dict(ckpt["optimizer"])
            
            # 🔥 Fix Pro: Bajar el LR a la mitad para estabilizar inmediatamente tras el rollback
            # Sanitizamos explícitamente todo el optimizador para la nueva curva térmica
            for param_group in optimizer.param_groups:
                param_group['initial_lr'] = lr * 0.5  # Modificar toda la curva base
                param_group['lr'] = lr * 0.5
                
            scaler.load_state_dict(ckpt["scaler"])
            queue.load_state_dict(ckpt["queue"])
            global_step = ckpt["global_step"]
            
            # Recreamos el scheduler estandar y avanzamos
            scheduler = build_scheduler(optimizer, warmup_steps, total_steps)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(global_step):
                    scheduler.step()
                    
            trainer.scheduler = scheduler # Actualizamos la referencia en el trainer
            
            controller.warmup_aborted = True
            stop_signal.fill_(0)
            if rank == 0: logger.info("✅ Rollback completado. Abortando Warmup e iniciando fase de Decaimiento Cosenoidal.")
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
        # B5 FIX: Verificar que el best checkpoint exista antes de intentar cargarlo.
        # Si nunca hubo una época de evaluación o best_acc > 0, usamos el checkpoint regular.
        best_ckpt_file = CONFIG["paths"]["best_checkpoint_path"]
        if not os.path.exists(best_ckpt_file):
            best_ckpt_file = CONFIG["paths"].get("checkpoint_path", "")
        if not best_ckpt_file or not os.path.exists(best_ckpt_file):
            logger.warning("⚠️ No se encontró ningún checkpoint para Linear Probe. Saltando.")
        else:
            best_ckpt = torch.load(best_ckpt_file, map_location=device, weights_only=False)
            # L6 FIX: clean_state_dict_for_save ya elimina _orig_mod y module prefixes.
            # Solo necesitamos extraer las claves del encoder (descartar projector/predictor).
            clean_enc = {}
            for k, v in best_ckpt["model_q"].items():
                k_clean = k.replace("_orig_mod.", "").replace("module.", "")
                if k_clean.startswith("encoder."):
                    clean_enc[k_clean.replace("encoder.", "")] = v

            eval_enc = models.resnet50(weights=None)
            eval_enc.fc = nn.Identity()
            eval_enc.load_state_dict(clean_enc, strict=False)

            num_classes = len(eval_ds.classes)
            head, acc, f1 = run_linear_probe(eval_enc, eval_ds, val_ds, num_classes, CONFIG, device)

            torch.save(head, CONFIG["paths"]["encoder_export_path"].replace(".pth", "_head.pth"))
            torch.save(clean_enc, CONFIG["paths"]["encoder_export_path"])
            with open(CONFIG["paths"]["metrics_path"], "w") as f: json.dump({"acc": acc, "f1": f1}, f, indent=4)
            logger.info("✅ Listo.")

if __name__ == "__main__":
    main()