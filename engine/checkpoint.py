"""engine/checkpoint.py — Gestión de checkpoints para AranduSSL.

Provee funciones de alto nivel para guardar, cargar y descubrir checkpoints
de forma atómica y segura (weights_only=True).

Extraído de train.py para mejorar la testeabilidad y separar responsabilidades.
"""

import os
import glob
import logging
import warnings

import torch

logger = logging.getLogger("AranduSSL")


# ---------------------------------------------------------------------------
# Utilidades de state_dict
# ---------------------------------------------------------------------------

def adapt_keys(state_dict: dict, is_compiled: bool, is_ddp: bool = False) -> dict:
    """Adapta las claves del state_dict para que coincidan con el modelo actual.

    Combinaciones soportadas:
      - plain:          key                  → key
      - compiled:       _orig_mod.key        → key
      - ddp:            module.key           → key
      - ddp+compiled:   module._orig_mod.key → key
    """
    new_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("_orig_mod.", "").replace("module.", "")
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


def clean_state_dict_for_save(state_dict: dict) -> dict:
    """Elimina prefijos de torch.compile (_orig_mod.) para portabilidad."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# Descubrimiento de checkpoints
# ---------------------------------------------------------------------------

def get_latest_valid_checkpoint(paths_dict: dict) -> "str | None":
    """Descubre el checkpoint más reciente válido (por global_step).

    Criterios de validez:
      - El archivo existe y es legible con weights_only=True.
      - No es un checkpoint de backbone pretrained (ej. resnet50-*.pth).
      - Contiene las claves 'global_step' y 'epoch'.

    Returns:
        Path al checkpoint más reciente, o None si no hay candidatos válidos.
    """
    candidates = []
    for key in ("checkpoint_path", "best_checkpoint_path"):
        p = paths_dict.get(key, "")
        if p and os.path.exists(p):
            candidates.append(p)

    base_dir = os.path.dirname(paths_dict.get("checkpoint_path", "/kaggle/working"))
    if os.path.isdir(base_dir):
        candidates.extend(glob.glob(os.path.join(base_dir, "*.pth")))

    valid_ckpts = []
    for p in set(candidates):
        if "resnet" in os.path.basename(p).lower():
            continue
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=True)
            if "global_step" in ckpt and "epoch" in ckpt:
                valid_ckpts.append((p, ckpt["global_step"]))
        except Exception:
            pass

    if not valid_ckpts:
        return None

    valid_ckpts.sort(key=lambda x: x[1], reverse=True)
    return valid_ckpts[0][0]


# ---------------------------------------------------------------------------
# Guardado atómico
# ---------------------------------------------------------------------------

def build_checkpoint_dict(
    model_q, model_k, optimizer, scheduler, scaler, queue,
    epoch: int, global_step: int, controller
) -> dict:
    """Construye el diccionario de checkpoint listo para torch.save().

    El controller.state_dict() serializa todos los tensores a listas Python
    nativas, garantizando compatibilidad con weights_only=True al cargar.
    """
    return {
        "model_q": clean_state_dict_for_save(model_q.state_dict()),
        "model_k": clean_state_dict_for_save(model_k.state_dict()),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "queue": queue.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "controller": controller.state_dict(),
    }


def save_checkpoint(path: str, ckpt_dict: dict) -> None:
    """Guarda un checkpoint de forma atómica: escribe en .tmp, luego os.replace().

    Garantía: el archivo en `path` nunca queda en estado parcial/corrupto.
    Si el proceso muere durante la escritura, el .tmp queda huérfano pero
    el checkpoint anterior en `path` permanece íntegro.
    """
    tmp_path = path + ".tmp"
    torch.save(ckpt_dict, tmp_path)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Carga completa (reanudación de entrenamiento)
# ---------------------------------------------------------------------------

def load_checkpoint(
    path: str,
    model_q, model_k, optimizer, scaler, queue, controller,
    lr: float, is_compiled: bool, is_distributed: bool,
    build_scheduler_fn, warmup_steps: int, total_steps: int,
    final_lr_ratio: float, trainer,
) -> "tuple[int, int, object]":
    """Carga un checkpoint y restaura todo el estado de entrenamiento.

    Usa weights_only=True para prevenir la ejecución de código arbitrario
    embebido en el pickle. Compatible con el formato de checkpoint actual
    donde controller.state_dict() contiene solo tipos Python nativos.

    Returns:
        (start_epoch, global_step, scheduler) restaurados.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    logger.info(
        f"📦 Checkpoint cargado: epoch={ckpt.get('epoch', '?')}, "
        f"global_step={ckpt.get('global_step', '?')}"
    )

    model_q.load_state_dict(adapt_keys(ckpt["model_q"], is_compiled, is_distributed), strict=False)
    model_k.load_state_dict(adapt_keys(ckpt["model_k"], is_compiled, False), strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])

    if "controller" in ckpt:
        controller.load_state_dict(ckpt["controller"])
    else:
        # Retrocompatibilidad con checkpoints anteriores al AAC
        controller.best_acc = ckpt.get("best_acc", 0.0)
        controller.warmup_aborted = ckpt.get("warmup_aborted", False)

    global_step = ckpt.get("global_step", 0)

    # Sanitizar el optimizer state para que el scheduler tome control limpio
    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = lr
        param_group["lr"] = lr

    # Reconstruir scheduler y hacer fast-forward determinista
    scheduler = build_scheduler_fn(optimizer, warmup_steps, total_steps, final_lr_ratio=final_lr_ratio)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(global_step):
            scheduler.step()

    trainer.scheduler = scheduler
    scaler.load_state_dict(ckpt["scaler"])
    queue.load_state_dict(ckpt["queue"])
    start_epoch = ckpt["epoch"] + 1
    return start_epoch, global_step, scheduler


# ---------------------------------------------------------------------------
# Carga parcial (rollback — solo pesos, sin restaurar controller ni epoch)
# ---------------------------------------------------------------------------

def load_weights_for_rollback(
    path: str,
    model_q, model_k, optimizer, scaler, queue,
    is_compiled: bool, is_distributed: bool,
) -> int:
    """Carga pesos del modelo para un Rollback sin restaurar el controller.

    A diferencia de load_checkpoint(), esta función NO toca el controller,
    el epoch ni el global_step del caller — solo restaura los pesos del modelo,
    el optimizer, el scaler y la queue desde el best checkpoint.

    Returns:
        global_step del checkpoint cargado.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model_q.load_state_dict(adapt_keys(ckpt["model_q"], is_compiled, is_distributed), strict=False)
    model_k.load_state_dict(adapt_keys(ckpt["model_k"], is_compiled, False), strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    queue.load_state_dict(ckpt["queue"])
    return ckpt.get("global_step", 0)
