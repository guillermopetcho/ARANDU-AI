import time
import torch
import torch.nn.functional as F
from torch.amp import autocast
import torch.distributed as dist
import contextlib
from tqdm.auto import tqdm

from utils.distributed import batch_shuffle_ddp, batch_unshuffle_ddp
from utils.metrics import compute_metrics
from engine.scheduler import momentum_update

class MoCoTrainer:
    def __init__(self, model_q, model_k, queue, optimizer, scheduler, scaler, config, device, is_distributed):
        self.model_q = model_q
        self.model_k = model_k
        self.queue = queue
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.config = config
        self.device = device
        self.is_distributed = is_distributed
        self.controller = config.get('_controller', None)
        self.last_unif = 0.0
        # E4 FIX: Resolver el tipo de device dinámicamente para que autocast funcione en CPU y GPU
        self.device_type = device.type if hasattr(device, 'type') else str(device).split(':')[0]
        # Peso de la pérdida de vistas locales (Multi-Crop)
        self.local_loss_weight = config['moco'].get('local_loss_weight', 0.5)

    def train_epoch(self, loader, epoch, global_step, total_steps, rank):
        self.model_q.train()
        epoch_loss, pos_sum, neg_sum, align_sum, unif_sum, std_sum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        grad_norm_sum, grad_steps = 0.0, 0
        # T1 FIX: Inicializar aliases antes del loop para evitar UnboundLocalError
        # si el primer batch produce NaN y el loop hace 'continue' sin asignarlos.
        q = k = l_pos = l_neg = None

        pbar = tqdm(loader) if rank == 0 else loader
        epoch_start = time.time()

        for step, batch in enumerate(pbar):
            # F6 FIX: local_crops es [B, N, C, H, W] (5D).
            # channels_last solo aplica a tensores 4D — se aplica por slice dentro del loop.
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                v_q, v_k, local_crops = batch
                local_crops = local_crops.to(self.device, non_blocking=True)  # mover sin format
            else:
                v_q, v_k = batch[0], batch[1]
                local_crops = None

            # --- Hiperparámetros dinámicos del Regulador Adaptativo ---
            if self.controller:
                momentum, temp = self.controller.get_dynamic_hyperparams(global_step, total_steps, self.last_unif)
            else:
                momentum, temp = 0.996, 0.2

            v_q = v_q.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            v_k = v_k.to(self.device, non_blocking=True, memory_format=torch.channels_last)

            is_accumulating = (step + 1) % self.config['training']['grad_accum_steps'] != 0 and (step + 1) != len(loader)
            sync_context = self.model_q.no_sync() if (self.is_distributed and is_accumulating) else contextlib.nullcontext()

            with sync_context:
                with autocast(self.device_type, enabled=self.config['training']['use_amp']):
                    # === Vista Global 1: query usa predictor (MoCo v3) ===
                    q1 = self.model_q(v_q, use_predictor=True)
                    with torch.no_grad():
                        if self.is_distributed:
                            v_k_sh, idx1 = batch_shuffle_ddp(v_k)
                            k1 = self.model_k(v_k_sh)
                            k1 = batch_unshuffle_ddp(k1, idx1)
                        else:
                            k1 = self.model_k(v_k)  # key NO usa predictor

                    l_pos1 = torch.einsum('nc,nc->n', [q1, k1]).unsqueeze(-1)
                    l_neg1 = torch.einsum('nc,ck->nk', [q1, self.queue.queue.detach()])
                    logits1 = torch.cat([l_pos1, l_neg1], dim=1) / temp

                    # === Vista Global 2: simétrica ===
                    q2 = self.model_q(v_k, use_predictor=True)
                    with torch.no_grad():
                        if self.is_distributed:
                            v_q_sh, idx2 = batch_shuffle_ddp(v_q)
                            k2 = self.model_k(v_q_sh)
                            k2 = batch_unshuffle_ddp(k2, idx2)
                        else:
                            k2 = self.model_k(v_q)

                    l_pos2 = torch.einsum('nc,nc->n', [q2, k2]).unsqueeze(-1)
                    l_neg2 = torch.einsum('nc,ck->nk', [q2, self.queue.queue.detach()])
                    logits2 = torch.cat([l_pos2, l_neg2], dim=1) / temp

                    labels = torch.zeros(logits1.shape[0], dtype=torch.long, device=self.device)
                    loss_global = (F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels)) * 0.5

                    # === Vistas Locales (Multi-Crop DINO-style) ===
                    # B7 FIX: Usar float 0.0 en vez de tensor sin grad para el acumulador.
                    # loss_local se convierte en tensor válido en el primer += desde F.cross_entropy.
                    loss_local = 0.0
                    if local_crops is not None:
                        n_local = local_crops.shape[1]
                        for i in range(n_local):
                            # F6 FIX: aplicar channels_last al slice 4D [B,C,H,W]
                            v_local = local_crops[:, i].contiguous().to(
                                memory_format=torch.channels_last)
                            q_local = self.model_q(v_local, use_predictor=True)
                            l_pos_l = torch.einsum('nc,nc->n', [q_local, k1]).unsqueeze(-1)
                            l_neg_l = torch.einsum('nc,ck->nk', [q_local, self.queue.queue.detach()])
                            logits_l = torch.cat([l_pos_l, l_neg_l], dim=1) / temp
                            loss_local = loss_local + F.cross_entropy(logits_l, labels) / n_local

                    loss = loss_global + self.local_loss_weight * loss_local

                    # Aliases para métricas
                    q, k, l_pos, l_neg = q1, k1, l_pos1, l_neg1

                # C3 FIX: usar .item() para evitar ambigüedad de torch.Tensor en contexto bool
                is_finite_val = loss.isfinite().item()
                is_finite = torch.tensor(1 if is_finite_val else 0, device=self.device)
                if self.is_distributed:
                    dist.all_reduce(is_finite, op=dist.ReduceOp.MIN)

                if is_finite.item() == 0:
                    self.optimizer.zero_grad(set_to_none=True)  # Limpiar grads contaminados
                    continue

                self.scaler.scale(loss / self.config['training']['grad_accum_steps']).backward()

            # === Paso de Optimización ===
            if not is_accumulating:
                # B8 FIX: unscale_ primero en ambos paths para medir grad_norm sobre gradientes reales.
                if self.config['training']['use_amp']:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(self.model_q.parameters(), 1.0)

                # Medir grad_norm DESPUÉS del unscale (siempre sobre gradientes en escala real)
                if global_step % 50 == 0:
                    gn = sum(p.grad.data.norm(2).item()**2 for p in self.model_q.parameters() if p.grad is not None)**0.5
                    grad_norm_sum += gn
                    grad_steps += 1

                if self.config['training']['use_amp']:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()

                # 🔥 Fix: Aplicar multiplicador dinámico de LR DESPUÉS del scheduler
                if self.controller:
                    lr_mult = getattr(self.controller, 'lr_multiplier', 1.0)
                    if lr_mult != 1.0:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= lr_mult

                self.optimizer.zero_grad(set_to_none=True)
                global_step += 1

                with torch.no_grad():
                    momentum_update(self.model_q, self.model_k, momentum)
                    self.queue.enqueue_dequeue(k, step=global_step)

            # === Métricas === (solo si q y k fueron asignados en este step)
            if q is not None:
                epoch_loss += loss.item()
                with torch.no_grad():
                    metrics_step = compute_metrics(q, k)
                    pos_sum += l_pos.mean().item()
                    neg_sum += l_neg.mean().item()
                    align_sum += metrics_step['alignment']
                    unif_sum += metrics_step['uniformity']
                    std_sum += metrics_step['std']
                    self.last_unif = metrics_step['uniformity']

        num_steps = max(1, len(loader))

        if self.is_distributed:
            metrics_tensor = torch.tensor([
                epoch_loss, pos_sum, neg_sum, align_sum, unif_sum, std_sum, grad_norm_sum
            ], device=self.device, dtype=torch.float32)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            metrics_tensor /= dist.get_world_size()
            epoch_loss, pos_sum, neg_sum, align_sum, unif_sum, std_sum, grad_norm_sum = metrics_tensor.tolist()

        return {
            'loss': epoch_loss / num_steps,
            'pos': pos_sum / num_steps,
            'neg': neg_sum / num_steps,
            'margin': (pos_sum - neg_sum) / num_steps,
            'align': align_sum / num_steps,
            'unif': unif_sum / num_steps,
            'std': std_sum / num_steps,
            'gn': grad_norm_sum / max(1, grad_steps),
            'tput': (len(loader) * self.config['training']['batch_size'] * (dist.get_world_size() if dist.is_initialized() else 1)) / max(1, time.time() - epoch_start)
        }, global_step