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
        pos_sim_sum, neg_sim_sum, grad_norm_sum, grad_steps = 0.0, 0.0, 0.0, 0
        valid_steps = 0
        # T1 FIX: Inicializar aliases antes del loop para evitar UnboundLocalError
        # si el primer batch produce NaN y el loop hace 'continue' sin asignarlos.
        q = k = l_pos = l_neg = None

        pbar = tqdm(loader) if rank == 0 else loader
        epoch_start = time.time()

        for step, batch in enumerate(pbar):
            # F6 FIX: local_crops es [B, N, C, H, W] (5D).
            # channels_last solo aplica a tensores 4D — se aplica por slice dentro del loop.
            v_q, v_k, local_crops = batch
            if local_crops.shape[1] > 0:
                local_crops = local_crops.to(self.device, non_blocking=True)  # mover sin format
            else:
                local_crops = None  # Placeholder vacío → desactivar multi-crop

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

                    # === Vistas Locales (Multi-Crop DINO-style) Vectorizadas ===
                    # T5 FIX: Vectorización completa para maximizar Throughput.
                    # Pasamos de O(N) llamadas secuenciales a O(1) llamada masiva.
                    loss_local = 0.0
                    if local_crops is not None:
                        B, N, C, H, W = local_crops.shape
                        # [B, N, C, H, W] -> [B*N, C, H, W]
                        v_local = local_crops.view(-1, C, H, W).contiguous().to(
                            memory_format=torch.channels_last)
                        
                        q_local = self.model_q(v_local, use_predictor=True)
                        
                        # Expandir k1 y labels para alinearse con el batch extendido [B*N]
                        k1_exp = k1.repeat_interleave(N, dim=0)
                        labels_local = labels.repeat_interleave(N, dim=0)
                        
                        l_pos_l = torch.einsum('nc,nc->n', [q_local, k1_exp]).unsqueeze(-1)
                        l_neg_l = torch.einsum('nc,ck->nk', [q_local, self.queue.queue.detach()])
                        logits_l = torch.cat([l_pos_l, l_neg_l], dim=1) / temp
                        
                        loss_local = F.cross_entropy(logits_l, labels_local)

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


                
                self.optimizer.zero_grad(set_to_none=True)
                global_step += 1

                with torch.no_grad():
                    momentum_update(self.model_q, self.model_k, momentum)
                    self.queue.enqueue_dequeue(k, step=global_step)

            # === Métricas === (solo si q y k fueron asignados en este step)
            if q is not None:
                epoch_loss += loss.item()
                valid_steps += 1  # L1 FIX: Solo contar batches procesados exitosamente
                with torch.no_grad():
                    metrics_step = compute_metrics(q, k)
                    pos_sum += l_pos.mean().item()
                    neg_sum += l_neg.mean().item()
                    align_sum += metrics_step['alignment']
                    unif_sum += metrics_step['uniformity']
                    pos_sim_sum += metrics_step['pos_sim']
                    neg_sim_sum += metrics_step['neg_sim']
                    std_sum += metrics_step['std']
                    self.last_unif = metrics_step['uniformity']

        # L1 FIX: Usar valid_steps (batches realmente procesados) en vez de len(loader)
        # para que los batches NaN saltados no diluyan las métricas reportadas.
        num_steps = max(1, valid_steps)

        if self.is_distributed:
            metrics_tensor = torch.tensor([
                epoch_loss, pos_sum, neg_sum, align_sum, unif_sum, std_sum,
                pos_sim_sum, neg_sim_sum, grad_norm_sum, float(grad_steps), float(valid_steps)
            ], device=self.device, dtype=torch.float32)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            metrics_tensor[:9] /= dist.get_world_size()
            epoch_loss, pos_sum, neg_sum, align_sum, unif_sum, std_sum, \
                pos_sim_sum, neg_sim_sum, grad_norm_sum, grad_steps, valid_steps = metrics_tensor.tolist()
            grad_steps = int(grad_steps)
            num_steps = max(1, int(valid_steps / dist.get_world_size()))  # Promedio de valid_steps

        # I1 FIX: Monitorear integridad del dataset.
        # Se lee desde _load_errors.Value (compartido entre workers via multiprocessing)
        # en lugar del atributo de instancia clásico, que sería invisible entre procesos forkeados.
        load_errors = 0
        if hasattr(loader.dataset, '_load_errors'):
            with loader.dataset._load_errors.get_lock():
                load_errors = loader.dataset._load_errors.value
                loader.dataset._load_errors.value = 0  # Reset atómico para la siguiente época
        
        error_rate = (load_errors / max(1, len(loader) * self.config['training']['batch_size'])) * 100
        if error_rate > 1.0 and rank == 0:
            logging.getLogger("AranduSSL").warning(f"⚠️ Alta tasa de errores de carga: {error_rate:.2f}% ({load_errors} imágenes).")

        return {
            'loss': epoch_loss / num_steps,
            'pos': pos_sum / num_steps,
            'neg': neg_sum / num_steps,
            'margin': (pos_sum - neg_sum) / num_steps,
            'align': align_sum / num_steps,
            'unif': unif_sum / num_steps,
            'pos_sim': pos_sim_sum / num_steps,
            'neg_sim': neg_sim_sum / num_steps,
            'std': std_sum / num_steps,
            'gn': grad_norm_sum / max(1, grad_steps),
            'tput': (len(loader) * self.config['training']['batch_size'] * (dist.get_world_size() if dist.is_initialized() else 1)) / max(1, time.time() - epoch_start),
            'data_err': error_rate
        }, global_step