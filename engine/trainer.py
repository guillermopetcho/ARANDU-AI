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
        self.controller = config.get('_controller', None) # Se inyecta temporalmente por config o parámetro
        self.last_unif = 0.0 # Tracking para autoregulación

    def train_epoch(self, loader, epoch, global_step, total_steps, rank):
        self.model_q.train()
        epoch_loss, pos_sum, neg_sum, align_sum, unif_sum, std_sum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        grad_norm_sum, grad_steps = 0.0, 0

        pbar = tqdm(loader) if rank == 0 else loader
        epoch_start = time.time()

        for step, (v_q, v_k) in enumerate(pbar):
            # 🔥 Auto-Regulación Termodinámica via Controller
            if self.controller:
                momentum, temp = self.controller.get_dynamic_hyperparams(global_step, total_steps, self.last_unif)
            else:
                momentum, temp = 0.996, 0.2 # Fallback si no hay controlador
            
            v_q, v_k = v_q.to(self.device, non_blocking=True, memory_format=torch.channels_last), v_k.to(self.device, non_blocking=True, memory_format=torch.channels_last)

            is_accumulating = (step + 1) % self.config['training']['grad_accum_steps'] != 0 and (step + 1) != len(loader)
            sync_context = self.model_q.no_sync() if (self.is_distributed and is_accumulating) else contextlib.nullcontext()

            with sync_context:
                with autocast("cuda", enabled=self.config['training']['use_amp']):
                    q = self.model_q(v_q)
                    with torch.no_grad():
                        if self.is_distributed:
                            v_k, idx_shuffle = batch_shuffle_ddp(v_k)
                            k = self.model_k(v_k)
                            k = batch_unshuffle_ddp(k, idx_shuffle)
                        else:
                            k = self.model_k(v_k)
                            
                    l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
                    l_neg = torch.einsum('nc,ck->nk', [q, self.queue.queue.detach()])
                    logits = torch.cat([l_pos, l_neg], dim=1) / temp
                    
                    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
                    loss = F.cross_entropy(logits, labels)

                is_finite = torch.tensor(1 if torch.isfinite(loss) else 0, device=self.device)
                if self.is_distributed:
                    dist.all_reduce(is_finite, op=dist.ReduceOp.MIN)

                if is_finite.item() == 0:
                    dummy_loss = (q.sum() * 0.0)
                    self.scaler.scale(dummy_loss).backward()
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                self.scaler.scale(loss / self.config['training']['grad_accum_steps']).backward()

            # Optimización
            if not is_accumulating:
                if self.config['training']['use_amp']: self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model_q.parameters(), 1.0)
                
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

            # Métricas
            epoch_loss += loss.item()
            with torch.no_grad():
                metrics = compute_metrics(q, k)
                pos_sum += l_pos.mean().item()
                neg_sum += l_neg.mean().item()
                align_sum += metrics['alignment']
                unif_sum += metrics['uniformity']
                std_sum += metrics['std']
                self.last_unif = metrics['uniformity']

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