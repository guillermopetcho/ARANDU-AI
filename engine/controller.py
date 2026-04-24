import math
import logging
from enum import IntEnum

class Action(IntEnum):
    CONTINUE = 0
    EARLY_STOP = 1
    ROLLBACK = 2

class EMA:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.value = None

    def update(self, x):
        if x is None: return self.value
        if self.value is None:
            self.value = x
        else:
            self.value = self.beta * self.value + (1 - self.beta) * x
        return self.value

class TrainingController:
    """
    AAC v3 (Edición Industrial): Microscopio de Observabilidad.
    Mantiene schedules deterministas y monitorea el espacio latente sin interferir.
    """
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("AranduSSL")
        
        # Estado de Control
        self.best_acc = 0.0
        self.patience = 0
        self.warmup_aborted = False
        self.temp_adj = 0.0
        
        # Observadores Suavizados (EMA)
        self.ema_unif = EMA(beta=0.9)
        self.ema_align = EMA(beta=0.9)
        self.ema_pos_sim = EMA(beta=0.9)
        self.ema_neg_sim = EMA(beta=0.9)
        self.ema_ratio_baseline = EMA(beta=0.99)
        self.last_ratio_ema = None
        
        # Historial
        self.history = {
            'loss': [], 'knn_acc': [], 'unif': [], 'align': [], 
            'pos_sim': [], 'neg_sim': [], 'ratio_norm': []
        }

    def get_dynamic_hyperparams(self, step, total_steps, metrics=None):
        if self.config.get("training", {}).get("exploitation_mode", False):
            # Explotación: parámetros fijos, temp baja, momentum alto
            return 0.999, 0.10

        m_base = self.config['moco']['momentum_base']
        momentum = 1 - (1 - m_base) * (math.cos(math.pi * step / max(1, total_steps)) + 1) / 2
        
        t_start = self.config['moco']['temp_start']
        t_end = self.config['moco']['temp_end']
        progress = min(step / max(1, total_steps), 1.0)
        temp_base = t_end + (t_start - t_end) * 0.5 * (1 + math.cos(math.pi * progress))
        
        # El ajuste temp_adj es sincronizado vía DDP en train.py
        final_temp = max(min(temp_base + self.temp_adj, 0.2), 0.05)
            
        return momentum, final_temp

    def step_epoch(self, epoch, curr_acc, metrics):
        u_ema = self.ema_unif.update(metrics['unif'])
        a_ema = self.ema_align.update(metrics['align'])
        ps_ema = self.ema_pos_sim.update(metrics['pos_sim'])
        ns_ema = self.ema_neg_sim.update(metrics['neg_sim'])
        
        self.history['loss'].append(metrics['loss'])
        self.history['unif'].append(metrics['unif'])
        self.history['align'].append(metrics['align'])
        self.history['pos_sim'].append(metrics['pos_sim'])
        self.history['neg_sim'].append(metrics['neg_sim'])
        if curr_acc >= 0: self.history['knn_acc'].append(curr_acc)

        is_warmup = (epoch < self.config["training"]["warmup_epochs"]) and not self.warmup_aborted
        is_exploitation = self.config.get("training", {}).get("exploitation_mode", False)
        
        if not is_warmup and not is_exploitation and a_ema is not None and u_ema is not None:
            current_ratio = a_ema / abs(u_ema)
            baseline = self.ema_ratio_baseline.update(current_ratio)
            ratio_norm = current_ratio / baseline if baseline > 0 else 1.0
            self.history['ratio_norm'].append(ratio_norm)
            
            if self.last_ratio_ema is not None:
                delta_R = current_ratio - self.last_ratio_ema
                eps = 0.001
                if delta_R < -eps:
                    self.temp_adj = min(self.temp_adj + 0.002, 0.03)
                    self.logger.info(f"📈 AAC: Tendencia ΔR={delta_R:.4f}. Temp +0.002")
                elif delta_R > eps:
                    self.temp_adj = max(self.temp_adj - 0.002, -0.03)
                    self.logger.info(f"📉 AAC: Tendencia ΔR={delta_R:.4f}. Temp -0.002")
            self.last_ratio_ema = current_ratio

        if curr_acc >= 0:
            if curr_acc > self.best_acc:
                self.best_acc = curr_acc
                self.patience = 0
            else:
                self.patience += 1
                if self.patience >= self.config["training"]["early_stopping_patience"]:
                    return Action.EARLY_STOP
        return Action.CONTINUE

    def state_dict(self):
        return {
            'best_acc': self.best_acc, 
            'patience': self.patience, 
            'temp_adj': self.temp_adj,
            'history': self.history,
            'ema_unif': self.ema_unif.value,
            'ema_align': self.ema_align.value,
            'ema_pos_sim': self.ema_pos_sim.value,
            'ema_neg_sim': self.ema_neg_sim.value,
            'ema_ratio_baseline': self.ema_ratio_baseline.value,
            'last_ratio_ema': self.last_ratio_ema
        }

    def load_state_dict(self, state):
        self.best_acc = state.get('best_acc', 0.0)
        self.patience = state.get('patience', 0)
        self.temp_adj = state.get('temp_adj', 0.0)
        if 'history' in state:
            for k, v in state['history'].items():
                if k in self.history: self.history[k] = v
        
        self.ema_unif.value = state.get('ema_unif')
        self.ema_align.value = state.get('ema_align')
        self.ema_pos_sim.value = state.get('ema_pos_sim')
        self.ema_neg_sim.value = state.get('ema_neg_sim')
        self.ema_ratio_baseline.value = state.get('ema_ratio_baseline')
        self.last_ratio_ema = state.get('last_ratio_ema')
