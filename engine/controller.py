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
        
        # Estado de Control (Mantenemos solo lo determinista/seguridad)
        self.best_acc = 0.0
        self.patience = 0
        self.warmup_aborted = False
        
        # Offsets (Fijados a 0 para reproducibilidad)
        self.temp_adj = 0.0
        self.momentum_boost = 0.0
        self.lr_multiplier = 1.0
        
        # Observadores Suavizados (EMA)
        self.ema_unif = EMA(beta=0.9)
        self.ema_align = EMA(beta=0.9)
        self.ema_pos_sim = EMA(beta=0.9)
        self.ema_neg_sim = EMA(beta=0.9)
        self.ema_ratio_baseline = EMA(beta=0.99)
        
        # Historial de Instrumentación
        self.history = {
            'loss': [], 'knn_acc': [], 'unif': [], 'align': [], 
            'pos_sim': [], 'neg_sim': [], 'ratio_norm': []
        }

    def get_dynamic_hyperparams(self, step, total_steps, metrics=None):
        """
        Schedules deterministas (Cosine) para garantizar estabilidad estadística.
        """
        # Momentum Cosine Schedule
        m_base = self.config['moco']['momentum_base']
        momentum = 1 - (1 - m_base) * (math.cos(math.pi * step / max(1, total_steps)) + 1) / 2
        
        # Temperature Cosine Schedule (Fijo, sin ajustes dinámicos)
        t_start = self.config['moco']['temp_start']
        t_end = self.config['moco']['temp_end']
        progress = min(step / max(1, total_steps), 1.0)
        temp = t_end + (t_start - t_end) * 0.5 * (1 + math.cos(math.pi * progress))
            
        return momentum, temp

    def step_epoch(self, epoch, curr_acc, metrics):
        """
        Observabilidad profunda. No modifica el estado del entrenamiento en caliente.
        """
        # 1. Actualizar Instrumentación
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

        # 2. Análisis de Ratio Normalizado (Para el Advisor Offline)
        if a_ema is not None and u_ema is not None:
            current_ratio = a_ema / abs(u_ema)
            baseline = self.ema_ratio_baseline.update(current_ratio)
            ratio_norm = current_ratio / baseline if baseline > 0 else 1.0
            self.history['ratio_norm'].append(ratio_norm)
            
            # --- NOTA: Ya no hay lógica reactiva de temp_adj aquí ---

        # 3. Safety Net (Checkpoints y Early Stop)
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
        return {'best_acc': self.best_acc, 'patience': self.patience, 'history': self.history}

    def load_state_dict(self, state):
        self.best_acc = state.get('best_acc', 0.0)
        self.patience = state.get('patience', 0)
        self.history = state.get('history', self.history)
