import math
import logging

class Action:
    CONTINUE = 0
    EARLY_STOP = 1
    ROLLBACK = 2

class TrainingController:
    """
    Controlador Adaptativo para el manejo del estado del entrenamiento y optimización 
    de hiperparámetros en función de métricas.
    """
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("AranduSSL")
        
        # Estado de Control
        self.best_acc = 0.0
        self.patience = 0
        self.warmup_aborted = False
        self.last_rollback_epoch = -999
        self.rollback_cooldown = 3
        
        # Memoria histórica
        self.history = {
            'loss': [], 'knn_acc': [], 'unif': [], 'align': [], 'gn': []
        }

    def get_dynamic_hyperparams(self, step, total_steps, current_unif):
        """
        Devuelve el Momentum y la Temperatura para el paso actual, reaccionando
        a posibles colapsos dimensionales detectados mediante 'current_unif'.
        """
        m_base = self.config['moco']['momentum_base']
        momentum = 1 - (1 - m_base) * (math.cos(math.pi * step / max(1, total_steps)) + 1) / 2
        
        warmup_steps = self.config['moco'].get('temp_warmup_steps', 100)
        if step < warmup_steps:
            temp = 0.5 - (0.5 - self.config['moco']['temp_start']) * (step / warmup_steps)
        else:
            progress = min(max((step - warmup_steps) / max(1, total_steps - warmup_steps), 0.0), 1.0)
            temp = self.config['moco']['temp_end'] + (self.config['moco']['temp_start'] - self.config['moco']['temp_end']) * 0.5 * (1 + math.cos(math.pi * progress))
            
        # Reflejo autoinmune: Si la uniformidad es muy baja (colapso inminente)
        temp_boost = 1.0
        if current_unif < -4.0:
            temp_boost = 1.15
            momentum = min(momentum * 0.99, 0.99) # Forzar rotación en la cola
            
        return momentum, max(temp * temp_boost, 0.05)

    def step_epoch(self, epoch, curr_acc, metrics):
        """
        Se llama al final de cada época para actualizar la memoria y tomar 
        decisiones arquitectónicas (ej. Rollback, Early Stop).
        """
        # Actualizar memoria
        self.history['loss'].append(metrics['loss'])
        if curr_acc >= 0: self.history['knn_acc'].append(curr_acc)
        self.history['unif'].append(metrics['unif'])
        self.history['gn'].append(metrics['gn'])
        
        action = Action.CONTINUE
        is_warmup = (epoch < self.config["training"]["warmup_epochs"]) and not self.warmup_aborted
        
        if curr_acc >= 0:
            # Lógica de progreso
            if curr_acc > self.best_acc:
                self.best_acc = curr_acc
                self.patience = 0
            else:
                self.patience += 1
                if self.patience >= self.config["training"]["early_stopping_patience"]:
                    self.logger.warning("⛔ Early Stopping activado por el Controlador.")
                    return Action.EARLY_STOP
                    
            # Lógica de Auto-Revert (Rollback) durante Warmup
            if is_warmup and self.best_acc > 0:
                threshold = max(0.02, 0.05 * (1 - self.best_acc))
                if curr_acc < (self.best_acc - threshold):
                    if (epoch - self.last_rollback_epoch) > self.rollback_cooldown:
                        self.logger.warning(f"⚠️ Caída catastrófica detectada (KNN {curr_acc:.4f} < Best {self.best_acc:.4f}). Controlador ordena ROLLBACK.")
                        self.last_rollback_epoch = epoch
                        action = Action.ROLLBACK
                    else:
                        self.logger.info(f"⏳ Ignorando caída (KNN {curr_acc:.4f}), en periodo de cooldown.")
                        
        return action

    def state_dict(self):
        return {
            'best_acc': self.best_acc,
            'patience': self.patience,
            'warmup_aborted': self.warmup_aborted,
            'last_rollback_epoch': self.last_rollback_epoch,
            'history': self.history
        }

    def load_state_dict(self, state):
        self.best_acc = state.get('best_acc', 0.0)
        self.patience = state.get('patience', 0)
        self.warmup_aborted = state.get('warmup_aborted', False)
        self.last_rollback_epoch = state.get('last_rollback_epoch', -999)
        self.history = state.get('history', self.history)
