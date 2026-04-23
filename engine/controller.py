import math
import logging

class Action:
    CONTINUE = 0
    EARLY_STOP = 1
    ROLLBACK = 2

class TrainingController:
    """
    Controlador Adaptativo Avanzado para el manejo del estado del entrenamiento 
    y regulación dinámica de hiperparámetros (LR, Momentum, Temp, WD) 
    en función de la evolución de las métricas.
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
        
        # Variables de Regulación Adaptativa (Multiplicadores/Offsets dinámicos)
        self.lr_multiplier = 1.0
        self.momentum_boost = 0.0
        self.temp_boost = 1.0
        
        # Estado Analítico (EMA - Exponential Moving Average)
        self.ema_loss = None
        self.ema_acc = None
        self.ema_unif = None
        self.alpha = 0.3  # Factor de suavizado para tendencias
        
        # Memoria histórica
        self.history = {
            'loss': [], 'knn_acc': [], 'unif': [], 'align': [], 'gn': []
        }

    def get_dynamic_hyperparams(self, step, total_steps, current_unif):
        """
        Calcula la base de Momentum y Temperatura y aplica los reguladores dinámicos.
        """
        m_base = self.config['moco']['momentum_base']
        momentum = 1 - (1 - m_base) * (math.cos(math.pi * step / max(1, total_steps)) + 1) / 2
        
        warmup_steps = self.config['moco'].get('temp_warmup_steps', 100)
        if step < warmup_steps:
            # Warmup lineal: comienza en temp_start y baja suavemente hacia temp_end
            temp = self.config['moco']['temp_start']
        else:
            progress = min(max((step - warmup_steps) / max(1, total_steps - warmup_steps), 0.0), 1.0)
            temp = self.config['moco']['temp_end'] + (self.config['moco']['temp_start'] - self.config['moco']['temp_end']) * 0.5 * (1 + math.cos(math.pi * progress))
            
        # Reflejo autoinmune inmediato por colapso (micro-step)
        if current_unif < -4.5:
            temp *= 1.1
            momentum = min(momentum * 0.99, 0.99)
            
        # Aplicar reguladores macro (época a época)
        final_momentum = min(max(momentum + self.momentum_boost, 0.9), 0.9999)
        final_temp = max(temp * self.temp_boost, 0.05)
            
        return final_momentum, final_temp

    def step_epoch(self, epoch, curr_acc, metrics):
        """
        Evalúa el ajuste óptimo de parámetros automático coordinando el 
        incremento y disminución de los valores de salida (Loss, Unif, Acc).
        """
        curr_loss = metrics['loss']
        curr_unif = metrics['unif']
        
        self.history['loss'].append(curr_loss)
        if curr_acc >= 0: self.history['knn_acc'].append(curr_acc)
        self.history['unif'].append(curr_unif)
        self.history['gn'].append(metrics['gn'])
        
        # 1. Actualización de Tendencias (EMA)
        if self.ema_loss is None:
            self.ema_loss = curr_loss
            self.ema_acc = curr_acc if curr_acc >= 0 else 0.0
            self.ema_unif = curr_unif
        else:
            self.ema_loss = (1 - self.alpha) * self.ema_loss + self.alpha * curr_loss
            if curr_acc >= 0:
                self.ema_acc = (1 - self.alpha) * self.ema_acc + self.alpha * curr_acc
            self.ema_unif = (1 - self.alpha) * self.ema_unif + self.alpha * curr_unif

        action = Action.CONTINUE
        is_warmup = (epoch < self.config["training"]["warmup_epochs"]) and not self.warmup_aborted
        
        # 2. Regulación Dinámica de Parámetros
        if not is_warmup:
            loss_delta = self.ema_loss - curr_loss # Positivo = Mejorando
            acc_delta = curr_acc - self.ema_acc    # Positivo = Mejorando
            
            # Caso A: Divergencia Severa / Pico de Loss
            if curr_loss > self.ema_loss * 1.15:
                self.logger.warning(f"📉 Divergencia detectada (Loss {curr_loss:.3f} > EMA {self.ema_loss:.3f}). Reduciendo multiplicador LR a la mitad.")
                self.lr_multiplier = max(self.lr_multiplier * 0.5, 0.1)
                self.momentum_boost = min(self.momentum_boost + 0.005, 0.02) # Subir momentum para estabilizar gradientes
                
            # Caso B: Alerta de Colapso Dimensional
            elif curr_unif < -4.0 or (self.ema_unif - curr_unif) > 0.4:
                self.logger.warning(f"⚠️ Alerta de Colapso (Unif: {curr_unif:.2f}). Aumentando Temp y reduciendo Momentum dinámico.")
                self.temp_boost = min(self.temp_boost * 1.1, 1.5)
                self.momentum_boost = max(self.momentum_boost - 0.01, -0.05)
                self.lr_multiplier = max(self.lr_multiplier * 0.8, 0.2)
                
            # Caso C: Estancamiento (Plateau)
            elif abs(loss_delta) < 0.005 and abs(acc_delta) < 0.005 and epoch > 15:
                self.logger.info("🐌 Estancamiento detectado. Aumentando multiplicador LR temporalmente (+10%) para explorar.")
                self.lr_multiplier = min(self.lr_multiplier * 1.1, 2.0)
                
            # Caso D: Convergencia Estable Robusta
            elif loss_delta > 0.02 or acc_delta > 0.01:
                # Normalizar los boosters suavemente hacia el baseline (Curva original)
                if self.lr_multiplier > 1.0: self.lr_multiplier = max(self.lr_multiplier * 0.95, 1.0)
                elif self.lr_multiplier < 1.0: self.lr_multiplier = min(self.lr_multiplier * 1.05, 1.0)
                
                self.temp_boost = max(self.temp_boost * 0.95, 1.0)
                if self.momentum_boost > 0: self.momentum_boost = max(self.momentum_boost - 0.002, 0.0)
                elif self.momentum_boost < 0: self.momentum_boost = min(self.momentum_boost + 0.002, 0.0)

        # 3. Lógica de Progreso y Control de Flujo (Early Stop / Rollback)
        if curr_acc >= 0:
            if curr_acc > self.best_acc:
                self.best_acc = curr_acc
                self.patience = 0
            else:
                self.patience += 1
                if self.patience >= self.config["training"]["early_stopping_patience"]:
                    self.logger.warning("⛔ Early Stopping activado por el Controlador.")
                    return Action.EARLY_STOP
                    
            if is_warmup and self.best_acc > 0:
                threshold = max(0.02, 0.05 * (1 - self.best_acc))
                if curr_acc < (self.best_acc - threshold):
                    if (epoch - self.last_rollback_epoch) > self.rollback_cooldown:
                        self.logger.warning(f"⚠️ Caída catastrófica (KNN {curr_acc:.4f} < {self.best_acc:.4f}). ROLLBACK.")
                        self.last_rollback_epoch = epoch
                        # Reset reguladores en rollback
                        self.lr_multiplier = 0.5 
                        self.temp_boost = 1.0
                        self.momentum_boost = 0.0
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
            'history': self.history,
            'lr_multiplier': self.lr_multiplier,
            'momentum_boost': self.momentum_boost,
            'temp_boost': self.temp_boost,
            'ema_loss': self.ema_loss,
            'ema_acc': self.ema_acc,
            'ema_unif': self.ema_unif
        }

    def load_state_dict(self, state):
        self.best_acc = state.get('best_acc', 0.0)
        self.patience = state.get('patience', 0)
        self.warmup_aborted = state.get('warmup_aborted', False)
        self.last_rollback_epoch = state.get('last_rollback_epoch', -999)
        self.history = state.get('history', self.history)
        
        self.lr_multiplier = state.get('lr_multiplier', 1.0)
        self.momentum_boost = state.get('momentum_boost', 0.0)
        self.temp_boost = state.get('temp_boost', 1.0)
        self.ema_loss = state.get('ema_loss', None)
        self.ema_acc = state.get('ema_acc', None)
        self.ema_unif = state.get('ema_unif', None)
