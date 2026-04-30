import math
import logging
import collections
import torch
from enum import IntEnum


class Action(IntEnum):
    CONTINUE = 0
    EARLY_STOP = 1
    ROLLBACK = 2


def deadband(x: float, threshold: float = 0.1) -> float:
    """Zona muerta: suprime señales de control pequeñas para evitar chattering."""
    return 0.0 if abs(x) < threshold else x


def _buffer_entry_to_serializable(entry: dict) -> dict:
    """Convierte una entrada del eval_buffer a tipos serializables (sin tensores GPU).
    
    Crítico para la portabilidad del checkpoint: un tensor en GPU no puede ser
    cargado directamente en un entorno CPU-only (ej. análisis offline, CI).
    """
    out = {}
    for k, v in entry.items():
        if isinstance(v, torch.Tensor):
            out[k] = {'__tensor__': True, 'data': v.detach().cpu().tolist()}
        else:
            out[k] = v
    return out


def _buffer_entry_from_serializable(entry: dict) -> dict:
    """Restaura una entrada del eval_buffer desde su forma serializable."""
    out = {}
    for k, v in entry.items():
        if isinstance(v, dict) and v.get('__tensor__'):
            out[k] = torch.tensor(v['data'])
        else:
            out[k] = v
    return out

class EMA:
    """Exponential Moving Average (media móvil exponencial).

    `update(x)` retorna el nuevo valor de la EMA después de incorporar `x`.
    Si `x is None`, la EMA se mantiene sin cambios (comportamiento pass-through).
    Esto permite pasar valores opcionales sin condiciones extra en el caller.
    Si el primer `x` es None, `value` sigue siendo None hasta que se reciba
    un valor real (inicialización diferida).
    """
    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self.value = None

    def update(self, x):
        if x is None:
            return self.value  # Pass-through: no actualiza si el input es nulo
        if self.value is None:
            self.value = x     # Inicialización diferida: primer valor real
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
        
        # Estado PID Geométrico (Control Continuo Acoplado)
        self.tau = config['moco']['temp_end'] if 'moco' in config else 0.10
        self.alpha = 1.0 - (config['moco']['momentum_base'] if 'moco' in config else 0.999)
        self.current_m = 1.0 - self.alpha
        self.lr_scale = 1.0
        self.lr_step_factor = 1.0
        
        self.eU_ema = 0.0
        self.eD_ema = 0.0
        self.eR_ema = 0.0
        self.I_U = 0.0
        self.steps = 0
        
        self.tau_rank_coef = 0.5
        self.prev_tau = self.tau
        
        self.crisis_counter = 0  # Contador de shocks consecutivos
        # Cuenta cuántas veces el GeoSat tuvo datos suficientes para actuar (eval_buffer lleno).
        # Las primeras N activaciones tienen drift artificialmente alto porque mu se está
        # moviendo desde la inicialización aleatoria — se suprimen las acciones hasta estabilizar.
        self.geosat_activations = 0
        
        # Observadores Suavizados (EMA)
        self.ema_unif = EMA(beta=0.9)
        self.ema_align = EMA(beta=0.9)
        self.ema_pos_sim = EMA(beta=0.9)
        self.ema_neg_sim = EMA(beta=0.9)
        self.ema_ratio_baseline = EMA(beta=0.99)
        self.last_ratio_ema = None
        
        # Detector de saturación geométrica
        self.sat_patience = 0
        self.sat_ema = None
        self.prev_mu = None
        self.prev_eff_rank = None
        self.prev_pos_sim = None
        self.prev_neg_sim = None
        self.prev_delta_pos = 0.0
        self.prev_delta_neg = 0.0
        self.best_geom_score = float('inf')
        self.is_best_geom = False
        # deque(maxlen=3): FIFO automático sin pop(0) explícito (O(1) vs O(n)).
        # El maxlen garantiza que nunca excede 3 entradas sin gestión manual.
        self.eval_buffer: collections.deque = collections.deque(maxlen=3)
        self.max_pos_sim = 0.0

        # Historial acotado: se conservan las últimas HISTORY_MAX_LEN épocas.
        # Esto evita crecimiento lineal del checkpoint en runs largos de Kaggle
        # (ej. 500 épocas × 7 listas × 8 bytes = 28 KB sin cap; con cap: constante).
        self.HISTORY_MAX_LEN: int = 300
        self.history = {
            'loss': [], 'knn_acc': [], 'unif': [], 'align': [],
            'pos_sim': [], 'neg_sim': [], 'ratio_norm': []
        }

    def _trim_history(self) -> None:
        """Recorta el historial si supera HISTORY_MAX_LEN para acotar el tamaño del checkpoint.

        Cuando cualquier lista supera el límite, se truncan TODAS a la mitad del cap
        para amortizar el costo del recorte (O(n) slicing) en lugar de hacerlo cada época.
        De este modo, trim ocurre solo cada HISTORY_MAX_LEN // 2 épocas.
        """
        if any(len(v) > self.HISTORY_MAX_LEN for v in self.history.values()):
            keep = self.HISTORY_MAX_LEN // 2
            for key in self.history:
                self.history[key] = self.history[key][-keep:]

    def get_dynamic_hyperparams(self, step, total_steps, metrics=None):
        # Auto-Tuner Geométrico (PID Latente):
        return self.current_m, self.tau

    def step_epoch(self, epoch, curr_acc, metrics):
        self.steps += 1
        
        u_ema = self.ema_unif.update(metrics['unif'])
        a_ema = self.ema_align.update(metrics['align'])
        # Actualizar EMAs de pos/neg sim (efecto secundario persiste en self.ema_pos_sim.value
        # y se serializa en state_dict). Las variables locales no se usan en este método.
        self.ema_pos_sim.update(metrics['pos_sim'])
        self.ema_neg_sim.update(metrics['neg_sim'])
        
        self.history['loss'].append(metrics['loss'])
        self.history['unif'].append(metrics['unif'])
        self.history['align'].append(metrics['align'])
        self.history['pos_sim'].append(metrics['pos_sim'])
        self.history['neg_sim'].append(metrics['neg_sim'])
        if curr_acc >= 0:
            self.history['knn_acc'].append(curr_acc)
        self._trim_history()  # Evita crecimiento ilimitado del checkpoint

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

        self.is_best_geom = False
        
        # Detector de Saturación Geométrica
        if 'mu' in metrics and 'eff_rank' in metrics:
            self.eval_buffer.append({
                'pos_sim': metrics['pos_sim'],
                'neg_sim': metrics['neg_sim'],
                'eff_rank': metrics['eff_rank'],
                'mu': metrics['mu']
            })
            # deque(maxlen=3) descarta automáticamente la entrada más antigua
            # cuando se supera el límite. No se necesita pop(0) explícito.

            if len(self.eval_buffer) == 3:
                avg_pos  = sum(x['pos_sim']  for x in self.eval_buffer) / 3.0
                avg_neg  = sum(x['neg_sim']  for x in self.eval_buffer) / 3.0
                avg_rank = sum(x['eff_rank'] for x in self.eval_buffer) / 3.0
                # Error 1 fix: sum() Python inicia con 0 (int), lo que falla en multi-GPU
                # cuando el primer tensor no está en el mismo device que 0.
                # torch.stack().mean() es el idiom correcto para promediar tensores.
                avg_mu = torch.stack([x['mu'] for x in self.eval_buffer]).mean(dim=0)

                # Error 3 fix: incrementar aquí, FUERA del bloque `if self.prev_mu is not None`,
                # para contar correctamente la primera activación (cuando prev_mu aún es None).
                self.geosat_activations += 1
                self.max_pos_sim = max(self.max_pos_sim, avg_pos)

                if self.prev_mu is not None:
                    # 1. Drift relativo
                    drift = torch.norm(avg_mu - self.prev_mu).item() / (torch.norm(self.prev_mu).item() + 1e-8)
                    
                    # 2. Diferencias con signo para detectar degradación explícita
                    diff_pos = avg_pos - self.prev_pos_sim
                    diff_neg = avg_neg - self.prev_neg_sim
                    delta_rank_abs = abs(avg_rank - self.prev_eff_rank)
                    
                    delta_pos_abs = abs(diff_pos)
                    delta_neg_abs = abs(diff_neg)
                    
                    # Suavizado temporal de los deltas (Memoria para evitar jitter)
                    delta_pos_smooth = 0.8 * self.prev_delta_pos + 0.2 * delta_pos_abs
                    delta_neg_smooth = 0.8 * self.prev_delta_neg + 0.2 * delta_neg_abs
                    self.prev_delta_pos = delta_pos_smooth
                    self.prev_delta_neg = delta_neg_smooth
                    
                    # 3. Score ponderado (normalizado por escalas empíricas)
                    sat_score = (
                        (delta_pos_smooth / 0.01) + 
                        (delta_neg_smooth / 0.01) + 
                        (drift / 0.05) + 
                        (delta_rank_abs / 0.1)
                    )
                    self.logger.info(f"🧬 GeoSat Score: {sat_score:.4f} | Drift: {drift:.4f} | ΔRank: {delta_rank_abs:.4f}")
                    
                    # ---- PID LATENTE CONTINUO (Auto-Tuner Nivel Research) ----
                    unif_val = self.history['unif'][-1] if self.history['unif'] else 0.0
                    
                    # Errores crudos respecto a los targets ideales
                    eU_raw = (-1.8) - unif_val
                    eD_raw = drift - 0.05
                    eR_raw = delta_rank_abs - 0.1
                    
                    # Normalización empírica (Sigma)
                    eU_norm = eU_raw / 0.5
                    eD_norm = eD_raw / 0.1
                    eR_norm = eR_raw / 0.5
                    
                    # Desacople Feed-Forward Adaptativo: La temperatura alta causa que el rank caiga
                    if epoch > 0 and self.prev_eff_rank is not None:
                        observed_rank_change = avg_rank - self.prev_eff_rank
                        delta_tau = self.tau - self.prev_tau
                        if abs(delta_tau) > 1e-4:
                            empirical_coef = observed_rank_change / delta_tau
                            # Clamp para evitar locuras por ruido
                            empirical_coef = max(min(empirical_coef, 2.0), -2.0)
                            adapt_rate = 0.01 if abs(delta_tau) > 0.01 else 0.001
                            self.tau_rank_coef = (1.0 - adapt_rate) * self.tau_rank_coef + adapt_rate * empirical_coef
                    self.prev_tau = self.tau
                    
                    tau_effect = (self.tau - 0.1) * self.tau_rank_coef
                    corrected_eR = eR_norm + tau_effect
                    
                    # Desacople Temporal (EMAs de Errores)
                    self.eU_ema = 0.7 * self.eU_ema + 0.3 * eU_norm
                    self.eD_ema = 0.9 * self.eD_ema + 0.1 * eD_norm
                    self.eR_ema = 0.85 * self.eR_ema + 0.15 * corrected_eR
                    
                    # Deadband (Zona Muerta) APLICADA DESPUÉS DEL EMA para evitar sesgos nulos
                    # (función definida a nivel de módulo para testabilidad y eficiencia)
                    eU_ctrl = deadband(self.eU_ema, 0.1)
                    eD_ctrl = deadband(self.eD_ema, 0.1)
                    eR_ctrl = deadband(self.eR_ema, 0.05)
                    
                    # Bandera para suspender el PID termostático si hay crisis
                    use_pid_tau = True
                    
                    # --- Eje A: Termostato de Repulsión (Control PI) ---
                    # Integral más pura con un leak muy suave para corregir offsets reales
                    self.I_U += eU_ctrl
                    self.I_U *= 0.99
                    if eU_ctrl * self.I_U < 0:
                        self.I_U *= 0.9 # Descarga rápida si cruza el target
                    self.I_U = max(min(self.I_U, 5.0), -5.0) # Clamp
                    
                    Kp_tau = 0.02
                    Ki_tau = Kp_tau / 50.0
                    # La actualización se realiza al final, si use_pid_tau es True
                    
                    
                    # --- Eje B: Freno de Inercia (Control P sobre Alpha) ---
                    Kp_alpha = 0.005
                    self.alpha -= Kp_alpha * eD_ctrl
                    self.alpha = max(min(self.alpha, 0.05), 1e-5) # Momentum [0.95, 0.99999]
                    self.current_m = 1.0 - self.alpha
                    
                    # --- Eje C: Amortiguador de LR Reversible ---
                    Kp_lr = 0.5
                    old_scale = self.lr_scale
                    
                    # REFLEJO ESPINAL: Control reactivo bypass de emergencia
                    # Umbral adaptativo: Mayor peso al histórico (eR_ema) que al spike instantáneo.
                    # Pesos 0.7/0.3: evita que un eR_ema alto por crisis previas eleve tanto el
                    # umbral que el reflejo quede silenciado ante un spike real (retroalimentación positiva).
                    # Error 4 fix: eR_ema es un error normalizado (puede ser negativo),
                    # mientras que delta_rank_abs es siempre >= 0. Mezclarlos sin abs()
                    # produce un threshold que disminuye cuando el controlador está bajo
                    # presión negativa, silenciando el reflejo de emergencia justo cuando
                    # más se necesita. abs() restaura la coherencia de unidades.
                    adaptive_threshold = max(0.7, 0.7 * abs(self.eR_ema) + 0.3 * delta_rank_abs)
                    
                    if delta_rank_abs > adaptive_threshold:
                        self.crisis_counter += 1
                        emergency_factor = math.exp(-1.5 * (delta_rank_abs - adaptive_threshold))
                        self.lr_scale *= emergency_factor
                        self.logger.warning(f"⚡ Reflejo de Emergencia: Varianza extrema ({delta_rank_abs:.4f} > {adaptive_threshold:.4f}). Hachazo instantáneo al LR.")
                    elif delta_rank_abs < 0.5:
                        self.crisis_counter = max(0, self.crisis_counter - 1)
                        
                    # MODO SUPERVIVENCIA ESTRUCTURAL (2da Ola de Crisis)
                    if self.crisis_counter >= 2:
                        tau_boost = 0.02 * min(2.0, delta_rank_abs)
                        self.logger.warning(f"🚨 SEGUNDA OLA DE CRISIS DETECTADA. Control estructural (Tau +{tau_boost:.4f}).")
                        self.tau += tau_boost
                        self.tau = max(min(self.tau, 0.25), 0.05)
                        # No asfixiamos más al optimizador, el problema es estructural
                        self.lr_scale = max(self.lr_scale, 0.4)
                        use_pid_tau = False
                    elif self.crisis_counter == 0:
                        # ENFRIAMIENTO POST-CRISIS: Disipamos el calor residual suavemente
                        self.tau *= 0.995
                        # ANCLAJE SUAVE: Evitamos el 'creep térmico' a largo plazo
                        tau_target = 0.10
                        self.tau += 0.01 * (tau_target - self.tau)
                        # DESCONGELAMIENTO DE MOMENTUM: Evitamos saturación crónica (m -> 0.99)
                        self.alpha = min(self.alpha + 1e-4, 0.01)
                        
                    if use_pid_tau:
                        self.tau += Kp_tau * eU_ctrl + Ki_tau * self.I_U
                        # Anclaje suave de PID también para evitar drift
                        tau_target = 0.10
                        self.tau += 0.01 * (tau_target - self.tau)
                        # Anti-Windup Dinámico
                        if self.tau >= 0.25 or self.tau <= 0.05:
                            self.I_U *= 0.9 # Leak cuando está saturado
                        self.tau = max(min(self.tau, 0.25), 0.05)
                        
                    # CONTROL CONTINUO: PID latente estándar sobre el rank de varianza.
                    # Asimetría INTENCIONAL: la penalización (riesgo inminente) es 10x más
                    # agresiva que la recuperación (salida del colapso). Esto implementa
                    # el principio de 'primero no hagas daño': se frena rápido, se libera despacio.
                    if eR_ctrl > 0:
                        # Penalización fuerte y rápida (riesgo inminente de colapso)
                        self.lr_scale *= math.exp(-Kp_lr * eR_ctrl)
                    else:
                        # Recuperación lenta y controlada (salida del colapso, 10x más lenta)
                        self.lr_scale *= math.exp(-0.1 * Kp_lr * eR_ctrl)
                        
                    self.lr_scale = max(min(self.lr_scale, 1.0), 0.1)
                    self.lr_step_factor = self.lr_scale / old_scale if old_scale > 0 else 1.0
                    
                    self.logger.info(f"⚙️ PID: Tau={self.tau:.4f} | Mom={self.current_m:.5f} | LR_Scale={self.lr_scale:.3f} | TauRankCoef={self.tau_rank_coef:.3f}")
                    # ----------------------------------------------------------
                    
                    # 4. Señal explícita de degradación con ancla empírica (KNN patience)
                    # Error 2 fix: unif_val ya fue calculado en la línea 247 del bloque PID.
                    # Se elimina el cálculo duplicado para evitar inconsistencias futuras
                    # si alguno de los dos se modifica sin actualizar el otro.
                    # Error 3 fix: geosat_activations ya se incrementó arriba (fuera del
                    # bloque prev_mu). Se elimina el incremento redundante de aquí.

                    # A. Degradación Extrema: Rango explotando o colapso de la distribución.
                    #
                    # SEMÁNTICA DE UNIFORMIDAD (crítico para no invertir la lógica):
                    #   - Uniformidad MÁS negativa = distribución MÁS uniforme = MEJOR.
                    #   - U ≈ 0   → COLAPSO (todos los embeddings en el mismo punto).
                    #   - U = -2.4 → Excelente distribución esférica (NO es degradación).
                    # Por tanto, el umbral de colapso es unif_val > -0.3 (cerca de cero),
                    # NO unif_val < -2.2 (que detectaría exactamente lo contrario: éxito).
                    #
                    # Guard de fase inicial: las primeras 2 activaciones del GeoSat tienen
                    # drift alto porque mu se mueve desde la inicialización aleatoria.
                    # No actuamos hasta tener al menos 3 activaciones con datos estables.
                    _geosat_mature = self.geosat_activations >= 3
                    
                    if delta_rank_abs > 0.6 or unif_val > -0.3:
                        self.logger.warning(f"⚠️ DEGRADACIÓN SEVERA DETECTADA: ΔRank={delta_rank_abs:.4f}, U={unif_val:.2f}")
                        if not _geosat_mature:
                            self.logger.warning(f"   ↳ GeoSat en fase inicial (activación {self.geosat_activations}/3). Suprimiendo acción preventiva.")
                        elif self.patience >= 1:
                            # Condicionamos el Rollback: Solo se hace Rollback preventivo 
                            # si el optimizador ya frenó y el manifold no se arregló tras varios shocks
                            if self.lr_scale < 0.3 and self.crisis_counter >= 2:
                                self.logger.warning("→ Confirmado por caída de KNN en crisis estructural prolongada. Iniciando ROLLBACK preventivo.")
                                return Action.ROLLBACK
                            else:
                                self.logger.warning("🛡️ Modo Supervivencia activo. Dando tiempo al manifold antes de forzar Rollback.")
                    
                    # B. Degradación Moderada (over-spreading leve)
                    if self.patience >= 2:
                        tau_pos, tau_neg = 1e-3, 1e-3
                        if diff_pos < -tau_pos and diff_neg > tau_neg:
                            self.logger.warning(f"⚠️ DEGRADACIÓN GEOMÉTRICA (over-spreading leve): ΔP={diff_pos:.4f}, ΔN={diff_neg:.4f}")
                            self.logger.warning("→ Confirmado por KNN empírico (patience >= 2). Iniciando ROLLBACK.")
                            return Action.ROLLBACK
                    
                    # 5. Detección de "Sweet Spot" Geométrico (Mejor cristalización con calidad dinámica)
                    if sat_score < self.best_geom_score and not is_warmup and avg_pos > 0.8 * self.max_pos_sim:
                        self.best_geom_score = sat_score
                        self.is_best_geom = True
                    
                    # 6. Baseline adaptativo (EMA de sat_score)
                    if self.sat_ema is None:
                        self.sat_ema = sat_score
                    else:
                        self.sat_ema = 0.9 * self.sat_ema + 0.1 * sat_score
                        
                    # Error 5 fix: en la segunda activación sat_ema aún tiene memoria del
                    # valor inicial alto (primera activación), lo que puede generar un falso
                    # positivo si sat_score mejora drásticamente entre la 1ª y la 2ª activación.
                    # Se requieren >= 2 activaciones para que la EMA tenga contexto suficiente.
                    if self.geosat_activations >= 2 and sat_score < 0.5 * self.sat_ema:
                        self.sat_patience += 1
                        self.logger.info(f"🧊 Espacio latente congelándose... (Paciencia Geo {self.sat_patience}/3)")
                    else:
                        self.sat_patience = 0
                        
                    if self.sat_patience >= 3:
                        self.logger.info("🧊 Embedding saturado geométricamente.")
                        if self.patience >= 2:
                            self.logger.info("→ Confirmado por KNN empírico (patience >= 2). EARLY STOP")
                            return Action.EARLY_STOP
                        else:
                            self.logger.info("→ Ignorado (KNN sigue mejorando)")
                        
                self.prev_mu = avg_mu
                self.prev_eff_rank = avg_rank
                self.prev_pos_sim = avg_pos
                self.prev_neg_sim = avg_neg
            else:
                self.prev_mu = self.eval_buffer[0]['mu']
                self.prev_eff_rank = self.eval_buffer[0]['eff_rank']
                self.prev_pos_sim = self.eval_buffer[0]['pos_sim']
                self.prev_neg_sim = self.eval_buffer[0]['neg_sim']

        return Action.CONTINUE

    def state_dict(self) -> dict:
        """Serializa el estado completo del controlador a tipos Python nativos.
        
        Invariantes de seguridad:
          - Todos los torch.Tensor en eval_buffer y prev_mu se convierten a listas
            para garantizar portabilidad CPU/GPU y compatibilidad con weights_only=True.
          - Nuevos campos añadidos aquí DEBEN tener un .get() con default en load_state_dict
            para mantener retrocompatibilidad con checkpoints anteriores.
        """
        # Serializar prev_mu de forma segura (puede ser tensor o None)
        prev_mu_safe = None
        if self.prev_mu is not None:
            if isinstance(self.prev_mu, torch.Tensor):
                prev_mu_safe = {'__tensor__': True, 'data': self.prev_mu.detach().cpu().tolist()}
            else:
                prev_mu_safe = self.prev_mu  # Ya es lista (checkpoint viejo)

        return {
            # --- Estado de entrenamiento ---
            'best_acc': self.best_acc,
            'patience': self.patience,
            'warmup_aborted': self.warmup_aborted,   # FIX #3: persiste post-resume
            'temp_adj': self.temp_adj,
            'history': self.history,
            # --- Observadores EMA ---
            'ema_unif': self.ema_unif.value,
            'ema_align': self.ema_align.value,
            'ema_pos_sim': self.ema_pos_sim.value,
            'ema_neg_sim': self.ema_neg_sim.value,
            'ema_ratio_baseline': self.ema_ratio_baseline.value,
            'last_ratio_ema': self.last_ratio_ema,
            # --- Detector de saturación geométrica ---
            'sat_patience': self.sat_patience,
            'sat_ema': self.sat_ema,
            'prev_mu': prev_mu_safe,                  # FIX #1: tensor → lista serializable
            'prev_eff_rank': self.prev_eff_rank,
            'prev_pos_sim': self.prev_pos_sim,
            'prev_neg_sim': self.prev_neg_sim,
            'prev_delta_pos': self.prev_delta_pos,
            'prev_delta_neg': self.prev_delta_neg,
            'best_geom_score': self.best_geom_score,
            # deque se serializa como lista: los tipos nativos de Python son
            # compatibles con pickle (torch.save) y con weights_only=True.
            'eval_buffer': [_buffer_entry_to_serializable(e) for e in self.eval_buffer],  # FIX #1
            'max_pos_sim': self.max_pos_sim,
            # --- Estado del controlador PID ---
            'tau': self.tau,
            'alpha': self.alpha,
            'current_m': self.current_m,
            'lr_scale': self.lr_scale,
            'lr_step_factor': self.lr_step_factor,
            'eU_ema': self.eU_ema,
            'eD_ema': self.eD_ema,
            'eR_ema': self.eR_ema,
            'I_U': self.I_U,
            'steps': self.steps,
            'tau_rank_coef': self.tau_rank_coef,
            'prev_tau': self.prev_tau,
            'crisis_counter': self.crisis_counter,
            'geosat_activations': self.geosat_activations,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restaura el estado completo del controlador desde un checkpoint.
        
        Retrocompatibilidad: todos los campos usan .get() con valores por defecto
        para que checkpoints de versiones anteriores carguen sin errores.
        """
        # --- Estado de entrenamiento ---
        self.best_acc = state.get('best_acc', 0.0)
        self.patience = state.get('patience', 0)
        self.warmup_aborted = state.get('warmup_aborted', False)  # FIX #3
        self.temp_adj = state.get('temp_adj', 0.0)
        if 'history' in state:
            for k, v in state['history'].items():
                if k in self.history:
                    self.history[k] = v

        # --- Observadores EMA ---
        self.ema_unif.value = state.get('ema_unif')
        self.ema_align.value = state.get('ema_align')
        self.ema_pos_sim.value = state.get('ema_pos_sim')
        self.ema_neg_sim.value = state.get('ema_neg_sim')
        self.ema_ratio_baseline.value = state.get('ema_ratio_baseline')
        self.last_ratio_ema = state.get('last_ratio_ema')

        # --- Detector de saturación geométrica ---
        self.sat_patience = state.get('sat_patience', 0)
        self.sat_ema = state.get('sat_ema', None)

        # FIX #1: Restaurar prev_mu soportando formato nuevo (dict serializado)
        # y formato legado (tensor directo o None) para retrocompatibilidad.
        raw_mu = state.get('prev_mu', None)
        if raw_mu is None:
            self.prev_mu = None
        elif isinstance(raw_mu, dict) and raw_mu.get('__tensor__'):
            self.prev_mu = torch.tensor(raw_mu['data'])
        elif isinstance(raw_mu, torch.Tensor):
            self.prev_mu = raw_mu  # Checkpoint legado con tensor directo
        else:
            self.prev_mu = raw_mu  # Lista plana u otro tipo legado

        self.prev_eff_rank = state.get('prev_eff_rank', None)
        self.prev_pos_sim = state.get('prev_pos_sim', None)
        self.prev_neg_sim = state.get('prev_neg_sim', None)
        self.prev_delta_pos = state.get('prev_delta_pos', 0.0)
        self.prev_delta_neg = state.get('prev_delta_neg', 0.0)
        self.best_geom_score = state.get('best_geom_score', float('inf'))
        self.max_pos_sim = state.get('max_pos_sim', 0.0)

        # FIX #1: Restaurar eval_buffer soportando formato nuevo y legado.
        # Se reconstruye siempre como deque(maxlen=3) independientemente de
        # cómo estaba guardado (lista plana, lista de dicts serializados).
        raw_buffer = state.get('eval_buffer', [])
        self.eval_buffer = collections.deque(
            (_buffer_entry_from_serializable(e) if isinstance(e, dict) else e
             for e in raw_buffer),
            maxlen=3
        )

        # --- Estado del controlador PID ---
        self.tau = state.get('tau', self.tau)
        self.alpha = state.get('alpha', self.alpha)
        self.current_m = state.get('current_m', self.current_m)
        self.lr_scale = state.get('lr_scale', self.lr_scale)
        self.lr_step_factor = state.get('lr_step_factor', 1.0)
        self.eU_ema = state.get('eU_ema', 0.0)
        self.eD_ema = state.get('eD_ema', 0.0)
        self.eR_ema = state.get('eR_ema', 0.0)
        self.I_U = state.get('I_U', 0.0)
        self.steps = state.get('steps', 0)
        self.tau_rank_coef = state.get('tau_rank_coef', 0.5)
        self.prev_tau = state.get('prev_tau', self.tau)
        self.crisis_counter = state.get('crisis_counter', 0)
        # Retrocompat: checkpoints anteriores no tienen este campo; por defecto 0
        # (el guard se re-activa correctamente durante el resume).
        self.geosat_activations = state.get('geosat_activations', 0)

        # Cold-start penalty: atenúa señales de error post-resume para evitar
        # que el controlador tome decisiones agresivas con estado EMA desactualizado.
        warmup_factor = min(1.0, self.steps / 100.0) if self.steps > 0 else 1.0
        self.eU_ema *= warmup_factor
        self.eD_ema *= warmup_factor
        self.eR_ema *= warmup_factor

