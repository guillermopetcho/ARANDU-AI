import math
import logging
import torch
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
        
        self.crisis_counter = 0 # Contador de shocks consecutivos
        
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
        self.eval_buffer = []
        self.max_pos_sim = 0.0
        
        # Historial
        self.history = {
            'loss': [], 'knn_acc': [], 'unif': [], 'align': [], 
            'pos_sim': [], 'neg_sim': [], 'ratio_norm': []
        }

    def get_dynamic_hyperparams(self, step, total_steps, metrics=None):
        # Auto-Tuner Geométrico (PID Latente):
        return self.current_m, self.tau

    def step_epoch(self, epoch, curr_acc, metrics):
        self.steps += 1
        
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

        self.is_best_geom = False
        
        # Detector de Saturación Geométrica
        if 'mu' in metrics and 'eff_rank' in metrics:
            self.eval_buffer.append({
                'pos_sim': metrics['pos_sim'],
                'neg_sim': metrics['neg_sim'],
                'eff_rank': metrics['eff_rank'],
                'mu': metrics['mu']
            })
            if len(self.eval_buffer) > 3:
                self.eval_buffer.pop(0)

            if len(self.eval_buffer) == 3:
                avg_pos = sum(x['pos_sim'] for x in self.eval_buffer) / 3.0
                avg_neg = sum(x['neg_sim'] for x in self.eval_buffer) / 3.0
                avg_rank = sum(x['eff_rank'] for x in self.eval_buffer) / 3.0
                avg_mu = sum(x['mu'] for x in self.eval_buffer) / 3.0
                
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
                    def deadband(x, threshold=0.1):
                        return 0.0 if abs(x) < threshold else x
                    
                    eU_ctrl = deadband(self.eU_ema, 0.1)
                    eD_ctrl = deadband(self.eD_ema, 0.1)
                    eR_ctrl = deadband(self.eR_ema, 0.05)
                    
                    # --- Eje A: Termostato de Repulsión (Control PI) ---
                    # Integral más pura con un leak muy suave para corregir offsets reales
                    self.I_U += eU_ctrl
                    self.I_U *= 0.99
                    if eU_ctrl * self.I_U < 0:
                        self.I_U *= 0.9 # Descarga rápida si cruza el target
                    self.I_U = max(min(self.I_U, 5.0), -5.0) # Clamp
                    
                    Kp_tau = 0.02
                    Ki_tau = Kp_tau / 50.0
                    self.tau += Kp_tau * eU_ctrl + Ki_tau * self.I_U
                    
                    # Anti-Windup Dinámico
                    if self.tau >= 0.25 or self.tau <= 0.05:
                        self.I_U *= 0.9 # Leak cuando está saturado
                    
                    self.tau = max(min(self.tau, 0.25), 0.05)
                    
                    # --- Eje B: Freno de Inercia (Control P sobre Alpha) ---
                    Kp_alpha = 0.005
                    self.alpha -= Kp_alpha * eD_ctrl
                    self.alpha = max(min(self.alpha, 0.05), 1e-5) # Momentum [0.95, 0.99999]
                    self.current_m = 1.0 - self.alpha
                    
                    # --- Eje C: Amortiguador de LR Reversible ---
                    Kp_lr = 0.5
                    old_scale = self.lr_scale
                    
                    # REFLEJO ESPINAL: Control reactivo bypass de emergencia
                    if delta_rank_abs > 0.7:
                        self.crisis_counter += 1
                        emergency_factor = math.exp(-1.5 * (delta_rank_abs - 0.7))
                        self.lr_scale *= emergency_factor
                        self.logger.warning(f"⚡ Reflejo de Emergencia: Varianza extrema ({delta_rank_abs:.4f}). Hachazo instantáneo al LR.")
                    else:
                        self.crisis_counter = 0
                        
                    # MODO SUPERVIVENCIA ESTRUCTURAL (2da Ola de Crisis)
                    if self.crisis_counter >= 2:
                        self.logger.warning("🚨 SEGUNDA OLA DE CRISIS DETECTADA. Cambiando a control estructural (subiendo Tau).")
                        self.tau += 0.03
                        self.tau = max(min(self.tau, 0.25), 0.05)
                        # No asfixiamos más al optimizador, el problema es estructural
                        self.lr_scale = max(self.lr_scale, 0.4)
                        
                    # CONTROL CONTINUO: PID latente estándar
                    if eR_ctrl > 0:
                        # Penalización fuerte y rápida (riesgo inminente)
                        self.lr_scale *= math.exp(-Kp_lr * eR_ctrl)
                    else:
                        # Recuperación lenta y controlada (salida del colapso)
                        self.lr_scale *= math.exp(-0.1 * Kp_lr * eR_ctrl)
                        
                    self.lr_scale = max(min(self.lr_scale, 1.0), 0.1)
                    self.lr_step_factor = self.lr_scale / old_scale if old_scale > 0 else 1.0
                    
                    self.logger.info(f"⚙️ PID: Tau={self.tau:.4f} | Mom={self.current_m:.5f} | LR_Scale={self.lr_scale:.3f} | TauRankCoef={self.tau_rank_coef:.3f}")
                    # ----------------------------------------------------------
                    
                    # 4. Señal explícita de degradación con ancla empírica (KNN patience)
                    unif_val = self.history['unif'][-1] if self.history['unif'] else 0.0
                    
                    # A. Degradación Extrema: Rango explotando o repulsión masiva.
                    # No esperamos 2 épocas (ahorramos 50 min de Kaggle) si la inestabilidad es obvia.
                    if delta_rank_abs > 0.6 or unif_val < -2.2:
                        self.logger.warning(f"⚠️ DEGRADACIÓN SEVERA DETECTADA: ΔRank={delta_rank_abs:.4f}, U={unif_val:.2f}")
                        if self.patience >= 1:
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
                        
                    if sat_score < 0.5 * self.sat_ema:
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
            'last_ratio_ema': self.last_ratio_ema,
            'sat_patience': self.sat_patience,
            'sat_ema': self.sat_ema,
            'prev_mu': self.prev_mu,
            'prev_eff_rank': self.prev_eff_rank,
            'prev_pos_sim': self.prev_pos_sim,
            'prev_neg_sim': self.prev_neg_sim,
            'prev_delta_pos': self.prev_delta_pos,
            'prev_delta_neg': self.prev_delta_neg,
            'best_geom_score': self.best_geom_score,
            'eval_buffer': self.eval_buffer,
            'max_pos_sim': self.max_pos_sim,
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
            'crisis_counter': getattr(self, 'crisis_counter', 0)
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
        self.sat_patience = state.get('sat_patience', 0)
        self.sat_ema = state.get('sat_ema', None)
        self.prev_mu = state.get('prev_mu', None)
        self.prev_eff_rank = state.get('prev_eff_rank', None)
        self.prev_pos_sim = state.get('prev_pos_sim', None)
        self.prev_neg_sim = state.get('prev_neg_sim', None)
        self.prev_delta_pos = state.get('prev_delta_pos', 0.0)
        self.prev_delta_neg = state.get('prev_delta_neg', 0.0)
        self.best_geom_score = state.get('best_geom_score', float('inf'))
        self.eval_buffer = state.get('eval_buffer', [])
        self.max_pos_sim = state.get('max_pos_sim', 0.0)
        
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
        
        # Evitar decisiones agresivas post-resume (Cold start penalty)
        warmup_factor = min(1.0, self.steps / 100.0) if self.steps > 0 else 1.0
        self.eU_ema *= warmup_factor
        self.eD_ema *= warmup_factor
        self.eR_ema *= warmup_factor

