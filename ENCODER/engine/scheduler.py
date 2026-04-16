import math
import torch

@torch.no_grad()
def momentum_update(model_q, model_k, m):
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)

def get_dynamic_hyperparams(step, total_steps, config, current_unif):
    """
    Calcula hiperparámetros dinámicos. 
    🔥 FIX: Temperatura Warmup y Auto-Regulación.
    """
    # 1. Momentum Base (Cosenoidal estándar)
    m_base = config['moco']['momentum_base']
    momentum = 1 - (1 - m_base) * (math.cos(math.pi * step / max(1, total_steps)) + 1) / 2

    # 2. Temperature Warmup (Reemplaza el hack loss=0)
    warmup_steps = config['moco'].get('temp_warmup_steps', 100)
    if step < warmup_steps:
        # Temperatura artificialmente alta al inicio para suavizar gradientes
        temp = 0.5 - (0.5 - config['moco']['temp_start']) * (step / warmup_steps)
    else:
        # Decaimiento térmico normal
        progress = min(max((step - warmup_steps) / max(1, total_steps - warmup_steps), 0.0), 1.0)
        temp = config['moco']['temp_end'] + (config['moco']['temp_start'] - config['moco']['temp_end']) * 0.5 * (1 + math.cos(math.pi * progress))

    # 3. Auto-regulación del Sistema basada en Uniformidad
    # Si la uniformidad es muy baja (muy negativa, e.g. < -4.0), los embeddings están colapsando.
    temp_boost = 1.0
    if current_unif < -4.0:
        temp_boost = 1.15  # Forzamos repulsión
        momentum = min(momentum * 0.99, 0.99) # Hacemos que la cola se actualice más rápido para purgar basura
    
    return momentum, max(temp * temp_boost, 0.05)
