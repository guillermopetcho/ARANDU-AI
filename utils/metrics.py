import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Límite para pdist: a partir de este N, se usa subsample aleatorio.
# Con N=512: 512*511/2 = 130,816 pares → memoria y tiempo controlados.
# Con el batch completo de 384: 384*383/2 = 73,536 pares → ok, pero se
# mantiene el cap para ser robusto ante batches grandes con multi-crop.
# ---------------------------------------------------------------------------
_UNIFORMITY_MAX_SAMPLES = 512


def compute_alignment(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """L_align: Error cuadrático medio entre vistas positivas (Wang & Isola).

    Mide cuánto se acercan las representaciones de dos vistas de la misma imagen.
    Valor ideal: 0. Valores altos indican colapso inverso (repulsión de positivos).
    """
    return (z1 - z2).pow(2).sum(dim=1).mean()


def compute_cosine_sims(
    z1: torch.Tensor, z2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Similitud coseno media para pares positivos y negativos (muestra).

    Los negativos se estiman permutando z1 contra sí mismo para comparar
    vectores que no son pares positivos, sin necesidad de una queue separada.
    """
    with torch.no_grad():
        z1_n = F.normalize(z1.float(), dim=1)
        z2_n = F.normalize(z2.float(), dim=1)

        # Similitud de positivos (alignment directo)
        pos_sim = (z1_n * z2_n).sum(dim=1).mean()

        # Similitud de negativos: shuffle aleatorio para evitar pares triviales
        perm = torch.randperm(z1_n.size(0), device=z1_n.device)
        neg_sim = (z1_n * z1_n[perm]).sum(dim=1).mean()

    return pos_sim, neg_sim


def compute_uniformity(
    z: torch.Tensor,
    t: float = 2.0,
    max_samples: int = _UNIFORMITY_MAX_SAMPLES,
) -> torch.Tensor:
    """L_unif: Uniformidad de la distribución en la hiperesfera (Wang & Isola, 2020).

    Fórmula: log( E[exp(-t * ||z_i - z_j||²)] )
    Valor ideal: -∞ (distribución perfectamente uniforme).
    Valores cercanos a 0 indican colapso (todos los embeddings en el mismo punto).

    Complejidad sin cap: O(N²) pares, O(N²) memoria → puede explotar en batches grandes.
    Con cap: O(max_samples²) constante, independiente del batch size.

    Estabilidad numérica:
      - Se usa logsumexp en lugar de log(exp(...).mean()) para evitar overflow/underflow
        en FP16 cuando t * ||z||² es grande.
      - z se normaliza a float32 antes del cálculo, incluso si la entrada es BF16/FP16.

    Args:
        z:           Tensor [N, D] de embeddings (cualquier dtype).
        t:           Temperatura del kernel gaussiano (default: 2.0).
        max_samples: Cap máximo de filas para pdist. Si N > max_samples, se toma
                     un subconjunto aleatorio sin reemplazo. (default: 512)

    Returns:
        Escalar torch con el valor de L_unif.
    """
    if z.shape[0] <= 1:
        return torch.tensor(0.0, device=z.device)

    z = F.normalize(z.float(), dim=1)

    # Subsample aleatorio si el batch supera el umbral.
    # randperm garantiza "sin reemplazo", lo que preserva la distribución empírica.
    if z.shape[0] > max_samples:
        perm = torch.randperm(z.shape[0], device=z.device)[:max_samples]
        z = z[perm]

    # pdist calcula ||z_i - z_j||² para todos los pares i < j → [N*(N-1)/2]
    sq_pdist = torch.pdist(z, p=2).pow(2)

    # Estabilidad numérica: logsumexp(x) - log(n) = log(mean(exp(x)))
    # Equivalente a log(exp(-t*d²).mean()) pero sin overflow
    n_pairs = sq_pdist.shape[0]
    return torch.logsumexp(-t * sq_pdist, dim=0) - math.log(max(n_pairs, 1))


def compute_metrics(q: torch.Tensor, k: torch.Tensor) -> dict:
    """Agregador de métricas instantáneas para el Trainer.

    Se llama en cada step del epoch. Todas las operaciones son O(batch_size)
    excepto uniformity que está acotada por _UNIFORMITY_MAX_SAMPLES.
    """
    metrics = {}
    with torch.no_grad():
        metrics['alignment'] = compute_alignment(q, k).item()
        metrics['uniformity'] = compute_uniformity(q).item()

        pos_sim, neg_sim = compute_cosine_sims(q, k)
        metrics['pos_sim'] = pos_sim.item()
        metrics['neg_sim'] = neg_sim.item()

        metrics['std'] = q.std(dim=0).mean().item()
    return metrics


def get_module_stats(module: torch.nn.Module) -> dict:
    """Estadísticas agregadas de los parámetros de un módulo.

    Calcula media, std y norma tanto por capa como globales.
    Usa estadísticas online (Welford) para las globales en lugar de concatenar
    todos los parámetros en un tensor masivo (O(1) memoria en lugar de O(P)).
    """
    stats = {}
    with torch.no_grad():
        # Estadísticas globales via Welford online (sin concatenar)
        total_count = 0
        total_mean = 0.0
        total_M2 = 0.0    # Suma de diferencias al cuadrado (para varianza)
        total_sum_sq = 0.0  # Para la norma L2

        for name, param in module.named_parameters():
            p_data = param.data.float().view(-1)
            n = p_data.numel()

            if n > 1:
                stats[f"{name}_std"]  = p_data.std().item()
            stats[f"{name}_mean"] = p_data.mean().item()

            # Actualización de Welford para media y varianza globales
            # Referencia: Knuth, TAOCP Vol. 2 §4.2.2
            for x in [p_data.mean().item()]:  # actualizar con la media del tensor
                total_count += n
                delta = p_data.sum().item() - total_mean * n
                total_mean += delta / total_count
                total_M2 += p_data.var(unbiased=False).item() * n

            total_sum_sq += p_data.pow(2).sum().item()

        if total_count > 0:
            stats['total_mean'] = total_mean
            stats['total_std']  = math.sqrt(total_M2 / max(total_count - 1, 1))
            stats['total_norm'] = math.sqrt(total_sum_sq)

    return stats