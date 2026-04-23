import torch
import torch.nn.functional as F

def compute_alignment(z1, z2):
    """L_align: Mide el error cuadrático entre vistas positivas."""
    return (z1 - z2).pow(2).sum(dim=1).mean()

def compute_cosine_sims(z1, z2):
    """Calcula similitud de coseno para positivos y negativos (muestra)."""
    with torch.no_grad():
        z1_n = F.normalize(z1.float(), dim=1)
        z2_n = F.normalize(z2.float(), dim=1)
        
        # Similitud de positivos (alignment directo)
        pos_sim = (z1_n * z2_n).sum(dim=1).mean()
        
        # Similitud de negativos (uniformity directa - muestra aleatoria)
        # Usamos un shuffle para comparar vectores que no son pares
        perm = torch.randperm(z1_n.size(0))
        neg_sim = (z1_n * z1_n[perm]).sum(dim=1).mean()
        
    return pos_sim, neg_sim

def compute_uniformity(z, t=2):
    """L_unif: Mide la dispersión global (Wang & Isola)."""
    if z.shape[0] <= 1: return torch.tensor(0.0)
    z = F.normalize(z.float(), dim=1)
    sq_pdist = torch.pdist(z, p=2).pow(2)
    return torch.log(torch.exp(-t * sq_pdist).mean() + 1e-8)

def compute_metrics(q, k):
    """Agregador de métricas para el Trainer."""
    metrics = {}
    with torch.no_grad():
        metrics['alignment'] = compute_alignment(q, k).item()
        metrics['uniformity'] = compute_uniformity(q).item()
        
        pos_sim, neg_sim = compute_cosine_sims(q, k)
        metrics['pos_sim'] = pos_sim.item()
        metrics['neg_sim'] = neg_sim.item()
        
        metrics['std'] = q.std(dim=0).mean().item()
    return metrics

def get_module_stats(module):
    """Calcula estadísticas agregadas de los parámetros de un módulo."""
    stats = {}
    with torch.no_grad():
        all_params = []
        for name, param in module.named_parameters():
            p_data = param.data.float()
            all_params.append(p_data.view(-1))
            if p_data.numel() > 1:
                stats[f"{name}_std"] = p_data.std().item()
            stats[f"{name}_mean"] = p_data.mean().item()
        
        if all_params:
            total_p = torch.cat(all_params)
            stats['total_mean'] = total_p.mean().item()
            stats['total_std'] = total_p.std().item()
            stats['total_norm'] = total_p.norm(2).item()
    return stats