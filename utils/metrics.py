import torch

def compute_alignment(q, k):
    """Mide qué tan cerca están las representaciones de la misma imagen (MSE)."""
    return (q - k).pow(2).sum(dim=1).mean()

def compute_uniformity(q, t=2):
    """Mide qué tan bien distribuidos están los vectores en la hiperesfera (Wang & Isola)."""
    if q.shape[0] <= 1: return torch.tensor(0.0)
    # Submuestreo para eficiencia O(N)
    sample_size = min(64, q.shape[0])
    q_sample = q[torch.randperm(q.shape[0])[:sample_size]]
    # Log-Sum-Exp trick para estabilidad numérica
    sq_dist = torch.pdist(q_sample.float(), p=2).pow(2)
    return torch.log(torch.exp(-t * sq_dist).mean() + 1e-8)

def compute_metrics(q, k):
    metrics = {}
    with torch.no_grad():
        metrics['alignment'] = (q - k).pow(2).sum(dim=1).mean().item()
        
        if q.shape[0] > 1:
            sample_size = min(64, q.shape[0])
            sample_q = q[torch.randperm(q.shape[0])[:sample_size]]
            u = torch.exp(-2 * torch.pdist(sample_q.float(), p=2).pow(2))
            metrics['uniformity'] = torch.log(u.mean() + 1e-8).item()
        else:
            metrics['uniformity'] = 0.0
            
        metrics['std'] = q.std(dim=0).mean().item()
    return metrics

def get_module_stats(module):
    """Calcula estadísticas agregadas de los parámetros de un módulo (mean, std, norm)."""
    stats = {}
    with torch.no_grad():
        all_params = []
        for name, param in module.named_parameters():
            p_data = param.data.float()
            all_params.append(p_data.view(-1))
            # Estadísticas por capa individual (simplificado)
            if p_data.numel() > 1:
                stats[f"{name}_std"] = p_data.std().item()
            stats[f"{name}_mean"] = p_data.mean().item()
        
        if all_params:
            total_p = torch.cat(all_params)
            stats['total_mean'] = total_p.mean().item()
            stats['total_std'] = total_p.std().item()
            stats['total_norm'] = total_p.norm(2).item()
    return stats