import torch

def build_scheduler(opt, w_steps, t_steps, c_step=0, skip=False):
    if skip:
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, t_steps - c_step))
    return torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=w_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, t_steps - w_steps))
        ], milestones=[w_steps]
    )

@torch.no_grad()
def momentum_update(model_q, model_k, m):
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)