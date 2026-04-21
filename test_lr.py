import torch
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

model = nn.Linear(10, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0225)
# Set initial_lr like the code does
for pg in optimizer.param_groups:
    pg['initial_lr'] = 0.0225

w_steps = 1765
t_steps = 70600

schedulers=[
    LinearLR(optimizer, start_factor=0.01, total_iters=w_steps),
    CosineAnnealingLR(optimizer, T_max=max(1, t_steps - w_steps))
]
scheduler = SequentialLR(optimizer, schedulers=schedulers, milestones=[w_steps])

for step in range(353 * 10): # 10 epochs
    scheduler.step()
    if step == 353 * 8 - 1:
        print(f"Epoch 8 LR: {optimizer.param_groups[0]['lr']}")
    if step == 353 * 9 - 1:
        print(f"Epoch 9 LR: {optimizer.param_groups[0]['lr']}")
