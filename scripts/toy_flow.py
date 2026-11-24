"""Flow matching on the smallest problem that shows why it exists.

One "situation", two correct actions (-1 and +1), like a gear you may grasp from the left or from the right.
A regression head learns the average (0, a grasp nobody showed). A flow-matching head lands on one of the two.
The update rule is the one SmolVLA uses: x_t = t*noise + (1-t)*action, target = noise - action, 10 Euler steps.

    uv run --extra robot python scripts/toy_flow.py
"""

import torch
from torch import nn

torch.manual_seed(0)
actions = torch.where(torch.rand(4096, 1) < 0.5, -1.0, 1.0) + 0.05 * torch.randn(4096, 1)  # the demonstrations


def mlp(n_in):
    return nn.Sequential(nn.Linear(n_in, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))


# 1. regression: the observation is always the same, so the network is a single learned number
regress = mlp(1)
opt = torch.optim.Adam(regress.parameters(), 1e-3)
obs = torch.zeros_like(actions)
for _ in range(2000):
    loss = ((regress(obs) - actions) ** 2).mean()
    opt.zero_grad(), loss.backward(), opt.step()

# 2. flow matching: learn the velocity that points from the action to the noise
velocity = mlp(2)  # input: (x_t, t)
opt = torch.optim.Adam(velocity.parameters(), 1e-3)
for _ in range(5000):
    a = actions[torch.randint(len(actions), (256,))]
    noise, t = torch.randn_like(a), torch.rand_like(a)
    x_t = t * noise + (1 - t) * a
    loss = ((velocity(torch.cat([x_t, t], 1)) - (noise - a)) ** 2).mean()
    opt.zero_grad(), loss.backward(), opt.step()

with torch.no_grad():
    x = torch.randn(1000, 1)  # start from noise
    for step in range(10):  # t = 1.0, 0.9, ... 0.1
        t = torch.full_like(x, 1 - step / 10)
        x = x - 0.1 * velocity(torch.cat([x, t], 1))
    print(f"regression predicts      {regress(obs[:1]).item():+.2f}   (the average: closes on nothing)")
    print(f"flow matching, 1000 runs {(x < 0).float().mean().item():.0%} near -1, {(x > 0).float().mean().item():.0%} near +1")
    print(f"  mean |x| = {x.abs().mean().item():.2f}, samples between -0.5 and 0.5: {(x.abs() < 0.5).float().mean().item():.1%}")
