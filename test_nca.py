"""Self-check for nca.py: python3 test_nca.py"""
import math

import torch

import nca

# seed: 72x72x16, only the center cell's ch 3..15 are 1
s = nca.make_seed()
assert s.shape == (1, 16, 72, 72)
assert s.sum() == 13 and (s[0, 3:, 36, 36] == 1).all() and (s[0, :3] == 0).all()

model = nca.CA()  # W2 is zero-initialised

# Sobel x on a horizontal ramp (value = column) -> gradient 1 in the interior, 0 along y; rotation by 90 deg swaps
ramp = torch.arange(72.0).expand(1, 16, 72, 72)
dx, dy = model.sobel(ramp, 0.0)
assert torch.allclose(dx[..., 1:-1, 1:-1], torch.ones(1)) and dy[..., 1:-1, 1:-1].abs().max() == 0
dx, dy = model.sobel(ramp, math.pi / 2)
assert torch.allclose(dy[..., 1:-1, 1:-1], torch.ones(1)) and dx[..., 1:-1, 1:-1].abs().max() < 1e-6

# an isolated cell with alpha < 0.1 and no live neighbours is zeroed
x = torch.zeros(1, 16, 72, 72)
x[0, :, 10, 10] = 0.05
assert model(x, fire_rate=1.0).abs().max() == 0

# zero W2 -> a step leaves the (alive) seed exactly unchanged
assert torch.equal(model(s, fire_rate=1.0), s)

# export flatten matches the layout the demo expects: index = (row*W + col)*16 + ch, and W1[o][i] row-major;
# 9 significant digits round-trip float32 exactly
x = torch.rand(1, 16, 72, 72)
f = torch.tensor(nca.flat(nca.hwc(x), 9))
for r, c, ch in [(0, 0, 0), (5, 70, 3), (71, 71, 15), (36, 12, 9)]:
    assert f[(r * 72 + c) * 16 + ch] == x[0, ch, r, c]
w1 = torch.tensor(nca.weights(model, 9)['w1'])
assert len(w1) == 128 * 48 and w1[5 * 48 + 7] == model.w1.weight[5, 7, 0, 0]

print('test_nca.py: all checks passed')
