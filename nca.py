"""Growing Neural Cellular Automata (Mordvintsev et al., Distill 2020): model, training, export.

demo/nca.js implements the exact same step in JavaScript (checked by demo/parity_test.mjs).

  python3 nca.py train --target lizard --regime regenerating --iters 3000 --out checkpoints/lizard.pt
  python3 nca.py export      # -> demo/weights.js + demo/parity_fixture.json
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).parent
CH, SIZE, PAD = 16, 40, 16
GRID = SIZE + 2 * PAD  # 72
POOL, BATCH = 1024, 8
SOBEL_X = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
# name -> (label, emoji) for demo/weights.js; checkpoints/<name>.pt
MODELS = {
    'lizard': ('Lizard', '\U0001F98E'),
    'lizard_persistent': ('Lizard (persistent)', '\U0001F98E'),
    'lizard_growing': ('Lizard (growing)', '\U0001F98E'),
    'butterfly': ('Butterfly', '\U0001F98B'),
    'mushroom': ('Mushroom', '\U0001F344'),
}


def load_target(name):
    """targets/<name>.png -> [4, 72, 72] RGBA, RGB premultiplied by alpha."""
    img = Image.open(ROOT / 'targets' / f'{name}.png').convert('RGBA')
    img = img.resize((SIZE, SIZE), Image.LANCZOS)  # Pillow resamples RGBA premultiplied, so no edge fringes
    a = torch.tensor(np.asarray(img), dtype=torch.float32) / 255  # [H, W, 4]
    a[..., :3] *= a[..., 3:]
    return F.pad(a.permute(2, 0, 1), (PAD,) * 4)


def make_seed(n=1):
    """[n, 16, 72, 72] zeros with ch 3..15 = 1 at the center cell."""
    x = torch.zeros(n, CH, GRID, GRID)
    x[:, 3:, GRID // 2, GRID // 2] = 1
    return x


def alive(x):
    # max_pool pads with -inf, so out-of-grid cells are ignored
    return F.max_pool2d(x[:, 3:4], 3, 1, 1) > 0.1


class CA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Conv2d(3 * CH, 128, 1)
        self.w2 = torch.nn.Conv2d(128, CH, 1, bias=False)
        torch.nn.init.zeros_(self.w2.weight)  # start as "do nothing"
        self.register_buffer('kx', SOBEL_X.repeat(CH, 1, 1, 1), persistent=False)
        self.register_buffer('ky', SOBEL_X.T.repeat(CH, 1, 1, 1), persistent=False)

    def sobel(self, x, angle=0.0):
        """Per-channel (dx, dy): Sobel gradients rotated by angle. Zero padding, cross-correlation."""
        c, s = math.cos(angle), math.sin(angle)
        # Rotating the kernels == rotating (gx, gy) (linear), minus 6 big ops.
        dx = F.conv2d(x, c * self.kx - s * self.ky, padding=1, groups=CH)
        dy = F.conv2d(x, s * self.kx + c * self.ky, padding=1, groups=CH)
        return dx, dy

    def forward(self, x, angle=0.0, fire_rate=0.5, fire=None):
        """One CA step on x [B, 16, H, W]. fire: optional precomputed [B, 1, H, W] bool update mask."""
        pre = alive(x)
        dx, dy = self.sobel(x, angle)
        p = torch.cat([x, dx, dy, torch.ones_like(x[:, :1])], 1)  # filter-major s, dx, dy (+ a ones channel)
        # b1 rides on the ones channel: same math as w1(p) + b1, but MPS's conv bias-gradient costs ~3 ms/step.
        w1 = torch.cat([self.w1.weight, self.w1.bias[:, None, None, None]], 1)
        ds = self.w2(F.relu(F.conv2d(p, w1)))
        if fire is None:
            fire = torch.rand_like(x[:, :1]) < fire_rate  # each cell fires independently
        x = x + ds * fire
        return x * (pre & alive(x))


def rollout(model, x, steps, angle=0.0, fire_rate=0.5):
    for _ in range(steps):
        x = model(x, angle, fire_rate)
    return x


def load_model(path, device='cpu'):
    model = CA().to(device)
    model.load_state_dict(torch.load(path, map_location=device)['model'])
    return model.eval()


def to_rgb(x):
    """State [..., 16, H, W] -> uint8 numpy [..., H, W, 3]: premultiplied RGB composited onto white."""
    a = x[..., 3:4, :, :].clamp(0, 1)
    rgb = (1 - a + x[..., :3, :, :]).clamp(0, 1)
    return (rgb.movedim(-3, -1) * 255).round().byte().cpu().numpy()


def circle_masks(n, h=GRID, w=GRID):
    """Distill damage masks [n, h, w]: 1 inside a circle, center in [-.5,.5], radius in [.1,.4] (grid spans [-1,1])."""
    x = torch.linspace(-1, 1, w)[None, None, :]
    y = torch.linspace(-1, 1, h)[None, :, None]
    cx, cy = torch.rand(2, n, 1, 1) - 0.5
    r = torch.rand(n, 1, 1) * 0.3 + 0.1
    return ((x - cx) ** 2 + (y - cy) ** 2 < r ** 2).float()


def loss_fn(x, target):
    """Per-sample MSE of RGBA vs target -> [B]."""
    return ((x[:, :4] - target) ** 2).mean([1, 2, 3])


def train_rollout(model, x, steps, chunk=8, tail=32):
    """rollout() for training, with a small memory footprint.

    Storing the autograd graph of all 96 steps needs ~5 GB on MPS (swaps an 8 GB Mac, ~10x slower), so
    each `chunk` of steps is checkpointed: only its input is kept and it is re-run during backward.
    The last `tail` steps keep their graph: ~10% faster for ~1 GB more (peak ~2.2 GB on MPS).
    Fire masks are drawn up front because checkpoint doesn't restore the MPS RNG on re-run.
    """
    fires = torch.rand(steps, *x[:, :1].shape, device=x.device) < 0.5

    def run(x, fires):
        for fire in fires:
            x = model(x, fire=fire)
        return x

    for i in range(0, steps, chunk):
        f = fires[i:i + chunk]
        x = run(x, f) if i >= steps - tail else checkpoint(run, x, f, use_reentrant=False, preserve_rng_state=False)
    return x


def train_step(model, opt, x, target, steps):
    x = train_rollout(model, x, steps)
    loss = loss_fn(x, target).mean()
    opt.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p.grad /= p.grad.norm() + 1e-8  # per-parameter gradient normalisation
    opt.step()
    return x.detach(), loss.item()


def bench(device, n=20):
    """Seconds per training iteration (80 steps) on device."""
    model = CA().to(device)
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    target = load_target('lizard').to(device)
    train_step(model, opt, make_seed(BATCH).to(device), target, 80)  # warm-up
    t = time.time()
    for _ in range(n):
        train_step(model, opt, make_seed(BATCH).to(device), target, 80)
    return (time.time() - t) / n


def pick_device():
    if not torch.backends.mps.is_available():
        return 'cpu'
    times = {d: bench(d) for d in ('cpu', 'mps')}
    print('benchmark sec/iter:', {d: round(t, 3) for d, t in times.items()})
    return min(times, key=times.get)


def train(target, regime, iters, out, device):
    device = pick_device() if device == 'auto' else device
    print(f'training {target} ({regime}) for {iters} iters on {device}')
    tgt = load_target(target).to(device)
    model = CA().to(device)
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [2000], 0.1)  # 2e-3 -> 2e-4 at iter 2000
    seed = make_seed().to(device)
    pool = seed.repeat(POOL, 1, 1, 1)
    log, t0 = [], time.time()
    for i in range(1, iters + 1):
        if regime == 'growing':
            x = seed.repeat(BATCH, 1, 1, 1)
        else:
            idx = torch.randperm(POOL)[:BATCH].to(device)
            x = pool[idx]
            with torch.no_grad():
                order = loss_fn(x, tgt).argsort(descending=True)
            idx, x = idx[order], x[order]
            x[:1] = seed  # highest-loss sample -> fresh seed
            if regime == 'regenerating':  # damage the 3 lowest-loss samples
                x[-3:] *= 1 - circle_masks(3).to(device)[:, None]
        x, loss = train_step(model, opt, x, tgt, random.randint(64, 96))
        sched.step()
        if regime != 'growing':
            pool[idx] = x
        if i % 10 == 0:
            log.append([i, loss])
        if i % 100 == 0 or i == iters:
            print(f'iter {i:5d}  loss {loss:.5f}  {i / (time.time() - t0):.2f} it/s', flush=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'target': target, 'regime': regime, 'iters': iters}, out)
    out.with_name(out.stem + '_loss.json').write_text(json.dumps(log))
    print(f'saved {out}  ({(time.time() - t0) / 60:.1f} min)')


def flat(t, digits):
    """Tensor -> flat list of floats rounded to `digits` significant digits (9 = lossless float32)."""
    return [float(f'{v:.{digits}g}') for v in t.detach().cpu().reshape(-1).tolist()]


def hwc(x):
    """[1, 16, H, W] -> HWC order, index = (row*W + col)*16 + ch."""
    return x[0].permute(1, 2, 0)


def weights(model, digits):
    # conv weights are [out, in, 1, 1] -> row-major W[o][i]
    return {'w1': flat(model.w1.weight, digits), 'b1': flat(model.w1.bias, digits), 'w2': flat(model.w2.weight, digits)}


def export():
    models = {}
    for name, (label, emoji) in MODELS.items():
        path = ROOT / 'checkpoints' / f'{name}.pt'
        if path.exists():
            models[name] = {'label': label, 'emoji': emoji, **weights(load_model(path), 7)}
    (ROOT / 'demo').mkdir(exist_ok=True)
    js = 'window.NCA_MODELS = ' + json.dumps(models, separators=(',', ':')) + ';\n'
    (ROOT / 'demo' / 'weights.js').write_text(js)
    print('demo/weights.js:', ', '.join(models))

    model = load_model(ROOT / 'checkpoints' / 'lizard.pt')
    torch.manual_seed(0)
    with torch.no_grad():
        x = rollout(model, make_seed(), 120)
        cases = [{'angle': a, 'state_in': flat(hwc(x), 9), 'state_out': flat(hwc(model(x, a, 1.0)), 9)}
                 for a in (0.0, 0.785398)]
    fixture = {'H': GRID, 'W': GRID, 'model': weights(model, 9), 'cases': cases}
    (ROOT / 'demo' / 'parity_fixture.json').write_text(json.dumps(fixture, separators=(',', ':')))
    print('demo/parity_fixture.json written')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--target', default='lizard')
    t.add_argument('--regime', choices=['growing', 'persistent', 'regenerating'], default='regenerating')
    t.add_argument('--iters', type=int, default=8000)
    t.add_argument('--out', required=True)
    t.add_argument('--device', default='auto', help='auto | cpu | mps')
    sub.add_parser('export')
    args = ap.parse_args()
    if args.cmd == 'train':
        train(args.target, args.regime, args.iters, args.out, args.device)
    else:
        export()
