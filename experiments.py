"""Experiments on the trained NCAs -> figures/ (PNG + GIF) and results.json (every number the README quotes).

  python3 experiments.py            # checkpoints/ -> figures/ + results.json, CPU only, ~5 min on an M1
  python3 experiments.py --quick    # fewer trials, for iterating
  python3 experiments.py --ckpt-dir D --out-dir F --results R.json

Fixed seeds: rerunning on the same checkpoints gives the same figures and numbers.
All losses are nca.loss_fn: per-pixel MSE of RGBA vs the target over the whole 72x72 grid.
"""
import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import nca

ROOT = Path(__file__).parent
REGIMES = {'regenerating': 'lizard', 'persistent': 'lizard_persistent', 'growing': 'lizard_growing'}  # -> checkpoint
COLORS = {'regenerating': '#2a78d6', 'persistent': '#eb6834', 'growing': '#1baf7a'}  # colour-blind-checked on white
CHUNK = 8  # samples per forward pass; measured on the M1 CPU: ~1.3 ms/sample-step up to 12, ~6 ms at 16
RADII = (3, 6, 9, 12, 15, 18)
RATES = (0.1, 0.25, 0.5, 0.75, 1.0)
HEAL_CUT = (26, 28, 12)  # row, col, radius (px) of the GIF's cut: the lizard's head and front legs
GIF_SCALE = 4
plt.rcParams.update({'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'savefig.dpi': 150,
                     'savefig.bbox': 'tight', 'axes.spines.top': False, 'axes.spines.right': False})


def img(x):
    """First sample of a state batch -> uint8 [72, 72, 3]. nan_to_num so an exploded state still renders."""
    return nca.to_rgb(torch.nan_to_num(x[0]))


@torch.no_grad()
def run(model, x, steps, target, fire_rate=0.5, angle=0.0, snaps=()):
    """Roll a batch forward `steps` steps.

    Returns (final x, loss vs target [steps + 1, B] as numpy, {step: img of sample 0 for step in snaps}).
    NaN losses (an exploded state) become inf, so they count as "far from the target", never as missing.
    """
    xs, losses, imgs = [], [], {}
    for i, c in enumerate(x.split(CHUNK)):
        ls = []
        for t in range(steps + 1):
            if t:
                c = model(c, angle, fire_rate)
            ls.append(nca.loss_fn(c, target))
            if i == 0 and t in snaps:
                imgs[t] = img(c)
        xs.append(c)
        losses.append(torch.stack(ls))
    return torch.cat(xs), np.nan_to_num(torch.cat(losses, 1).numpy(), nan=np.inf, posinf=np.inf), imgs


def holes(cy, cx, r):
    """[n, 72, 72] masks: 1 inside the circle of radius r px around (cy[i], cx[i])."""
    yy, xx = torch.arange(nca.GRID)[:, None], torch.arange(nca.GRID)
    return (((yy - cy[:, None, None]) ** 2 + (xx - cx[:, None, None]) ** 2) <= r * r).float()


def show(ax, im, title, **kw):
    out = ax.imshow(im, interpolation='nearest', **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color('#cccccc')
    ax.set_title(title, fontsize=9)
    return out


def image_grid(path, rows, titles, row_labels=None, suptitle=None):
    """rows/titles: lists of rows of images/strings."""
    nr, nc = len(rows), len(rows[0])
    fig, axes = plt.subplots(nr, nc, figsize=(1.7 * nc, 1.8 * nr + 0.5), squeeze=False)
    for r in range(nr):
        for c in range(nc):
            show(axes[r, c], rows[r][c], titles[r][c])
        if row_labels:
            axes[r, 0].set_ylabel(row_labels[r], fontsize=9)
    if suptitle:
        fig.suptitle(suptitle)
    save(fig, path)


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print('  wrote', path)


def gif_frame(im, label, font):
    """72x72 image -> nearest-neighbour upscaled GIF frame with a caption bar underneath."""
    big = Image.fromarray(im).resize((nca.GRID * GIF_SCALE,) * 2, Image.NEAREST)
    frame = Image.new('RGB', (big.width, big.height + 26), 'white')
    frame.paste(big)
    ImageDraw.Draw(frame).text((8, big.height + 4), label, fill=(70, 70, 70), font=font)
    return frame


def growth(m, tgt, path, title):
    torch.manual_seed(0)
    steps = (0, 10, 20, 40, 60, 100, 200)
    _, loss, imgs = run(m, nca.make_seed(), 200, tgt, snaps=steps)
    image_grid(path, [[imgs[s] for s in steps]], [[f'step {s}' for s in steps]], suptitle=title)
    return {'loss_at_step': {s: loss[s, 0] for s in steps}}


def heal(m, tgt, out):
    """Hero GIF: grow 150 steps from the seed, cut a big circle out, regrow 150 steps."""
    torch.manual_seed(0)
    x, l1, grow = run(m, nca.make_seed(), 150, tgt, snaps=range(151))
    cy, cx, r = HEAL_CUT
    x = x * (1 - holes(torch.tensor([cy]), torch.tensor([cx]), r)[:, None])
    _, l2, regrow = run(m, x, 150, tgt, snaps=range(151))

    font = ImageFont.load_default(size=16)
    frames, ms = [], []
    for s in range(0, 151, 2):  # every 2nd step keeps the file small; holds on the grown lizard, the cut, the end
        frames.append(gif_frame(grow[s], f'step {s}   growing', font))
        ms.append(800 if s == 150 else 40)
    frames.append(gif_frame(regrow[0], 'step 150   cut!', font))
    ms.append(900)
    for s in range(2, 151, 2):
        frames.append(gif_frame(regrow[s], f'step {150 + s}   regrowing', font))
        ms.append(1500 if s == 150 else 40)
    frames[0].save(out / 'heal.gif', save_all=True, append_images=frames[1:], duration=ms, loop=0)
    print('  wrote', out / 'heal.gif', f'({(out / "heal.gif").stat().st_size / 1e6:.2f} MB)')

    strip = [(grow[60], 'step 60'), (grow[150], 'step 150'), (regrow[0], 'cut'),
             (regrow[20], '+20 steps'), (regrow[60], '+60 steps'), (regrow[150], '+150 steps')]
    image_grid(out / 'heal_strip.png', [[im for im, _ in strip]], [[t for _, t in strip]],
               suptitle=f'Regeneration: grow 150 steps, cut a radius-{r} circle, regrow 150 steps')
    return {'cut_row': cy, 'cut_col': cx, 'cut_radius_px': r, 'loss_before_cut': l1[-1, 0],
            'loss_right_after_cut': l2[0, 0], 'loss_150_steps_after_cut': l2[-1, 0]}


def training_curves(ckpt_dir, path):
    # Not a like-for-like comparison: "growing" starts every batch from the seed; the pool regimes
    # replace 1 of 8 samples with the seed, and "regenerating" also damages 3 of 8.
    fig, ax = plt.subplots(figsize=(7, 4))
    res, k = {}, 10  # k = 10 log points = 100-iteration moving average
    for regime, name in REGIMES.items():
        p = ckpt_dir / f'{name}_loss.json'
        if not p.exists():
            continue
        it, loss = np.array(json.loads(p.read_text())).T
        ax.plot(it, loss, color=COLORS[regime], alpha=0.2, lw=0.8)
        ax.plot(it[k - 1:], np.convolve(loss, np.ones(k) / k, 'valid'), color=COLORS[regime], lw=2, label=regime)
        res[regime] = {'final_train_loss_last_100_iters': loss[-k:].mean()}
    ax.set(yscale='log', xlabel='training iteration', ylabel='batch loss (log)',
           title='Training loss (faint: every 10 iterations, bold: 100-iteration mean)')
    ax.grid(alpha=0.3)
    ax.legend(title='regime')
    save(fig, path)
    return res


def long_run(models, tgt, thr, path, runs, steps=2000):
    """Persistence: does the pattern hold for 2,000 steps, far past the 64-96 steps it was trained on?"""
    torch.manual_seed(1)
    fig = plt.figure(figsize=(9, 8.4))
    gs = fig.add_gridspec(3, len(models), height_ratios=[2.2, 1.3, 1.3], hspace=0.45)
    ax = fig.add_subplot(gs[0, :])
    t = np.arange(steps + 1)
    res = {'steps': steps, 'runs': runs, 'fire_rate': 0.5}
    for j, (regime, m) in enumerate(models.items()):
        _, loss, imgs = run(m, nca.make_seed(runs), steps, tgt, snaps=(100, steps))
        mean, final = loss.mean(1), loss[-1]
        ax.fill_between(t, loss.min(1), loss.max(1), color=COLORS[regime], alpha=0.2, lw=0)
        ax.plot(t, mean, color=COLORS[regime], lw=1.8, label=regime)
        for k, s in enumerate((100, steps)):
            show(fig.add_subplot(gs[1 + k, j]), imgs[s], f'{regime}, step {s}')
        res[regime] = {'loss_at_100_mean': mean[100], 'best_mean_loss': mean.min(), 'best_step': int(mean.argmin()),
                       f'loss_at_{steps}_mean': final.mean(), f'loss_at_{steps}_max': final.max(),
                       f'runs_like_target_at_{steps}': int((final < thr).sum()),
                       f'looks_like_target_at_{steps}': bool((final < thr).all())}
    ax.axhline(thr, color='0.35', ls='--', lw=1, label='"looks like the target" threshold')
    ax.set(yscale='log', xlabel='step', ylabel='loss vs target (log)', xlim=(0, steps),
           title=f'Persistence: {runs} runs per regime from the seed (line: mean, band: min to max)')
    ax.set_ylim(top=min(ax.get_ylim()[1], 1))  # an exploding run leaves through the top, not squash the rest
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, path)
    return res


def damage_sweep(models, tgt, thr, path, trials):
    """Grow 200 steps, cut a circular hole (all channels -> 0) centred on a random creature pixel, run 300 more."""
    torch.manual_seed(2)
    cy, cx = torch.nonzero(tgt[3] > 0.1, as_tuple=True)  # creature pixels of the target
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    res = {'radii_px': list(RADII), 'trials_per_radius': trials, 'grow_steps': 200, 'steps_after_cut': 300}
    for regime, m in models.items():
        c = COLORS[regime]
        grown, _, _ = run(m, nca.make_seed(trials), 200, tgt)
        _, base, _ = run(m, grown, 300, tgt)  # undamaged control: the same creatures, no cut
        base = base.mean(1, keepdims=True)  # [301, 1]
        rr = res[regime] = {'undamaged_loss_at_500': base[-1, 0], 'healed_pct': {}, 'final_loss_mean': {}}
        for r in RADII:
            pick = torch.randint(len(cy), (trials,))
            _, loss, _ = run(m, grown * (1 - holes(cy[pick], cx[pick], r)[:, None]), 300, tgt)
            # healed = within 1.5x the model's own undamaged loss at the same step (so a model is never
            # asked to beat itself) AND below the absolute "looks like the target" threshold (so a model
            # whose undamaged pattern has already decayed can't count as "healed" back to a blob)
            ok = (loss <= 1.5 * base) & (loss < thr)  # [301, trials]
            rr['healed_pct'][r] = 100 * ok[-1].mean()
            rr['final_loss_mean'][r] = loss[-1].mean()
            if regime == 'regenerating':  # first step the healed bar is met, over trials healed at the end
                first = ok.argmax(0)[ok[-1]]
                rr.setdefault('median_steps_to_heal', {})[r] = float(np.median(first)) if len(first) else None
            if r == 9:
                a2.plot(loss.mean(1), color=c, lw=1.8, label=regime)
                a2.plot(base[:, 0], color=c, lw=1, ls='--')
        a1.plot(RADII, [rr['healed_pct'][r] for r in RADII], 'o-', color=c, lw=1.8, ms=7, label=regime)
    a1.set(xticks=RADII, ylim=(-4, 104), xlabel='hole radius (px; the lizard is ~40 px across)',
           ylabel='healed after 300 steps (%)', title=f'Healing vs hole size ({trials} trials per radius)')
    a2.plot([], [], color='0.35', lw=1, ls='--', label='undamaged control')
    a2.axhline(thr, color='0.35', ls=':', lw=1, label='"looks like the target" threshold')
    a2.set(yscale='log', xlim=(0, 300), xlabel='steps after the cut', ylabel='loss vs target (log)',
           title=f'Recovery after a radius-9 cut (mean of {trials})')
    a2.set_ylim(top=min(a2.get_ylim()[1], 1))
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    save(fig, path)
    return res


def knockout(m, tgt, thr, path, runs):
    """Where is the memory? Whole-body surgeries on the grown lizard, then 200 more steps."""
    torch.manual_seed(3)
    grown, _, _ = run(m, nca.make_seed(runs), 200, tgt)
    _, ctrl, _ = run(m, grown, 200, tgt)
    base = ctrl.mean(1, keepdims=True)
    surgeries = {  # key: (label, channels zeroed, columns zeroed)
        'wipe_rgb': ('wipe RGB\n(ch 0-2)', slice(0, 3), slice(None)),
        'wipe_hidden': ('wipe hidden\n(ch 4-15)', slice(4, 16), slice(None)),
        'wipe_left_half': ('wipe left half\n(all channels)', slice(None), slice(0, nca.GRID // 2)),
    }
    snaps = (0, 20, 50, 200)
    res = {'runs': runs, 'grow_steps': 200, 'steps_after': 200, 'control_loss_at_+200': base[-1, 0]}
    rows, labels = [], []
    for key, (label, ch, cols) in surgeries.items():
        x = grown.clone()
        x[:, ch, :, cols] = 0
        _, loss, imgs = run(m, x, 200, tgt, snaps=snaps)
        ok = (loss[-1] <= 1.5 * base[-1]) & (loss[-1] < thr)  # same "healed" bar as the damage sweep
        res[key] = {'loss_right_after': loss[0].mean(), 'loss_at_+200': loss[-1].mean(), 'healed_pct': 100 * ok.mean()}
        rows.append([imgs[s] for s in snaps])
        labels.append(f'{label}\nloss at +200: {loss[-1].mean():.4f}')
    image_grid(path, rows, [[f'+{s} steps' for s in snaps]] + [[''] * len(snaps)] * (len(rows) - 1), labels,
               suptitle=f'Knockouts on the grown lizard (undamaged control loss at +200: {base[-1, 0]:.4f})')
    return res, grown[:1]


def hidden_channels(x, path):
    h = x[0, 4:].numpy()
    v = np.abs(h).max()
    fig, axes = plt.subplots(3, 4, figsize=(8.4, 6.6))
    for k, ax in enumerate(axes.flat):
        im = show(ax, h[k], f'ch {k + 4}', cmap='RdBu_r', vmin=-v, vmax=v)
    fig.colorbar(im, ax=axes, shrink=0.8, label='channel value (shared scale)')
    fig.suptitle('The 12 hidden channels of the grown lizard (step 200)')
    save(fig, path)


def rotation(m, tgt, path):
    torch.manual_seed(4)
    angles = (0, 45, 90, 180)
    ims = [run(m, nca.make_seed(), 200, tgt, angle=math.radians(a), snaps=(200,))[2][200] for a in angles]
    image_grid(path, [ims], [[f'angle {a}°' for a in angles]],
               suptitle='Same weights, perception rotated: grown 200 steps from the seed')


def fire_rate(m, tgt, thr, path, runs):
    """Robustness to the update rate. steps = round(150 / p): every rate gets 150 expected updates per cell
    (300 steps at the trained p = 0.5). p = 1 updates every cell every step, which training never showed."""
    torch.manual_seed(5)
    fig = plt.figure(figsize=(10, 5.8))
    gs = fig.add_gridspec(2, len(RATES), height_ratios=[2, 1.15], hspace=0.55)
    ax = fig.add_subplot(gs[0, :])
    res = {'runs': runs, 'steps_formula': 'round(150 / fire_rate)'}
    mean, lo, hi = [], [], []
    for j, p in enumerate(RATES):
        steps = round(150 / p)
        _, loss, imgs = run(m, nca.make_seed(runs), steps, tgt, fire_rate=p, snaps=(steps,))
        f = loss[-1]
        res[p] = {'steps': steps, 'final_loss_mean': f.mean(), 'final_loss_min': f.min(), 'final_loss_max': f.max(),
                  'runs_like_target': int((f < thr).sum())}
        mean.append(f.mean())
        lo.append(f.mean() - f.min())
        hi.append(f.max() - f.mean())
        show(fig.add_subplot(gs[1, j]), imgs[steps], f'p = {p}, {steps} steps')
    ax.errorbar(RATES, mean, yerr=[lo, hi], fmt='o-', color=COLORS['regenerating'], ms=7, capsize=4,
                label=f'regenerating lizard (mean, min to max of {runs})')
    ax.axhline(thr, color='0.35', ls='--', lw=1, label='"looks like the target" threshold')
    ax.axvline(0.5, color='0.6', lw=1, ls=':', label='trained fire rate')
    ax.set(yscale='log', xticks=RATES, xlabel='fire rate p (chance each cell updates per step)',
           ylabel='final loss vs target (log)', title='Robustness to the update rate')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, path)
    return res


def clean(o):
    """results -> JSON-safe: 4 significant digits, string keys, non-finite floats -> null."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(f'{o:.4g}') if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', type=Path, default=ROOT / 'checkpoints')
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'figures')
    ap.add_argument('--results', type=Path, default=ROOT / 'results.json')
    ap.add_argument('--quick', action='store_true', help='fewer trials, for iterating')
    args = ap.parse_args()
    torch.set_num_threads(4)  # chunks of 8 gain little past 2-4 threads; leaves cores for a concurrent training run
    t0 = time.time()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    runs, trials = (2, 4) if args.quick else (8, 20)

    ckpts = {name: torch.load(args.ckpt_dir / f'{name}.pt', map_location='cpu')
             for name in nca.MODELS if (args.ckpt_dir / f'{name}.pt').exists()}
    models = {r: nca.load_model(args.ckpt_dir / f'{n}.pt') for r, n in REGIMES.items() if n in ckpts}
    liz, tgt = models['regenerating'], nca.load_target('lizard')
    empty = nca.loss_fn(torch.zeros(1, 4, nca.GRID, nca.GRID), tgt).item()
    # "Looks like the target" = loss below that of the PERFECT target shifted by one pixel. A one-pixel
    # shift is visually the same lizard, so anything closer than that reads as the lizard; anything
    # worse (blurred, faded, overgrown, exploded) does not. For the lizard this is ~0.006, about 1/5
    # of the empty-grid loss (~0.03).
    thr = nca.loss_fn(torch.roll(tgt, 1, dims=2)[None], tgt).item()
    res = {
        'quick_mode': args.quick,
        'param_count': sum(p.numel() for p in liz.parameters()),
        'checkpoints': {n: {'regime': c['regime'], 'target': c['target'], 'iters': c['iters']} for n, c in ckpts.items()},
        'loss': 'per-pixel MSE of RGBA vs the target over the 72x72 grid (nca.loss_fn)',
        'empty_grid_loss': empty,
        'looks_like_target_threshold': thr,
        'looks_like_target_definition': 'loss < loss of the perfect target shifted by 1 px',
        'healed_definition': "loss <= 1.5x the same model's mean undamaged loss at the same step "
                             'AND loss < looks_like_target_threshold',
    }

    print('training curves')
    res['training'] = training_curves(args.ckpt_dir, out / 'training_curves.png')
    print('growth')
    res['growth'] = growth(liz, tgt, out / 'growth.png', 'Growing from a single cell (regenerating lizard)')
    for name in ('butterfly', 'mushroom'):
        if name in ckpts:
            m = nca.load_model(args.ckpt_dir / f'{name}.pt')
            res[f'growth_{name}'] = growth(m, nca.load_target(name), out / f'growth_{name}.png',
                                           f'Growing from a single cell ({name})')
    print('heal')
    res['heal'] = heal(liz, tgt, out)
    print('long run')
    res['long_run'] = long_run(models, tgt, thr, out / 'long_run.png', runs)
    print('damage sweep')
    res['damage_sweep'] = damage_sweep(models, tgt, thr, out / 'damage_sweep.png', trials)
    print('knockout')
    res['knockout'], grown = knockout(liz, tgt, thr, out / 'knockout.png', runs)
    hidden_channels(grown, out / 'hidden_channels.png')
    print('rotation')
    rotation(liz, tgt, out / 'rotation.png')
    print('fire rate')
    res['fire_rate'] = fire_rate(liz, tgt, thr, out / 'fire_rate.png', runs)

    args.results.write_text(json.dumps(clean(res), indent=2, allow_nan=False) + '\n')
    print(f'wrote {args.results}  ({(time.time() - t0) / 60:.1f} min)')


if __name__ == '__main__':
    main()
