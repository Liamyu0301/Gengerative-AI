"""HW1 Q4: parse logs/<run>.log -> loss-curve plots in plots/ and a summary table in results/summary.md."""
import re, os, sys
import matplotlib.pyplot as plt

LOGDIR = sys.argv[1] if len(sys.argv) > 1 else 'logs'
COLORS = ['#2a78d6', '#eb6834', '#1baf7a']   # categorical slots 1-3, fixed order: baseline first
INK, MUTED, GRID = '#1f2328', '#6e7781', '#e6e8eb'
NAMES = {'baseline': 'baseline (LN, GELU, learned PE, MHA)', 'rmsnorm': 'RMSNorm', 'swiglu': 'SwiGLU',
         'nope': 'NoPE', 'rope': 'RoPE', 'gqa': 'GQA (3 KV heads)'}
COMPARISONS = [('q4_2_rmsnorm', ['baseline', 'rmsnorm']), ('q4_3_swiglu', ['baseline', 'swiglu']),
               ('q4_4_posenc', ['baseline', 'nope', 'rope']), ('q4_5_gqa', ['baseline', 'gqa'])]
LRS = [('', 'lr 1e-3'), ('_lr2e-3', 'lr 2e-3')]

def parse(run):
    path = f'{LOGDIR}/{run}.log'
    if not os.path.exists(path):
        return None
    ev, params = [], 0
    for line in open(path):
        if m := re.match(r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)', line):
            ev.append(tuple(map(float, m.groups())))
        elif m := re.match(r'num (?:non-)?decayed parameter tensors: \d+, with ([\d,]+) parameters', line):
            params += int(m.group(1).replace(',', ''))  # total = decayed + non-decayed (includes wpe)
    return {'step': [e[0] for e in ev], 'train': [e[1] for e in ev], 'val': [e[2] for e in ev], 'params': params}

def style(ax, title):
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel('iteration', color=INK); ax.set_ylabel('loss', color=INK)
    ax.grid(color=GRID, lw=0.6); ax.tick_params(colors=MUTED)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)

def plot(name, runs, suffix, lr_label):
    data = [(r, parse(r + suffix)) for r in runs]
    data = [(r, d) for r, d in data if d and d['step']]
    if not data:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, zoom in zip(axes, (False, True)):
        for (r, d), c in zip(data, COLORS):
            label = NAMES[r] if len(runs) > 1 else None
            ax.plot(d['step'], d['val'], color=c, lw=1.8, label=f'{label} val' if label else 'val')
            ax.plot(d['step'], d['train'], color=c, lw=1.2, ls='--', label=f'{label} train' if label else 'train')
        if zoom:
            tail = [v for _, d in data for k in ('val', 'train') for s, v in zip(d['step'], d[k]) if s >= 1500]
            ax.set_xlim(1500, max(max(d['step']) for _, d in data) + 50)
            if tail: ax.set_ylim(min(tail) - 0.03, max(tail) + 0.03)
        style(ax, ('zoom: iter >= 1500' if zoom else 'full run') + f'  ({lr_label})')
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f'plots/{name}{suffix}.png', dpi=200); plt.close(fig)

os.makedirs('plots', exist_ok=True); os.makedirs('results', exist_ok=True)
for suffix, lr_label in LRS:
    for d in [parse('baseline' + suffix)]:
        if d and d['step']:
            fig, ax = plt.subplots(figsize=(6, 3.4))
            ax.plot(d['step'], d['train'], color=COLORS[0], lw=1.8, label='train')
            ax.plot(d['step'], d['val'], color=COLORS[1], lw=1.8, label='val')
            style(ax, f'Q4.1 baseline ({lr_label})'); ax.legend(frameon=False)
            fig.tight_layout(); fig.savefig(f'plots/q4_1_baseline{suffix}.png', dpi=200); plt.close(fig)
    for name, runs in COMPARISONS:
        plot(name, runs, suffix, lr_label)

rows = ['| run | params | lr | final train | final val | best val (iter) |', '|---|---|---|---|---|---|']
for r in NAMES:
    for suffix, lr_label in LRS:
        d = parse(r + suffix)
        if d and d['step']:
            i = min(range(len(d['val'])), key=d['val'].__getitem__)
            rows.append(f"| {r} | {d['params'] / 1e6:.3f}M | {lr_label[3:]} | {d['train'][-1]:.4f} | {d['val'][-1]:.4f} | "
                        f"{d['val'][i]:.4f} ({int(d['step'][i])}) |")
open('results/summary.md', 'w').write('\n'.join(rows) + '\n')
print('\n'.join(rows))
