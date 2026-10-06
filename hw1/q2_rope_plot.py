"""HW1 Q2: RoPE attention score between two all-ones vectors vs. relative distance |i-j|."""
import numpy as np
import matplotlib.pyplot as plt

d = 128
theta = 10000.0 ** (-np.arange(0, d, 2) / d)          # 64 frequencies
dist = np.arange(0, 65537)
A = np.cos(np.outer(dist, theta)).mean(axis=1)       # A_ij = (1/64) * sum_m cos(|i-j| * theta_m)

first_neg = int(np.argmax(A < 0))
print(f"A(0)={A[0]:.3f}  first negative at |i-j|={first_neg}  "
      f"mean|A| over [1k,64k]={np.abs(A[1000:]).mean():.3f}")

INK, MUTED, GRID, LINE = "#1f2328", "#6e7781", "#e6e8eb", "#2a6fdb"
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, xs, ys, title in [(axes[0], dist[:1025], A[:1025], "|i-j| = 0 ... 1024 (linear)"),
                          (axes[1], dist[1:], A[1:], "|i-j| = 1 ... 65536 (log x)")]:
    ax.plot(xs, ys, color=LINE, lw=1.2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel("|i - j|", color=INK); ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
axes[1].set_xscale("log")
axes[0].set_ylabel(r"$A_{ij}$", color=INK)
fig.tight_layout()
fig.savefig("plots/q2_rope_decay.png", dpi=200)
