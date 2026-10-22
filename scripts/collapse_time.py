"""Time to collapse for H > H_c (task 3 of A4).

For H > H_c the right-hand side is negative for every N. Completing the square,
    dN/dt = -a [ (N - K/2)^2 + w^2 ],   a = r/K,   w^2 = (H - H_c)/a,
and separation of variables gives
    t_c = [ atan((N0 - K/2)/w) + atan(K/(2w)) ] / sqrt(a (H - H_c)).
The script compares this formula with the RK4 run that stops at N = 0.

Run from the project folder:
    python scripts/collapse_time.py
Output: figures/collapse_time.png and a table on screen.
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import A4_fish as F  # noqa: E402

N0 = 800.0
DT = 0.05
P0 = F.make_params()
Hc = F.H_critical(P0)

H_values = np.array([126, 128, 130, 135, 140, 150, 175, 200, 250], dtype=float)
t_exact = []
t_sim = []

print("   H    t_c formula   t_c RK4      difference")
for H in H_values:
    p = F.make_params(H=float(H))
    te = F.collapse_time_exact(p, N0)
    t_end = np.ceil(1.2 * te / DT) * DT
    res = F.simulate(p, N0, (0.0, t_end), DT, "rk4")
    ts = res["collapse_time"]
    t_exact.append(te)
    t_sim.append(ts)
    print("%5.0f  %11.4f  %11.4f  %10.2e" % (H, te, ts, abs(te - ts)))

H_line = np.linspace(Hc + 0.5, 250.0, 300)
t_line = [F.collapse_time_exact(F.make_params(H=float(h)), N0) for h in H_line]

fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=150)
ax.plot(H_line, t_line, color="#1f5fa8", lw=1.8, label="formula")
ax.plot(H_values, t_sim, "o", ms=4.5, color="#c4501b", mfc="none", label="RK4 with stop at N = 0")
ax.axvline(Hc, color="#777777", lw=0.8, ls=":")
ax.set_yscale("log")
ax.set_xlabel("Quota H (tonnes/year)")
ax.set_ylabel("Time to collapse (years)")
ax.set_title("Collapse time above H_c, N0 = %.0f t" % N0, fontsize=10)
ax.grid(True, which="both", color="#dddddd", lw=0.6)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()

out_dir = os.path.join(ROOT, "figures")
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, "collapse_time.png"))
print("saved figures/collapse_time.png")
