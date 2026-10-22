"""Long-time stock N against quota H (task 2 of A4).

For each H from 0 to 150 the stock starts at N0 = K and is integrated for
1000 years with RK4. Where the stock collapses the long-time value is 0.
The exact equilibria N+ and N- are drawn on top for comparison.

Close to H_c the relaxation is very slow (f'(N+) goes to 0), so at H = 125 the
run still reads about 502 t after 1000 years while N+ = 500 t. This is critical
slowing down, not an error of the integrator.

Run from the project folder:
    python scripts/bifurcation.py
Output: figures/bifurcation.png and a short table on screen.
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

P0 = F.make_params()
Hc = F.H_critical(P0)
H_values = np.arange(0.0, 150.5, 1.0)
T_END = 1000.0
DT = 0.25

N_long = []
for H in H_values:
    p = F.make_params(H=float(H))
    res = F.simulate(p, p["K"], (0.0, T_END), DT, "rk4")
    N_long.append(res["N"][-1])
N_long = np.array(N_long)

print("   H     N(long)   N+ exact")
for H, n in zip(H_values, N_long):
    if H in (0, 50, 100, 120, 124, 125, 126, 150):
        eq = F.equilibria(F.make_params(H=float(H)))["stable"]
        eq_txt = "none" if eq is None else "%.1f" % eq
        print("%5.0f  %9.1f  %8s" % (H, n, eq_txt))

H_fine = np.linspace(0.0, Hc, 400)
N_plus = [F.equilibria(F.make_params(H=float(h)))["stable"] for h in H_fine]
N_minus = [F.equilibria(F.make_params(H=float(h)))["unstable"] for h in H_fine]

fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=150)
ax.plot(H_fine, N_plus, color="#1f5fa8", lw=1.8, label="N+ (stable)")
ax.plot(H_fine, N_minus, color="#1f5fa8", lw=1.8, ls="--", label="N- (unstable)")
ax.plot(H_values, N_long, "o", ms=3.5, color="#c4501b", mfc="none",
        label="simulation, start at N0 = K")
ax.axvline(Hc, color="#777777", lw=0.8, ls=":")
ax.text(Hc + 1.5, 900, "H_c = %.0f" % Hc, color="#444444", fontsize=9)
ax.set_xlabel("Quota H (tonnes/year)")
ax.set_ylabel("Long-time stock N (tonnes)")
ax.set_xlim(0, 150)
ax.set_ylim(-20, 1050)
ax.grid(True, color="#dddddd", lw=0.6)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc="lower left", fontsize=9)
ax.set_title("Fish stock against quota (r = %.1f /year, K = %.0f t)" % (P0["r"], P0["K"]),
             fontsize=10)
fig.tight_layout()

out_dir = os.path.join(ROOT, "figures")
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, "bifurcation.png"))
print("saved figures/bifurcation.png")
