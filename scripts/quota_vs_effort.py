"""Constant quota against constant effort (task 4 of A4).

Both policies are set to give the same stable equilibrium and the same
sustainable yield as the baseline: quota H = 100 t/year, which gives
N+ = 723.6 t. The matching effort solves E K (1 - E/r) = H, so
    E = (r/2) (1 - sqrt(1 - 4H/(rK))) = 0.1382 /year.
The stock is then started at several values of N0, including values below
the unstable equilibrium N- = 276.4 t.

Run from the project folder:
    python scripts/quota_vs_effort.py
Output: figures/quota_vs_effort.png and a table on screen.
"""

import math
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

H = 100.0
P_q = F.make_params(policy="quota", H=H)
r, K = P_q["r"], P_q["K"]
E = 0.5 * r * (1.0 - math.sqrt(1.0 - 4.0 * H / (r * K)))
P_e = F.make_params(policy="effort", E=E)

print("matching effort E = %.4f /year" % E)
print("quota:  N+ = %.1f, N- = %.1f" % (F.equilibria(P_q)["stable"], F.equilibria(P_q)["unstable"]))
print("effort: N* = %.1f, yield at N* = %.1f t/year" % (F.equilibria(P_e)["stable"],
                                                       E * F.equilibria(P_e)["stable"]))

N0_list = [150.0, 250.0, 300.0, 500.0, 800.0, 1000.0]
T_END = 60.0
DT = 0.05

print()
print("   N0     quota                 effort")
runs_q, runs_e = [], []
for N0 in N0_list:
    rq = F.simulate(P_q, N0, (0.0, T_END), DT, "rk4")
    re_ = F.simulate(P_e, N0, (0.0, T_END), DT, "rk4")
    runs_q.append(rq)
    runs_e.append(re_)
    if rq["collapse_time"] is None:
        q_txt = "N(60) = %.1f" % rq["N"][-1]
    else:
        q_txt = "collapse at t = %.1f" % rq["collapse_time"]
    e_txt = "N(60) = %.1f" % re_["N"][-1]
    print("%6.0f   %-20s  %s" % (N0, q_txt, e_txt))

fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), dpi=150, sharey=True)
for ax, runs, title in ((axes[0], runs_q, "Constant quota H = %.0f t/year" % H),
                        (axes[1], runs_e, "Constant effort E = %.4f /year" % E)):
    for res in runs:
        collapsed = res["collapse_time"] is not None
        ax.plot(res["t"], res["N"], lw=1.6,
                color="#c4501b" if collapsed else "#1f5fa8")
    ax.axhline(723.6, color="#777777", lw=0.8, ls=":")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (years)")
    ax.grid(True, color="#dddddd", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].axhline(276.4, color="#777777", lw=0.8, ls="--")
axes[0].text(33, 290, "N- = 276.4", fontsize=8, color="#444444")
axes[0].text(33, 735, "N+ = 723.6", fontsize=8, color="#444444")
axes[0].text(2, 60, "orange: stock collapses", fontsize=8, color="#c4501b")
axes[0].set_ylabel("Stock N (tonnes)")
axes[0].set_ylim(0, 1050)
fig.tight_layout()

out_dir = os.path.join(ROOT, "figures")
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, "quota_vs_effort.png"))
print("saved figures/quota_vs_effort.png")
