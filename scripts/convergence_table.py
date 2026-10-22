"""Convergence table for Euler, Heun and RK4.

Part 1 measures the orders directly from A4_fish.py: baseline problem on
[0, 10] years, step sizes 1, 1/2, 1/4, 1/8, 1/16, error at t = 10 against an
RK4 run with dt = 0.001. The observed order between two steps is
    p = ln(e_i / e_{i+1}) / ln(dt_i / dt_{i+1}).

Part 2 lists every field whose name contains "order" in the report files in
convergence/reports/*.json (written by wmlab), so the two sources can be
compared. The layout of those files is not assumed.

Run from the project folder:
    python scripts/convergence_table.py
"""

import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import A4_fish as F  # noqa: E402

P = F.make_params()
N0 = 800.0
T_END = 10.0
DT_LIST = [1.0, 0.5, 0.25, 0.125, 0.0625]
REF_DT = 0.001
EXPECTED = {"euler": 1, "heun": 2, "rk4": 4}

ref = F.simulate(P, N0, (0.0, T_END), REF_DT, "rk4")["N"][-1]
print("reference N(10) = %.10f (RK4, dt = %g)" % (ref, REF_DT))
print()

for method in ("euler", "heun", "rk4"):
    errs = []
    for dt in DT_LIST:
        n_end = F.simulate(P, N0, (0.0, T_END), dt, method)["N"][-1]
        errs.append(abs(n_end - ref))
    print("%s (expected order %d)" % (method, EXPECTED[method]))
    print("    dt        error        observed order")
    for i, dt in enumerate(DT_LIST):
        if i == 0:
            order_txt = "-"
        else:
            order_txt = "%.2f" % (math.log(errs[i - 1] / errs[i]) / math.log(DT_LIST[i - 1] / dt))
        print("  %7.4f   %.3e   %s" % (dt, errs[i], order_txt))
    print()


def find_orders(obj, path=""):
    """Yield (path, value) for numeric fields whose key contains 'order'."""
    if isinstance(obj, dict):
        for key, val in obj.items():
            sub = path + "/" + str(key)
            if "order" in str(key).lower() and isinstance(val, (int, float)):
                yield sub, val
            else:
                for item in find_orders(val, sub):
                    yield item
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            for item in find_orders(val, path + "[%d]" % i):
                yield item


report_files = sorted(glob.glob(os.path.join(ROOT, "convergence", "reports", "*.json")))
if not report_files:
    print("No wmlab reports found in convergence/reports/ yet.")
for fname in report_files:
    with open(fname) as fh:
        data = json.load(fh)
    found = list(find_orders(data))
    print(os.path.basename(fname))
    if not found:
        print("  no field with 'order' in its name")
    for path, val in found:
        print("  %s = %s" % (path, val))
