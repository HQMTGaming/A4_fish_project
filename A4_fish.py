"""A4 - Fish population with harvesting.

Model
    constant quota:   dN/dt = r N (1 - N/K) - H
    constant effort:  dN/dt = r N (1 - N/K) - E N

Units
    N  fish stock           [tonnes]
    t  time                 [years]
    r  intrinsic growth     [1/year]
    K  carrying capacity    [tonnes]
    H  quota                [tonnes/year]
    E  fishing effort       [1/year]

Assumptions
    No age structure, no seasons, no delay, no noise, closed population.
    Harvest does not depend on the weather or on the market.

Note on the interface
    The function names below (rhs, simulate, analytic, quantities, sample,
    EXTRA_CHECK) follow the catalogue. Compare the signatures with
    de_kit/template_system.py and adapt the wrappers if they differ.
    The Euler / Heun / RK4 steps are written here so the file runs on its own;
    replace them with the ones from the template if the course requires it.
"""

import math

import numpy as np

SYSTEM_NAME = "A4_fish"

DEFAULTS = {
    "policy": "quota",   
    "r": 0.5,            
    "K": 1000.0,         
    "H": 100.0,          
    "E": 0.0,            
}


def make_params(**kwargs):
    """Return a parameter dict: defaults overridden by kwargs."""
    p = dict(DEFAULTS)
    p.update(kwargs)
    if p["policy"] not in ("quota", "effort"):
        raise ValueError("policy must be 'quota' or 'effort'")
    return p




def rhs(t, N, p):
    """Right-hand side f(t, N). No clipping here, the event handles N <= 0."""
    growth = p["r"] * N * (1.0 - N / p["K"])
    if p["policy"] == "quota":
        return growth - p["H"]
    return growth - p["E"] * N




def euler_step(f, t, y, h, p):
    return y + h * f(t, y, p)


def heun_step(f, t, y, h, p):
    k1 = f(t, y, p)
    k2 = f(t + h, y + h * k1, p)
    return y + 0.5 * h * (k1 + k2)


def rk4_step(f, t, y, h, p):
    k1 = f(t, y, p)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1, p)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2, p)
    k4 = f(t + h, y + h * k3, p)
    return y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


STEPPERS = {"euler": euler_step, "heun": heun_step, "rk4": rk4_step}


def _step_to_zero(step, t, N, dt, p):
    """Length h in (0, dt] such that one step from N lands on N = 0.

    Bisection on the step length, using the same method as the main run,
    so the collapse time keeps the order of the method.
    """
    lo, hi = 0.0, dt
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if step(rhs, t, N, mid, p) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)




def simulate(p, N0, t_span, dt, method="rk4"):
    """Integrate with a fixed step and stop at the collapse event N = 0.

    Returns a dict with keys t, N, collapse_time (None if the stock survives),
    method and dt. If the stock collapses, the last point is (t_c, 0).
    """
    step = STEPPERS[method]
    t0, t1 = float(t_span[0]), float(t_span[1])
    n_steps = int(round((t1 - t0) / dt))
    if abs(n_steps * dt - (t1 - t0)) > 1e-9 * max(1.0, t1 - t0):
        raise ValueError("t_span must be a whole number of steps of size dt")

    t_list = [t0]
    N_list = [float(N0)]
    collapse_time = None

    if N0 <= 0.0:
        return {"t": np.array(t_list), "N": np.array(N_list),
                "collapse_time": t0, "method": method, "dt": dt}

    t = t0
    N = float(N0)
    for i in range(n_steps):
        N_new = step(rhs, t, N, dt, p)
        if N_new <= 0.0:
            h = _step_to_zero(step, t, N, dt, p)
            collapse_time = t + h
            t_list.append(collapse_time)
            N_list.append(0.0)
            break
        t = t0 + (i + 1) * dt
        N = N_new
        t_list.append(t)
        N_list.append(N)

    return {"t": np.array(t_list), "N": np.array(N_list),
            "collapse_time": collapse_time, "method": method, "dt": dt}




def analytic(t, p, N0):
    """Exact N(t) when there is no quota (H = 0, or effort policy).

    With harvest E N the equation is still logistic:
        dN/dt = a N - b N^2,  a = r - E,  b = r / K
    """
    if p["policy"] == "quota":
        if p["H"] > 0.0:
            raise ValueError("closed form only for H = 0 or the effort policy")
        E = 0.0
    else:
        E = p["E"]
    a = p["r"] - E
    b = p["r"] / p["K"]
    t = np.asarray(t, dtype=float)
    if abs(a) < 1e-14:
        return N0 / (1.0 + b * N0 * t)
    return a / (b + (a / N0 - b) * np.exp(-a * t))


def H_critical(p):
    """Maximum sustainable yield, H_c = r K / 4."""
    return p["r"] * p["K"] / 4.0


def equilibria(p):
    """Equilibria as a dict {"stable": ..., "unstable": ...}, None if absent."""
    r, K = p["r"], p["K"]
    if p["policy"] == "quota":
        disc = 1.0 - 4.0 * p["H"] / (r * K)
        if disc < 0.0:
            return {"stable": None, "unstable": None}
        root = math.sqrt(disc)
        return {"stable": 0.5 * K * (1.0 + root),
                "unstable": 0.5 * K * (1.0 - root)}
    E = p["E"]
    if E < r:
        return {"stable": K * (1.0 - E / r), "unstable": 0.0}
    return {"stable": 0.0, "unstable": None}


def fprime(N, p):
    """Derivative of the right-hand side with respect to N."""
    d = p["r"] * (1.0 - 2.0 * N / p["K"])
    if p["policy"] == "effort":
        d -= p["E"]
    return d


def collapse_time_exact(p, N0):
    """Collapse time for the quota policy with H > H_c.

    f(N) = -a [ (N - K/2)^2 + w^2 ],  a = r/K,  w^2 = (H - H_c)/a
    t_c = [ atan((N0 - K/2)/w) + atan(K/(2w)) ] / sqrt(a (H - H_c))
    """
    if p["policy"] == "effort":
        raise ValueError("quota policy only")
    Hc = H_critical(p)
    if p["H"] <= Hc:
        raise ValueError("needs H > H_c")
    a = p["r"] / p["K"]
    w = math.sqrt((p["H"] - Hc) / a)
    return (math.atan((N0 - 0.5 * p["K"]) / w)
            + math.atan(0.5 * p["K"] / w)) / math.sqrt(a * (p["H"] - Hc))




def quantities(p, result):
    """Numbers reported by the simulator: N_final, equilibrium, collapse_time."""
    eq = equilibria(p)["stable"]
    return {
        "N_final": float(result["N"][-1]),
        "equilibrium": eq,
        "collapse_time": result["collapse_time"],
    }




def sample(rng):
    """One random, physically valid scenario for the dataset.

    r in [0.2, 1.0] 1/year, K in [500, 2000] tonnes.
    Quota H in [0, 0.9 H_c], so two equilibria exist.
    N0 between 1.1 N_unstable and K, so the stock does not start in the
    collapse region. (A start below N_unstable is covered in the scenarios.)
    """
    r = rng.uniform(0.2, 1.0)
    K = rng.uniform(500.0, 2000.0)
    p = make_params(policy="quota", r=r, K=K, H=0.0)
    p["H"] = rng.uniform(0.0, 0.9 * H_critical(p))
    low = max(1.1 * equilibria(p)["unstable"], 0.05 * K)
    N0 = rng.uniform(low, K)
    t_end = 10.0 / (0.2 * r)          
    dt = 0.05
    t_end = round(t_end / dt) * dt
    return {"params": p, "N0": N0, "t_span": (0.0, t_end), "dt": dt,
            "method": "rk4"}




def extra_check_basin(p, N0_above, N0_below, dt=0.05, t_end=200.0):
    """Starting above N_unstable the stock tends to N_stable; below, it collapses."""
    eq = equilibria(p)
    up = simulate(p, N0_above, (0.0, t_end), dt)
    down = simulate(p, N0_below, (0.0, t_end), dt)
    ok_up = up["collapse_time"] is None and abs(up["N"][-1] - eq["stable"]) < 1e-3 * p["K"]
    ok_down = down["collapse_time"] is not None
    return ok_up and ok_down


EXTRA_CHECK = extra_check_basin


if __name__ == "__main__":
    P = make_params()
    eq = equilibria(P)
    print("N+ =", round(eq["stable"], 1), " N- =", round(eq["unstable"], 1),
          " H_c =", H_critical(P))
    res = simulate(P, 800.0, (0.0, 50.0), 0.1)
    print("N(50) =", round(res["N"][-1], 4))
