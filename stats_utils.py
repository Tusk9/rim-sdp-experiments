"""
stats_utils.py — small, dependency-free statistics used by the Step 2/3 analyses
(numpy only, so nothing new needs installing on the GPU box).

  rankdata(x)              average ranks (ties share the mean rank), like scipy.stats.rankdata
  spearman(x, y)           rho, two-sided p from the t approximation, and a permutation p
  cohen_kappa(a, b)        Cohen's kappa for two label lists
  confusion(a, b, labels)  counts[i][j] = #(a == labels[i] and b == labels[j])
"""
from __future__ import annotations

import math

import numpy as np


def rankdata(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum()) / den if den > 0 else float("nan")


def _t_pvalue(r: float, n: int) -> float:
    """Two-sided p for H0 rho = 0 using t = r sqrt((n-2)/(1-r^2)) with n-2 df."""
    if n < 3 or math.isnan(r):
        return float("nan")
    if abs(r) >= 1:
        return 0.0
    t = abs(r) * math.sqrt((n - 2) / (1 - r * r))
    df = n - 2
    # regularized incomplete beta via continued fraction (Numerical Recipes betacf)
    x = df / (df + t * t)
    return _betai(df / 2.0, 0.5, x)


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1 - front * _betacf(b, a, 1 - x) / b


def _betacf(a: float, b: float, x: float, itmax: int = 300, eps: float = 3e-14) -> float:
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(1 + aa / c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(1 + aa / c) > 1e-300 else 1e-300
        de = d * c; h *= de
        if abs(de - 1) < eps:
            break
    return h


def spearman(x, y, n_perm: int = 10000, seed: int = 3407) -> dict:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return {"rho": None, "p_t": None, "p_perm": None, "n": n}
    rx, ry = rankdata(x), rankdata(y)
    rho = _pearson(rx, ry)
    out = {"rho": round(rho, 4), "p_t": _t_pvalue(rho, n), "n": n}
    if n_perm and not math.isnan(rho):
        rng = np.random.default_rng(seed)
        ryc = ry - ry.mean(); rxc = rx - rx.mean()
        den = math.sqrt(float((rxc ** 2).sum()) * float((ryc ** 2).sum()))
        hits = 0
        for _ in range(n_perm):
            if abs(float((rxc * rng.permutation(ryc)).sum()) / den) >= abs(rho) - 1e-12:
                hits += 1
        out["p_perm"] = (hits + 1) / (n_perm + 1)
    else:
        out["p_perm"] = None
    return out


def cohen_kappa(a, b) -> float | None:
    a = list(a); b = list(b)
    if not a or len(a) != len(b):
        return None
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else 1.0


def confusion(a, b, labels) -> list[list[int]]:
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for x, y in zip(a, b):
        if x in idx and y in idx:
            m[idx[x]][idx[y]] += 1
    return m
