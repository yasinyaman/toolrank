"""Paired tests for two runs over the same queries (``toolrank compare --paired``).

A runs file (``toolrank eval --runs-out``) carries one row per query, so two scorers are compared
query by query — the pairing removes the between-query variance a two-sample test would drown in.
P@1 and hit@5 are per-query 0/1, so they get the sign test; NDCG@10 is graded, so it gets a paired
permutation (sign-flip) test on the per-query differences: exact over all 2**n assignments up to
12 queries, a seeded Monte Carlo above.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

EXACT_UP_TO = 12  # 2**12 = 4096 assignments: cheaper to be exact than to sample


def sign_test(a: Sequence[float], b: Sequence[float]) -> tuple[int, int, float]:
    """Wins of ``a`` over ``b``, losses, and the two-sided exact sign-test p-value.

    Ties carry no information and drop out; under the null the wins are binomial(1/2) over what
    is left.
    """
    if len(a) != len(b):
        raise ValueError(f"paired rows differ: {len(a)} vs {len(b)}")
    wins = sum(1 for x, y in zip(a, b, strict=True) if x > y)
    losses = sum(1 for x, y in zip(a, b, strict=True) if x < y)
    n = wins + losses
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return wins, losses, min(1.0, 2 * tail)


def permutation_test(a: Sequence[float], b: Sequence[float], *, perms: int = 10000, seed: int = 0) -> float:
    """Two-sided paired permutation p-value on the mean difference ``mean(a) - mean(b)``.

    Each query's difference keeps its size; the null says its sign is chance. The Monte Carlo
    counts the observed assignment too (the +1 correction), so p never reports 0.
    """
    if len(a) != len(b):
        raise ValueError(f"paired rows differ: {len(a)} vs {len(b)}")
    d = [x - y for x, y in zip(a, b, strict=True)]
    n = len(d)
    if n == 0:
        return 1.0
    obs = abs(sum(d))
    eps = 1e-9 * max(1.0, obs)  # sign flips reorder the additions; floats need slack
    if n <= EXACT_UP_TO:
        reach = sum(
            1
            for bits in range(1 << n)
            if abs(sum(x if bits >> i & 1 else -x for i, x in enumerate(d))) >= obs - eps
        )
        return reach / (1 << n)
    rng = random.Random(seed)
    reach = 1  # the observed assignment
    for _ in range(perms):
        reach += abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= obs - eps
    return reach / (perms + 1)
