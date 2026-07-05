"""Geographic clustering + route ordering for the Manchester visit list.

Two things the ranking and the day-plan need:

1. **Walkable zones** — group shops that sit within a short walk of each other so a
   day of visits doesn't zig-zag across the city. Cluster density also feeds the
   rank: a shop with several other genuine shops around it is worth more of your
   day than an isolated one.
2. **A route** — nearest-neighbour ordering within a zone so you actually know the
   walking order and rough distance.

Distances use the haversine formula (good enough at city scale). For 40 genuine
shops an all-pairs matrix is trivial; at 30k rows you'd swap the O(n²) matrix for a
BallTree/`DBSCAN(metric="haversine")` or a geohash-grid blocking pass — the
`assign_clusters` signature stays the same, only its internals change.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance in km. Scalar or numpy-broadcastable inputs."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def pairwise_m(lat, lng) -> np.ndarray:
    """N×N matrix of pairwise distances in metres."""
    lat = np.asarray(lat, dtype=float)
    lng = np.asarray(lng, dtype=float)
    return haversine_km(lat[:, None], lng[:, None], lat[None, :], lng[None, :]) * 1000.0


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def assign_clusters(df: pd.DataFrame, radius_m: float = 500.0,
                    lat_col: str = "lat", lng_col: str = "lng",
                    ) -> Tuple[pd.Series, pd.Series]:
    """Single-linkage cluster of shops within ``radius_m`` of each other.

    Returns two Series aligned to ``df.index``:
    - ``cluster`` — integer zone id (0 is the largest/most-central-ranked zone,
      ordered by size then descending),
    - ``neighbours`` — count of *other* shops within ``radius_m`` (density signal).

    Rows with missing coordinates get a unique singleton cluster and 0 neighbours.
    """
    n = len(df)
    if n == 0:
        return pd.Series([], dtype=int), pd.Series([], dtype=int)

    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy()
    lng = pd.to_numeric(df[lng_col], errors="coerce").to_numpy()
    valid = ~(np.isnan(lat) | np.isnan(lng))

    dist = pairwise_m(lat, lng)
    within = (dist <= radius_m) & valid[:, None] & valid[None, :]
    np.fill_diagonal(within, False)

    uf = _UnionFind(n)
    ii, jj = np.where(np.triu(within, k=1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        uf.union(i, j)

    # invalid-coord rows stay singletons
    raw = [uf.find(i) if valid[i] else -(i + 1) for i in range(n)]

    # relabel: biggest cluster first (that's "zone 0", the anchor for the day plan)
    sizes: dict = {}
    for r in raw:
        sizes[r] = sizes.get(r, 0) + 1
    order = sorted(sizes, key=lambda r: (-sizes[r], r))
    remap = {r: k for k, r in enumerate(order)}
    cluster = pd.Series([remap[r] for r in raw], index=df.index, name="cluster")

    neighbours = pd.Series(within.sum(axis=1), index=df.index, name="neighbours")
    return cluster, neighbours


def nearest_neighbour_route(df: pd.DataFrame, start_index=None,
                            lat_col: str = "lat", lng_col: str = "lng",
                            ) -> Tuple[List, float]:
    """Greedy nearest-neighbour visiting order for the rows in ``df``.

    Returns (ordered list of df index labels, total walking distance in metres).
    Starts from ``start_index`` if given (e.g. the top-ranked shop), else row 0.
    """
    n = len(df)
    if n == 0:
        return [], 0.0
    labels = list(df.index)
    dist = pairwise_m(pd.to_numeric(df[lat_col], errors="coerce"),
                      pd.to_numeric(df[lng_col], errors="coerce"))

    start_pos = labels.index(start_index) if start_index in labels else 0
    unvisited = set(range(n))
    order_pos = [start_pos]
    unvisited.discard(start_pos)
    total = 0.0
    current = start_pos
    while unvisited:
        nxt = min(unvisited, key=lambda k: dist[current, k])
        total += float(dist[current, nxt])
        order_pos.append(nxt)
        unvisited.discard(nxt)
        current = nxt
    return [labels[p] for p in order_pos], total
