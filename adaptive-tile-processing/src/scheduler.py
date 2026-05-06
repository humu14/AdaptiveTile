"""Schedulers for tile processing: Static, Dynamic, Predictive, Hybrid."""
from __future__ import annotations

import queue
from typing import Optional
import numpy as np

from src.tiling import TileInfo


class StaticScheduler:
    """Round-robin static assignment of tiles to workers."""

    def assign(self, tiles: list[TileInfo], n_workers: int) -> list[list[TileInfo]]:
        """Assign tiles to workers in round-robin fashion.

        Returns a list of n_workers batches (each a list of TileInfo).
        """
        batches: list[list[TileInfo]] = [[] for _ in range(n_workers)]
        for i, tile in enumerate(tiles):
            batches[i % n_workers].append(tile)
        return batches


class DynamicScheduler:
    """Dynamic work queue — all tiles ordered by tile_id."""

    def get_queue(self, tiles: list[TileInfo]) -> list[TileInfo]:
        """Return tiles in tile_id order (workers pull from queue dynamically)."""
        return sorted(tiles, key=lambda t: t.tile_id)


class PredictiveScheduler:
    """Sort tiles by predicted processing time (longest first = LPT heuristic)."""

    def get_queue(self, tiles: list[TileInfo], predictions: np.ndarray) -> list[TileInfo]:
        """Return tiles sorted by descending predicted processing time.

        Args:
            tiles: list of TileInfo
            predictions: array of predicted times (ms), same order as tiles
        """
        order = np.argsort(predictions)[::-1]
        return [tiles[i] for i in order]


class HybridScheduler:
    """Start with predictive ordering, then adapt based on actual completions."""

    def __init__(self) -> None:
        self._actual_times: dict[int, float] = {}
        self._predicted_times: dict[int, float] = {}
        self._errors: list[float] = []

    def get_initial_queue(
        self, tiles: list[TileInfo], predictions: np.ndarray
    ) -> list[TileInfo]:
        """Return tiles sorted by descending predicted time (initial ordering)."""
        for i, tile in enumerate(tiles):
            self._predicted_times[tile.tile_id] = float(predictions[i])
        order = np.argsort(predictions)[::-1]
        return [tiles[i] for i in order]

    def record_completion(
        self, tile_id: int, predicted_ms: float, actual_ms: float
    ) -> None:
        """Record a completed tile's predicted vs actual time for adaptation."""
        self._actual_times[tile_id] = actual_ms
        self._predicted_times[tile_id] = predicted_ms
        self._errors.append(abs(actual_ms - predicted_ms))

    @property
    def mean_absolute_error(self) -> float:
        if not self._errors:
            return 0.0
        return float(np.mean(self._errors))
