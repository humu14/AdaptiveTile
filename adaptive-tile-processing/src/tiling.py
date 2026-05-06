import math
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TileInfo:
    tile_id: int
    row: int
    col: int
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    left_halo: int
    top_halo: int
    right_halo: int
    bottom_halo: int
    data: np.ndarray = field(repr=False)


def extract_tiles(image: np.ndarray, tile_size: int, halo: int) -> list[TileInfo]:
    h, w = image.shape[:2]
    n_rows = math.ceil(h / tile_size)
    n_cols = math.ceil(w / tile_size)
    tiles = []
    tid = 0
    for r in range(n_rows):
        for c in range(n_cols):
            x_start = c * tile_size
            y_start = r * tile_size
            x_end = min(x_start + tile_size, w)
            y_end = min(y_start + tile_size, h)

            xs_halo = max(0, x_start - halo)
            ys_halo = max(0, y_start - halo)
            xe_halo = min(w, x_end + halo)
            ye_halo = min(h, y_end + halo)

            data = image[ys_halo:ye_halo, xs_halo:xe_halo]

            tiles.append(TileInfo(
                tile_id=tid, row=r, col=c,
                x_start=x_start, y_start=y_start,
                x_end=x_end, y_end=y_end,
                left_halo=x_start - xs_halo,
                top_halo=y_start - ys_halo,
                right_halo=xe_halo - x_end,
                bottom_halo=ye_halo - y_end,
                data=data,
            ))
            tid += 1
    return tiles
