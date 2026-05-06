import numpy as np
from src.tiling import TileInfo


def merge_tiles(
    results: list[tuple[TileInfo, np.ndarray]],
    image_shape: tuple[int, int],
    halo: int,
) -> np.ndarray:
    h, w = image_shape
    first_data = results[0][1]
    if first_data.ndim == 3:
        out = np.zeros((h, w, first_data.shape[2]), dtype=first_data.dtype)
    else:
        out = np.zeros((h, w), dtype=first_data.dtype)

    for tile, processed in results:
        th = processed.shape[0]
        tw = processed.shape[1]
        crop_top = tile.top_halo
        crop_bottom = th - tile.bottom_halo if tile.bottom_halo > 0 else th
        crop_left = tile.left_halo
        crop_right = tw - tile.right_halo if tile.right_halo > 0 else tw

        interior = processed[crop_top:crop_bottom, crop_left:crop_right]
        if interior.ndim == 3:
            out[tile.y_start:tile.y_end, tile.x_start:tile.x_end] = interior
        else:
            out[tile.y_start:tile.y_end, tile.x_start:tile.x_end] = interior

    return out
