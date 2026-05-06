import numpy as np
import pytest
from src.tiling import TileInfo, extract_tiles


def make_image(h=512, w=768, c=3):
    return np.random.randint(0, 256, (h, w, c), dtype=np.uint8)


def test_tile_count_divides_evenly():
    img = make_image(512, 512)
    tiles = extract_tiles(img, tile_size=256, halo=0)
    assert len(tiles) == 4  # 2x2 grid


def test_tile_count_with_remainder():
    img = make_image(500, 700)
    tiles = extract_tiles(img, tile_size=256, halo=0)
    assert len(tiles) == 6  # ceil(500/256)*ceil(700/256) = 2*3 = 6


def test_tile_data_shape_with_halo():
    img = make_image(512, 768)
    tiles = extract_tiles(img, tile_size=256, halo=16)
    t = tiles[0]
    assert t.data.shape[0] == t.y_end - t.y_start + t.top_halo + t.bottom_halo
    assert t.data.shape[1] == t.x_end - t.x_start + t.left_halo + t.right_halo


def test_tile_ids_unique():
    img = make_image(512, 768)
    tiles = extract_tiles(img, tile_size=256, halo=16)
    ids = [t.tile_id for t in tiles]
    assert len(ids) == len(set(ids))


def test_tiles_cover_full_image():
    img = make_image(400, 600)
    tiles = extract_tiles(img, tile_size=200, halo=0)
    covered = np.zeros((400, 600), dtype=bool)
    for t in tiles:
        covered[t.y_start:t.y_end, t.x_start:t.x_end] = True
    assert covered.all()


from src.merge import merge_tiles


def test_merge_roundtrip_no_halo():
    img = make_image(512, 768)
    tiles = extract_tiles(img, tile_size=256, halo=0)
    results = [(t, t.data.copy()) for t in tiles]
    out = merge_tiles(results, img.shape[:2], halo=0)
    assert out.shape == img.shape[:2] + (img.shape[2],)
    np.testing.assert_array_equal(out, img)


def test_merge_roundtrip_with_halo():
    img = make_image(512, 768)
    tiles = extract_tiles(img, tile_size=256, halo=16)
    results = [(t, t.data.copy()) for t in tiles]
    out = merge_tiles(results, img.shape[:2], halo=16)
    np.testing.assert_array_equal(out, img)
