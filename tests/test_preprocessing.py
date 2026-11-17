import numpy as np
import pytest

from dogbreeds.preprocessing import letterbox, resize_stretch


@pytest.mark.parametrize("shape", [(30, 90), (90, 30), (64, 64), (200, 120), (10, 10)])
def test_letterbox_output_is_square_and_keeps_ratio(shape):
    img = np.random.default_rng(0).integers(1, 255, shape).astype(np.uint8)
    out = letterbox(img, size=64)
    assert out.shape == (64, 64)
    # le plus grand côté remplit le canevas
    rows, cols = np.where(out > 0)
    assert max(rows.max() - rows.min() + 1, cols.max() - cols.min() + 1) >= 62


def test_letterbox_centers_the_image():
    img = np.full((20, 80), 200, dtype=np.uint8)
    out = letterbox(img, size=64)
    rows = np.where(out.max(axis=1) > 0)[0]
    assert abs(rows.min() - (63 - rows.max())) <= 1


def test_letterbox_rejects_unknown_padding():
    with pytest.raises(ValueError):
        letterbox(np.ones((10, 20), dtype=np.uint8), pad="nope")


def test_stretch_shape():
    assert resize_stretch(np.ones((30, 90), dtype=np.uint8)).shape == (64, 64)
