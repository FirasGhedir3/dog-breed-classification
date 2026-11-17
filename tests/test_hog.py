import cv2
import numpy as np

from dogbreeds.hog import compute_hog, compute_hog_fast, sobel

RNG = np.random.default_rng(0)


def test_sobel_matches_opencv():
    img = RNG.random((64, 64))
    gx, _, _, angle = sobel(img)
    reference = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    assert np.allclose(gx, reference)
    assert angle.min() >= 0 and angle.max() < np.pi + 1e-12


def test_vertical_edge_votes_for_first_bin():
    img = np.zeros((64, 64))
    img[:, 32:] = 1.0
    hog = compute_hog(img).reshape(4, 4, 8)
    assert hog.sum(axis=(0, 1))[1:].sum() < 1e-9
    assert hog.sum(axis=(0, 1))[0] > 0


def test_horizontal_edge_votes_for_middle_bin():
    img = np.zeros((64, 64))
    img[32:, :] = 1.0
    assert np.argmax(compute_hog(img).reshape(4, 4, 8).sum(axis=(0, 1))) == 4


def test_loop_and_fast_versions_agree():
    for _ in range(5):
        img = RNG.random((64, 64))
        slow, fast = compute_hog(img), compute_hog_fast(img)
        assert slow.shape == (128,)
        assert np.allclose(slow, fast)


def test_votes_add_up_to_total_magnitude():
    img = RNG.random((64, 64))
    assert np.isclose(compute_hog(img).sum(), sobel(img)[2].sum())


def test_angle_exactly_pi_does_not_overflow_last_bin():
    # un angle négatif minuscule devient exactement pi après le modulo
    angle = np.arctan2(-1e-300, 1.0) % np.pi
    assert angle == np.pi
    img = np.zeros((16, 16))  # une image uniforme doit tourner sans IndexError
    assert compute_hog(img, cell=16).shape == (8,)
