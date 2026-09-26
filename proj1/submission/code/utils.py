# CS180 (CS280A) Project 1 - shared alignment functions.
#
# euclidean_align / pyramind_align are metric-agnostic: they just compare
# whatever two 2D arrays they're given (raw pixels, or a gradient image, or
# anything else with the same shape as the reference).

import numpy as np
import skimage as sk


# align im1 to im2 using a simple exhaustive search over a window of shifts
def euclidean_align(im1, im2, center=(0, 0), window=(-15, 15), square_frac=0.5):
    h, w = im2.shape
    ch, cw = int(h * square_frac / 2), int(w * square_frac / 2)
    cy, cx = h // 2, w // 2
    y0, y1 = cy - ch, cy + ch
    x0, x1 = cx - cw, cx + cw
    # only score a center square region, so borders/wraparound dont skew the match
    im2_sq = im2[y0:y1, x0:x1]

    min_ssd = np.inf
    best_x, best_y = 0, 0

    for dx in range(window[0], window[1] + 1):
        for dy in range(window[0], window[1] + 1):
            new_x, new_y = center[0] + dx, center[1] + dy
            # shift by indexing instead of np.roll, so there's no wraparound
            shifted_sq = im1[y0 - new_x:y1 - new_x, x0 - new_y:x1 - new_y]
            # euclidean distance: sq root of sum of squared differences
            ssd = np.sqrt(np.sum((shifted_sq - im2_sq) ** 2))
            if ssd < min_ssd:
                min_ssd = ssd
                best_x = new_x
                best_y = new_y

    return best_x, best_y


def pyramind_align(im1, im2, center=(0, 0), window=4, min_size=100):
    if min(im1.shape) < min_size * 2:
        return euclidean_align(im1, im2, center=center, window=(-window, window))

    # scale down the images
    small_im1 = sk.transform.rescale(im1, 0.5, anti_aliasing=True)
    small_im2 = sk.transform.rescale(im2, 0.5, anti_aliasing=True)

    best_x, best_y = pyramind_align(small_im1, small_im2, center=(center[0] // 2, center[1] // 2),
                                     window=window, min_size=min_size)
    return euclidean_align(im1, im2, center=(best_x * 2, best_y * 2), window=(-window, window))


# np.roll wraps content around instead of leaving a gap, so whatever we
# shifted by is exactly how much wrapped-in garbage sits on that edge now.
# positive shift wraps in at the top/left, negative at the bottom/right.
# just crop off the biggest margin either channel needs per side.
def crop_wraparound(im, shift_g, shift_r):
    h, w = im.shape[:2]
    top = max(0, shift_g[0], shift_r[0])
    bottom = max(0, -shift_g[0], -shift_r[0])
    left = max(0, shift_g[1], shift_r[1])
    right = max(0, -shift_g[1], -shift_r[1])
    return im[top:h - bottom, left:w - right]


def load_channels(imname):
    import skimage.io as skio
    im = sk.img_as_float(skio.imread(imname))
    height = np.floor(im.shape[0] / 3.0).astype(int)
    b = im[:height]
    g = im[height:2 * height]
    r = im[2 * height:3 * height]
    return b, g, r
