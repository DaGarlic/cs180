import cv2
import numpy as np
import scipy.ndimage

def build_system(im1_pts, im2_pts):
    # Stack the linear system A h = b for the 8 unknowns h = (a, b, c, d, e, f, g, h) of H, with H[2, 2] fixed to 1.
    # Every correspondence (x, y) -> (x', y') gives two rows:
    #   x' = a x + b y + c - g x x' - h y x'
    #   y' = d x + e y + f - g x y' - h y y'
    pts1 = np.array(im1_pts, dtype=float)
    pts2 = np.array(im2_pts, dtype=float)
    n = len(pts1)
    A = np.zeros((2 * n, 8))
    b = np.zeros(2 * n)
    x, y = pts1[:, 0], pts1[:, 1]
    xp, yp = pts2[:, 0], pts2[:, 1]
    A[0::2, 0], A[0::2, 1], A[0::2, 2] = x, y, 1
    A[0::2, 6], A[0::2, 7] = -x * xp, -y * xp
    A[1::2, 3], A[1::2, 4], A[1::2, 5] = x, y, 1
    A[1::2, 6], A[1::2, 7] = -x * yp, -y * yp
    b[0::2], b[1::2] = xp, yp
    return A, b

def computeH(im1_pts, im2_pts):
    # Homography H with im2_pts ~ H im1_pts. Both inputs are n x 2 arrays of (x, y). Four points solve the system exactly
    # but a few pixels of click error move H a lot, so use more and solve the overdetermined system by least squares
    if len(im1_pts) != len(im2_pts) or len(im1_pts) < 4:
        raise ValueError("computeH needs the same number of points in both images, and at least 4")
    A, b = build_system(im1_pts, im2_pts)
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.append(h, 1).reshape(3, 3)

def apply_H(H, pts):
    # Push n x 2 points through H and divide by the homogeneous coordinate
    pts = np.array(pts, dtype=float)
    mapped = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return mapped[:, :2] / mapped[:, 2:]

def reprojection_error(H, im1_pts, im2_pts):
    # Distance in pixels between where H sends each point of the first image and where it was clicked in the second
    return np.linalg.norm(apply_H(H, im1_pts) - np.array(im2_pts), axis=1)

def order_corners(pts):
    # Order 4 clicked corners as top-left, top-right, bottom-right, bottom-left, whatever order they were clicked in:
    # the top-left has the smallest x + y, the bottom-right the largest, and y - x is smallest at the top-right
    pts = np.array(pts, dtype=float)
    s, d = pts.sum(axis=1), pts[:, 1] - pts[:, 0]
    return pts[[np.argmin(s), np.argmin(d), np.argmax(s), np.argmax(d)]]

def output_bbox(shape, H):
    # Integer bounding box (x0, y0, x1, y1) of the image after H, found by pushing its four corners through H
    height, width = shape[:2]
    corners = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    mapped = apply_H(H, corners)
    x0, y0 = np.floor(mapped.min(axis=0)).astype(int)
    x1, y1 = np.ceil(mapped.max(axis=0)).astype(int)
    return x0, y0, x1, y1

def inverse_coords(H, bbox):
    # For every pixel of the output (inside bbox) find the position in the source image that lands on it. Looking up
    # the source this way (inverse warping) gives every output pixel a value, so there are no holes
    x0, y0, x1, y1 = bbox
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    mapped = np.linalg.inv(H) @ np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    return (mapped[0] / mapped[2]).reshape(xs.shape), (mapped[1] / mapped[2]).reshape(xs.shape)

def warpImageNearestNeighbor(im, H, bbox=None):
    # Inverse warp with nearest neighbor: round the source position to the closest pixel. Pixel (0, 0) is the center of
    # the first pixel. Returns the warped image, an alpha mask (1 where the output pixel came from inside im) and the
    # (x, y) of the output's top left corner in the target frame
    height, width = im.shape[:2]
    bbox = bbox or output_bbox(im.shape, H)
    src_x, src_y = inverse_coords(H, bbox)
    col, row = np.round(src_x).astype(int), np.round(src_y).astype(int)
    alpha = (col >= 0) & (col < width) & (row >= 0) & (row < height)
    warped = np.zeros(src_x.shape + im.shape[2:], dtype=im.dtype)
    warped[alpha] = im[row[alpha], col[alpha]]
    return warped, alpha.astype(float), (bbox[0], bbox[1])

def warpImageBilinear(im, H, bbox=None):
    # Inverse warp with bilinear interpolation: a weighted average of the four pixels around the source position,
    # where each is weighted by how close the position is to it. Returns the same three things as the nearest neighbor warp
    height, width = im.shape[:2]
    bbox = bbox or output_bbox(im.shape, H)
    src_x, src_y = inverse_coords(H, bbox)
    alpha = (src_x >= 0) & (src_x <= width - 1) & (src_y >= 0) & (src_y <= height - 1)
    x, y = src_x[alpha], src_y[alpha]

    # Top left of the four neighbors (kept one short of the last row and column so the other three exist)
    col = np.minimum(np.floor(x).astype(int), width - 2)
    row = np.minimum(np.floor(y).astype(int), height - 2)
    dx, dy = x - col, y - row
    if im.ndim == 3:
        dx, dy = dx[:, np.newaxis], dy[:, np.newaxis]
    im = im.astype(float)
    values = ((1 - dx) * (1 - dy) * im[row, col] + dx * (1 - dy) * im[row, col + 1]
              + (1 - dx) * dy * im[row + 1, col] + dx * dy * im[row + 1, col + 1])
    warped = np.zeros(src_x.shape + im.shape[2:])
    warped[alpha] = values
    return np.clip(np.round(warped), 0, 255).astype(np.uint8), alpha.astype(float), (bbox[0], bbox[1])

def shrink(image, scale):
    # Downscale an image (INTER_AREA so shrinking doesn't alias)
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

def gaussian_blur(image, sigma):
    return cv2.GaussianBlur(image, (0, 0), sigma)

def place_on_canvas(layers):
    # Paste warped layers, each (image, alpha, (x, y) of its top left corner), onto one canvas that covers all of them.
    # Returns the images and alphas as stacks the size of the canvas, and the canvas' top left corner
    x0 = min(offset[0] for _, _, offset in layers)
    y0 = min(offset[1] for _, _, offset in layers)
    x1 = max(offset[0] + alpha.shape[1] for _, alpha, offset in layers)
    y1 = max(offset[1] + alpha.shape[0] for _, alpha, offset in layers)
    images = np.zeros((len(layers), y1 - y0, x1 - x0, 3))
    alphas = np.zeros((len(layers), y1 - y0, x1 - x0))
    for k, (im, alpha, (x, y)) in enumerate(layers):
        images[k, y - y0:y - y0 + alpha.shape[0], x - x0:x - x0 + alpha.shape[1]] = im
        alphas[k, y - y0:y - y0 + alpha.shape[0], x - x0:x - x0 + alpha.shape[1]] = alpha
    return images, alphas, (x0, y0)

def edge_weights(alphas):
    # Weight of every pixel of every layer: its distance to the nearest empty pixel, so a layer counts most at its
    # center and fades to 0 at its border. Scaled so each layer's largest weight is 1
    weights = np.array([scipy.ndimage.distance_transform_edt(alpha > 0) for alpha in alphas])
    return weights / np.maximum(weights.max(axis=(1, 2), keepdims=True), 1)

def fill_empty(image, alpha):
    # Copy the nearest real pixel into the empty area around a warped layer. The Gaussian stack blurs the layer, and
    # blurring against black would pull dark halos into the mosaic along its border
    rows, cols = scipy.ndimage.distance_transform_edt(alpha == 0, return_distances=False, return_indices=True)
    return image[rows, cols]

def blend_laplacian(images, alphas, levels=4, sigma=8):
    # Multiresolution blend. Each pixel is given to the layer with the largest weight, that hard mask is blurred more
    # at every level, and each frequency band of the layers is mixed with the mask of its own level. Low frequencies
    # (exposure, color) are mixed over a wide seam and high frequencies (edges) over a narrow one, so the seam
    # disappears without ghosting
    weights = edge_weights(alphas)
    winner = weights.argmax(axis=0)
    sigmas = [sigma * 2 ** i for i in range(levels - 1)]
    weighted_bands = np.zeros((levels,) + images.shape[1:])
    mask_sums = np.zeros((levels,) + images.shape[1:3])
    for k, (im, alpha) in enumerate(zip(images, alphas)):
        im = fill_empty(im, alpha)
        mask = ((winner == k) & (alpha > 0)).astype(float)
        blurred = [im] + [gaussian_blur(im, s) for s in sigmas]
        for level in range(levels):
            # Level i is the band between Gaussian i and i + 1, and the last level is the blurriest image left over
            band = blurred[level] - blurred[level + 1] if level < levels - 1 else blurred[level]
            level_mask = gaussian_blur(mask, sigmas[level - 1]) if level > 0 else mask
            weighted_bands[level] += level_mask[..., np.newaxis] * band
            mask_sums[level] += level_mask

    # Normalize every level by the total mask so the layers' shares add up to 1, then add the levels back together
    mosaic = (weighted_bands / np.maximum(mask_sums, 1e-9)[..., np.newaxis]).sum(axis=0)
    mosaic[alphas.sum(axis=0) == 0] = 0
    return np.clip(np.round(mosaic), 0, 255).astype(np.uint8)
