from helper import computeH, build_system, apply_H, reprojection_error, order_corners, output_bbox, warpImageNearestNeighbor, warpImageBilinear, shrink, place_on_canvas, blend_laplacian
import cv2
import json
import numpy as np
import os
import time

# The clicked points are in the full size photos, the mosaics are built from copies shrunk to this fraction to keep
# the canvases (and the blending) fast
SCALE = 0.5

# (name, photo numbers in left to right order, index of the photo the others are warped into). The reference is a
# middle photo, so the warps are as small as possible. Only photos that fit one planar mosaic are used: with all six
# lounge or plaza photos the outer ones are so oblique that the canvas grows to over 10000 pixels and the far edges
# stretch into long wedges
SETS = [
    ("lounge", [1505, 1506, 1507, 1508], 1),
    ("toy", [1527, 1528, 1529, 1530, 1531], 2),
    ("plaza", [1563, 1564, 1565, 1566], 1),
]

# (name, photo number, clicked corners file, width and height of the rectangle the corners are sent to, row and column
# of the patch that is enlarged to compare the interpolation methods). The sizes are the proportions of a door (about
# 0.9 m by 2 m) and of the window pane, not measured exactly. The patches are picked on edges, since a flat area
# looks the same with either method
RECTIFY = [
    ("door", 1510, "door", (500, 1100), (366, 166)),
    ("window", 1509, "window", (560, 800), (570, 130)),
]

def timed(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start

def save_img(path, image):
    # Encode by file extension (png for the lossless comparisons, jpg for photos)
    with open(path, "wb") as f:
        f.write(cv2.imencode(os.path.splitext(path)[1], image)[1].tobytes())

def load_img(path_to_img_folder, number):
    img_path = os.path.join(path_to_img_folder, f"IMG_{number}.jpg")
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    return img

def load_points(name, key="im1"):
    with open(os.path.join("data", f"{name}.json")) as f:
        return np.array(json.load(f)[key])

def pair_name(set_name, numbers, k):
    return f"{set_name}_{numbers[k]}_{numbers[k + 1]}"

def draw_correspondences(im1, im2, pts1, pts2):
    # The two photos side by side with the matching points numbered, and a line between every pair
    height = max(im1.shape[0], im2.shape[0])
    canvas = np.zeros((height, im1.shape[1] + im2.shape[1], 3), dtype=np.uint8)
    canvas[:im1.shape[0], :im1.shape[1]] = im1
    canvas[:im2.shape[0], im1.shape[1]:] = im2
    offset = np.array([im1.shape[1], 0])
    for i, (p1, p2) in enumerate(zip(pts1, pts2)):
        p1, p2 = tuple(np.round(p1).astype(int)), tuple(np.round(p2 + offset).astype(int))
        cv2.line(canvas, p1, p2, (0, 255, 255), 1, cv2.LINE_AA)
        for p in (p1, p2):
            cv2.circle(canvas, p, 6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return canvas

def montage_row(images, height=500, gap=12):
    # The images in a row, all scaled to the same height, with a white gap between them
    resized = [cv2.resize(im, (round(im.shape[1] * height / im.shape[0]), height), interpolation=cv2.INTER_AREA) for im in images]
    pieces = []
    for im in resized:
        pieces += [im, np.full((height, gap, 3), 255, dtype=np.uint8)]
    return np.hstack(pieces[:-1])

def run_partA2(path_to_img_folder, path_to_out_folder):
    # Recover a homography for every neighboring pair of photos from the clicked points and print its matrix and how far
    # it lands from the clicked points. For the first pair of every set also write out the linear system and draw the
    # correspondences on the photos
    os.makedirs(path_to_out_folder, exist_ok=True)
    for set_name, numbers, _ in SETS:
        for k in range(len(numbers) - 1):
            name = pair_name(set_name, numbers, k)
            pts1, pts2 = load_points(name, "im1"), load_points(name, "im2")
            H = computeH(pts1, pts2)
            error = reprojection_error(H, pts1, pts2)
            print(f"{name}: {len(pts1)} points, reprojection error mean {error.mean():.2f}px, max {error.max():.2f}px")
            print(np.round(H, 5))

            if k == 0:
                A, b = build_system(pts1, pts2)
                with open(os.path.join(path_to_out_folder, f"system_{name}.txt"), "w") as f:
                    f.write(f"A (2n x 8), b (2n) for {name}\n")
                    for row, value in zip(A, b):
                        f.write(" ".join(f"{v:12.2f}" for v in row) + f" | {value:10.2f}\n")
                    f.write("H =\n" + np.array2string(H, precision=5, suppress_small=True) + "\n")

                im1 = load_img(path_to_img_folder, numbers[k])
                im2 = load_img(path_to_img_folder, numbers[k + 1])
                save_img(os.path.join(path_to_out_folder, f"corr_{name}.jpg"), shrink(draw_correspondences(im1, im2, pts1, pts2), SCALE))

def run_partA3(path_to_img_folder, path_to_out_folder):
    # Rectify a planar object: send its four clicked corners to the corners of an upright rectangle, so it is seen head
    # on. Warp it with both interpolation methods, time them, and compare a zoomed in patch of the two
    os.makedirs(path_to_out_folder, exist_ok=True)
    for name, number, points_file, (width, height), (r, c) in RECTIFY:
        img = load_img(path_to_img_folder, number)
        corners = order_corners(load_points(points_file))
        target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        H = computeH(corners, target)
        print(f"{name}: corner error {reprojection_error(H, corners, target).max():.2e}px, H =\n{np.round(H, 5)}")

        # Only the object's rectangle is wanted, so the output is the target rectangle itself
        bbox = (0, 0, width - 1, height - 1)
        (nearest, _, _), t_nearest = timed(warpImageNearestNeighbor, img, H, bbox)
        (bilinear, _, _), t_bilinear = timed(warpImageBilinear, img, H, bbox)
        print(f"{name}: output {nearest.shape[1]}x{nearest.shape[0]}  nearest {t_nearest:.3f}s  bilinear {t_bilinear:.3f}s")

        marked = img.copy()
        cv2.polylines(marked, [np.round(corners).astype(np.int32)], True, (0, 0, 255), 8, cv2.LINE_AA)
        save_img(os.path.join(path_to_out_folder, f"rect_{name}_input.jpg"), shrink(marked, SCALE))
        save_img(os.path.join(path_to_out_folder, f"rect_{name}_nearest.jpg"), nearest)
        save_img(os.path.join(path_to_out_folder, f"rect_{name}_bilinear.jpg"), bilinear)

        # A 60 x 60 patch of the output, blown up 8x without smoothing, so single pixels show
        zoom = [cv2.resize(im[r:r + 60, c:c + 60], None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST) for im in (nearest, bilinear)]
        save_img(os.path.join(path_to_out_folder, f"rect_{name}_zoom.png"), montage_row(zoom, 480))

def chain_homographies(pair_H, reference):
    # Homography of every photo into the reference photo's frame. pair_H[k] sends photo k to photo k + 1, so photos left
    # of the reference multiply the pairs going toward it, and photos to the right use the inverse of the pairs
    to_ref = [None] * (len(pair_H) + 1)
    to_ref[reference] = np.eye(3)
    for k in range(reference - 1, -1, -1):
        to_ref[k] = to_ref[k + 1] @ pair_H[k]
    for k in range(reference + 1, len(to_ref)):
        to_ref[k] = to_ref[k - 1] @ np.linalg.inv(pair_H[k - 1])
    return [H / H[2, 2] for H in to_ref]

def run_partA4(path_to_img_folder, path_to_out_folder):
    # Warp the photos of every set into the middle photo's frame (bilinear), place them on one canvas and blend them
    # with the multiresolution blend
    os.makedirs(path_to_out_folder, exist_ok=True)
    for set_name, numbers, reference in SETS:
        imgs = [shrink(load_img(path_to_img_folder, n), SCALE) for n in numbers]
        pair_H = []
        for k in range(len(numbers) - 1):
            name = pair_name(set_name, numbers, k)
            pair_H.append(computeH(load_points(name, "im1") * SCALE, load_points(name, "im2") * SCALE))
        to_ref = chain_homographies(pair_H, reference)

        # One bounding box for all the warps, so every warped photo is already in place on the canvas
        boxes = np.array([output_bbox(im.shape, H) for im, H in zip(imgs, to_ref)])
        bbox = (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max())
        print(f"{set_name}: canvas {bbox[2] - bbox[0] + 1}x{bbox[3] - bbox[1] + 1}")
        if bbox[2] - bbox[0] > 12000 or bbox[3] - bbox[1] > 12000:
            raise ValueError(f"{set_name}: canvas is far too large, use fewer photos or a cylindrical projection")

        layers = [warpImageBilinear(im, H, bbox) for im, H in zip(imgs, to_ref)]
        images, alphas, _ = place_on_canvas(layers)
        save_img(os.path.join(path_to_out_folder, f"mosaic_{set_name}_sources.jpg"), montage_row(imgs, 400))
        mosaic, t = timed(blend_laplacian, images, alphas)
        print(f"{set_name}: blend {t:.1f}s")
        save_img(os.path.join(path_to_out_folder, f"mosaic_{set_name}_laplacian.jpg"), mosaic)
