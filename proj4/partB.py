from helper import harris_corners, anms, extract_descriptors, match_features, ransac_homography, reprojection_error, shrink
from partA import SETS, SCALE, montage_row, load_img, save_img, pair_name, chain_homographies, build_mosaic, timed
import cv2
import numpy as np
import os

# Corners kept by ANMS per photo, how many of the strongest Harris corners go into it (ANMS compares every pair, so the
# thousands of weak corners are cut first), the Lowe ratio threshold of the matching, and the RANSAC iterations and the
# distance (in pixels of the shrunk photos) within which a match counts as an inlier
N_KEEP = 3000
N_CANDIDATES = 6000
RATIO = 0.7
ITERATIONS = 3000
THRESHOLD = 3

def gray_of(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def detect(image):
    # Harris corners of a photo, then the ANMS selection among its strongest ones. Returns the corners before ANMS
    # (n x 2 as (x, y), strongest first) and the ones ANMS kept
    pts, strengths = harris_corners(gray_of(image))
    kept, _ = anms(pts[:N_CANDIDATES], strengths[:N_CANDIDATES], N_KEEP)
    return pts, kept

def draw_points(image, pts, color, radius=4):
    marked = image.copy()
    for x, y in np.round(pts).astype(int):
        cv2.circle(marked, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(marked, (x, y), radius, (0, 0, 0), 1, cv2.LINE_AA)
    return marked

def draw_matches(im1, im2, pts1, pts2, inliers=None, limit=150):
    # The two photos side by side with a line for every match, or for a random sample of limit of them when there are
    # more, since hundreds of lines hide the photos. With inliers, the inlier matches are green and the rest red
    height = max(im1.shape[0], im2.shape[0])
    canvas = np.zeros((height, im1.shape[1] + im2.shape[1], 3), dtype=np.uint8)
    canvas[:im1.shape[0], :im1.shape[1]] = im1
    canvas[:im2.shape[0], im1.shape[1]:] = im2
    offset = np.array([im1.shape[1], 0])
    inliers = np.ones(len(pts1), dtype=bool) if inliers is None else inliers
    if len(pts1) > limit:
        sample = np.sort(np.random.default_rng(0).choice(len(pts1), limit, replace=False))
        pts1, pts2, inliers = pts1[sample], pts2[sample], inliers[sample]
    # Outliers first, so the inliers are drawn on top
    for i in np.argsort(inliers):
        color = (0, 220, 0) if inliers[i] else (0, 0, 255)
        p1, p2 = tuple(np.round(pts1[i]).astype(int)), tuple(np.round(pts2[i] + offset).astype(int))
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        for p in (p1, p2):
            cv2.circle(canvas, p, 4, color, -1, cv2.LINE_AA)
    return canvas

def features_of(imgs):
    # Corners, ANMS corners and descriptors of every photo of a set
    features = []
    for image in imgs:
        pts, kept = detect(image)
        features.append((pts, kept, extract_descriptors(gray_of(image), kept)))
    return features

def load_set(path_to_img_folder, numbers):
    return [shrink(load_img(path_to_img_folder, n), SCALE) for n in numbers]

def match_pair(features, k):
    # Ratio test matches between photo k and photo k + 1 as the matched corners of each, n x 2 and n x 2
    (_, kept1, desc1), (_, kept2, desc2) = features[k], features[k + 1]
    pairs = match_features(desc1, desc2, RATIO)
    return kept1[pairs[:, 0]], kept2[pairs[:, 1]]

def run_partB1(path_to_img_folder, path_to_out_folder):
    # Harris corners on the first photo of every set, drawn all together and then after ANMS, and the ANMS result with
    # the same number of corners picked by Harris strength alone, to show what the suppression radius changes
    os.makedirs(path_to_out_folder, exist_ok=True)
    for set_name, numbers, _ in SETS:
        image = load_set(path_to_img_folder, numbers[:1])[0]
        pts, kept = detect(image)
        strongest = pts[:N_KEEP]
        print(f"{set_name}: {len(pts)} Harris corners, ANMS keeps {len(kept)} of the strongest {N_CANDIDATES}")
        save_img(os.path.join(path_to_out_folder, f"harris_{set_name}.jpg"), draw_points(image, pts, (0, 0, 255), 3))
        save_img(os.path.join(path_to_out_folder, f"strongest_{set_name}.jpg"), draw_points(image, strongest, (0, 255, 255)))
        save_img(os.path.join(path_to_out_folder, f"anms_{set_name}.jpg"), draw_points(image, kept, (0, 255, 0)))

def run_partB2(path_to_img_folder, path_to_out_folder):
    # The descriptors of a few corners kept by ANMS: the 40 x 40 window each is sampled from (top) and the 8 x 8 normalized
    # descriptor, stretched to the full gray range (bottom)
    os.makedirs(path_to_out_folder, exist_ok=True)
    for set_name, numbers, _ in SETS:
        image = load_set(path_to_img_folder, numbers[:1])[0]
        gray = gray_of(image)
        _, kept = detect(image)
        desc = extract_descriptors(gray, kept)
        rng = np.random.default_rng(0)
        chosen = np.sort(rng.choice(len(kept), 8, replace=False))
        windows, cells = [], []
        for i in chosen:
            x, y = np.round(kept[i]).astype(int)
            windows.append(cv2.resize(image[y - 20:y + 20, x - 20:x + 20], (160, 160), interpolation=cv2.INTER_NEAREST))
            cell = np.clip((desc[i] + 2.5) / 5 * 255, 0, 255).astype(np.uint8)
            cells.append(cv2.cvtColor(cv2.resize(cell, (160, 160), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR))
        gap = np.full((160, 8, 3), 255, dtype=np.uint8)
        top = np.hstack([p for w in windows for p in (w, gap)][:-1])
        bottom = np.hstack([p for c in cells for p in (c, gap)][:-1])
        save_img(os.path.join(path_to_out_folder, f"desc_{set_name}.png"), np.vstack([top, np.full((8, top.shape[1], 3), 255, dtype=np.uint8), bottom]))

def run_partB3(path_to_img_folder, path_to_out_folder):
    # Ratio test matches between the first two photos of every set, before RANSAC
    os.makedirs(path_to_out_folder, exist_ok=True)
    for set_name, numbers, _ in SETS:
        imgs = load_set(path_to_img_folder, numbers[:2])
        features = features_of(imgs)
        pts1, pts2 = match_pair(features, 0)
        print(f"{pair_name(set_name, numbers, 0)}: {len(pts1)} matches of {N_KEEP} corners at ratio {RATIO}")
        save_img(os.path.join(path_to_out_folder, f"matches_{pair_name(set_name, numbers, 0)}.jpg"), draw_matches(imgs[0], imgs[1], pts1, pts2))

def run_partB4(path_to_img_folder, path_to_out_folder):
    # Match every neighboring pair of a set, estimate its homography with RANSAC, chain the homographies into the
    # reference photo's frame and blend, exactly as for the manual mosaics. Writes the RANSAC inliers of the first pair
    # and the automatic mosaic next to the manual one
    os.makedirs(path_to_out_folder, exist_ok=True)
    for set_name, numbers, reference in SETS:
        imgs = load_set(path_to_img_folder, numbers)
        features = features_of(imgs)
        pair_H = []
        for k in range(len(numbers) - 1):
            name = pair_name(set_name, numbers, k)
            pts1, pts2 = match_pair(features, k)
            (H, inliers), t = timed(ransac_homography, pts1, pts2, ITERATIONS, THRESHOLD)
            error = reprojection_error(H, pts1[inliers], pts2[inliers])
            print(f"{name}: {len(pts1)} matches, {inliers.sum()} inliers, inlier error mean {error.mean():.2f}px, RANSAC {t:.2f}s")
            pair_H.append(H)
            if k == 0:
                save_img(os.path.join(path_to_out_folder, f"ransac_{name}.jpg"), draw_matches(imgs[k], imgs[k + 1], pts1, pts2, inliers))

        mosaic = build_mosaic(imgs, chain_homographies(pair_H, reference), set_name)
        save_img(os.path.join(path_to_out_folder, f"auto_{set_name}.jpg"), mosaic)
        manual = cv2.imread(os.path.join(path_to_out_folder, f"mosaic_{set_name}_laplacian.jpg"))
        save_img(os.path.join(path_to_out_folder, f"compare_{set_name}.jpg"), montage_row([manual, mosaic], 600, labels=["Manual points", "Automatic features"]))
