"""Click points with matplotlib's ginput and save them as JSON in data/.

pairs:   python click_points.py images/a.jpg images/b.jpg data/pair.json [n_points]
corners: python click_points.py images/door.jpg data/door.json [n_points]   (rectification)

Pairs: both images are shown side by side. Click a feature in the LEFT image,
then the same feature in the RIGHT image; it is numbered in both so you can
see what is done. Repeat n times, or press Enter to finish early. A click on
the wrong side is ignored and asked again.
"""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def click_corners(path, n):
    plt.imshow(Image.open(path))
    plt.title(f"click {n} points in {path}")
    pts = plt.ginput(n, timeout=0)
    plt.close()
    return pts


def click_pairs(path1, path2, n):
    im1, im2 = np.asarray(Image.open(path1)), np.asarray(Image.open(path2))
    w = im1.shape[1]
    canvas = np.zeros((max(im1.shape[0], im2.shape[0]), w + im2.shape[1], 3), np.uint8)
    canvas[:im1.shape[0], :w] = im1
    canvas[:im2.shape[0], w:] = im2
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(canvas)
    ax.axvline(w - 0.5, color="w", lw=1)
    ax.axis("off")
    fig.tight_layout()
    p1, p2 = [], []
    while len(p1) < n:
        ax.set_title(f"point {len(p1) + 1}/{n}: click LEFT, then the same feature on the RIGHT  (Enter = finish)")
        fig.canvas.draw()
        got = plt.ginput(2, timeout=0)
        if len(got) < 2:
            break
        (xa, ya), (xb, yb) = got
        if not (xa < w <= xb):
            print("ignored: first click must be on the left image and second on the right")
            continue
        p1.append((xa, ya))
        p2.append((xb - w, yb))
        for (x, y), c in ((got[0], "r"), (got[1], "r")):
            ax.plot(x, y, "o", color=c, ms=6, mfc="none")
            ax.annotate(str(len(p1)), (x, y), color="yellow", fontsize=11, xytext=(5, -5), textcoords="offset points")
    plt.close(fig)
    return p1, p2


def main(args):
    if args[1].endswith(".json"):
        out, pts = args[1], {"im1": click_corners(args[0], int(args[2]) if len(args) > 2 else 4)}
    else:
        out = args[2]
        p1, p2 = click_pairs(args[0], args[1], int(args[3]) if len(args) > 3 else 12)
        pts = {"im1": p1, "im2": p2}
    with open(out, "w") as f:
        json.dump(pts, f, indent=1)
    print("saved", out, {k: len(v) for k, v in pts.items()})


if __name__ == "__main__":
    main(sys.argv[1:])
