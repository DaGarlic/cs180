# CS180 (CS280A) Project 1: Images of the Russian Empire
#
# Colorizes the Prokudin-Gorskii glass-plate negatives by splitting each into
# its B/G/R thirds and aligning G and R onto B. Runs three variants for the
# writeup:
#   1. single_scale  - exhaustive L2 search, one scale, on the small JPEGs
#   2. pyramid_l2    - multi-scale pyramid, L2 on raw RGB pixels (baseline)
#   3. golden        - pyramid_l2 + Sobel-gradient features
#
# Run from the proj1/ directory: `python code/main.py`

import glob
import json
import os
import time

import numpy as np
import skimage as sk
import skimage.filters as skf
import skimage.io as skio

from utils import crop_wraparound, euclidean_align, load_channels, pyramind_align

DATA_DIRS = ['data', 'data_custom']
RESULTS_PATH = 'results.json'


def image_paths():
    paths = []
    for d in DATA_DIRS:
        paths += glob.glob(f'{d}/*.jpg') + glob.glob(f'{d}/*.tif')
    return sorted(paths)


def save_aligned(b, g, r, shift_g, shift_r, imname, out_dir):
    ag = np.roll(g, shift_g, axis=(0, 1))
    ar = np.roll(r, shift_r, axis=(0, 1))
    im_out = np.dstack([ar, ag, b])
    # np.roll wraps garbage in on whichever edge we shifted from so we crop it back off
    im_out = crop_wraparound(im_out, shift_g, shift_r)

    os.makedirs(out_dir, exist_ok=True)
    basename, ext = os.path.splitext(os.path.basename(imname))
    fname = f'{out_dir}/{basename}_aligned{ext}'
    skio.imsave(fname, sk.img_as_ubyte(im_out))
    return fname


def run_single_scale(results):
    # exhaustive, single-scale, raw-pixel L2 search on the small JPEGs only
    print('=== single-scale (L2, raw pixels) ===')
    out_dir = 'output/single_scale'
    results['single_scale'] = {}
    for imname in sorted(glob.glob('data/*.jpg')):
        t0 = time.time()
        b, g, r = load_channels(imname)
        shift_g = euclidean_align(g, b, window=(-15, 15))
        shift_r = euclidean_align(r, b, window=(-15, 15))
        fname = save_aligned(b, g, r, shift_g, shift_r, imname, out_dir)
        dt = time.time() - t0
        print(f'{imname}: g={shift_g} r={shift_r} ({dt:.1f}s)')
        results['single_scale'][os.path.basename(imname)] = {
            'g_shift': shift_g, 'r_shift': shift_r, 'seconds': round(dt, 2),
        }


def run_pyramid_l2(results):
    # multi-scale pyramid, L2/SSD computed directly on raw RGB pixel values
    print('=== pyramid, L2 on raw pixels (required baseline) ===')
    out_dir = 'output/pyramid_l2'
    results['pyramid_l2'] = {}
    for imname in image_paths():
        t0 = time.time()
        b, g, r = load_channels(imname)
        shift_g = pyramind_align(g, b, min_size=100)
        shift_r = pyramind_align(r, b, min_size=100)
        fname = save_aligned(b, g, r, shift_g, shift_r, imname, out_dir)
        dt = time.time() - t0
        print(f'{imname}: g={shift_g} r={shift_r} ({dt:.1f}s)')
        results['pyramid_l2'][os.path.basename(imname)] = {
            'g_shift': shift_g, 'r_shift': shift_r, 'seconds': round(dt, 2),
        }


def run_golden(results):
    # pyramid + Sobel-gradient features: our best version
    print('=== golden standard (gradient-based pyramid) ===')
    out_dir = 'output/golden'
    results['golden'] = {}
    for imname in image_paths():
        t0 = time.time()
        b, g, r = load_channels(imname)
        b_edges, g_edges, r_edges = skf.sobel(b), skf.sobel(g), skf.sobel(r)
        shift_g = pyramind_align(g_edges, b_edges, min_size=100)
        shift_r = pyramind_align(r_edges, b_edges, min_size=100)
        fname = save_aligned(b, g, r, shift_g, shift_r, imname, out_dir)
        dt = time.time() - t0
        print(f'{imname}: g={shift_g} r={shift_r} ({dt:.1f}s)')
        results['golden'][os.path.basename(imname)] = {
            'g_shift': shift_g, 'r_shift': shift_r, 'seconds': round(dt, 2),
        }


if __name__ == '__main__':
    results = {}
    run_single_scale(results)
    run_pyramid_l2(results)
    run_golden(results)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'wrote {RESULTS_PATH}')
