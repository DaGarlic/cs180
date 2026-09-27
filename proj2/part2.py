from helper import unsharp, convol_symm_with_scipy, gaussian_kernel_cv, align_images, crop_to_valid, gaussian_blur, hybrid_image
from part1 import save_img
import os
import cv2
import numpy as np

def shrink(image, longest_side):
    # Downscale so the longer side is at most longest_side pixels (leaves smaller images alone)
    height, width = image.shape[:2]
    scale = longest_side / max(height, width)
    if scale >= 1:
        return image
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)

def clip_uint8(image):
    # Sharpening pushes values outside 0 to 255, clip them (not abs, a negative value should become black)
    return np.clip(np.round(image), 0, 255).astype(np.uint8)

def run_part2p1(path_to_img_folder):
    # (image, gaussian size, sigma): the two phone photos are out of focus and much larger than taj, so use a wider blur
    images = [("taj.jpg", 5, 1.0), ("blurred_duck.JPG", 9, 2.0), ("blurred_fishes.JPG", 9, 2.0)]
    alphas = [0.5, 1, 2, 4]
    for img_name, kernel, sigma in images:
        img_path = os.path.join(path_to_img_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Keep test images small (under 0.8 MB), save the shrunk input so it can be compared with the result
        small_img = shrink(img, 1000)
        name = os.path.splitext(img_name)[0]
        if small_img is not img:
            save_img(f"{path_to_img_folder}/{name}_small.jpg", small_img)

        # The blurred version, and the high frequencies the blur removed (shown around mid gray, scaled up 4x)
        blurred = convol_symm_with_scipy(small_img, gaussian_kernel_cv(kernel, sigma))
        save_img(f"{path_to_img_folder}/{name}_blurred.jpg", clip_uint8(blurred))
        save_img(f"{path_to_img_folder}/{name}_highfreq.jpg", clip_uint8(4 * (small_img - blurred) + 128))

        # Sharpen with a few different amounts
        for alpha in alphas:
            sharpened = unsharp(small_img, kernel, sigma, alpha)
            save_img(f"{path_to_img_folder}/{name}_sharpened_alpha{alpha:g}.jpg", clip_uint8(sharpened))

    # Blur a sharp image, then sharpen it again and compare the result with the original by eye
    img = cv2.imread(os.path.join(path_to_img_folder, "taj.jpg"), cv2.IMREAD_COLOR)
    blurred = clip_uint8(convol_symm_with_scipy(img, gaussian_kernel_cv(11, 1.5)))
    save_img(f"{path_to_img_folder}/taj_test_blurred.jpg", blurred)
    save_img(f"{path_to_img_folder}/taj_test_sharpened.jpg", clip_uint8(unsharp(blurred, 11, 1.5, 2)))

def load_rgb(path):
    # cv2 reads BGR and applies the photo's rotation, convert to RGB floats in 0 to 1 like the starter code
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.

def rgb_to_bgr_uint8(image):
    # RGB floats in 0 to 1 back to something cv2 can save
    return cv2.cvtColor(clip_uint8(image * 255), cv2.COLOR_RGB2BGR)

def log_spectrum(image):
    # Log magnitude of the 2D Fourier transform of the grayscale image, low frequencies in the center
    gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))))

def run_part2p2(path_to_img_folder):
    # (name, image for the high frequencies, image for the low frequencies, alignment points, sigma of the high pass,
    # sigma of the low pass, gain on the high frequencies). The cutoffs are tuned for each image by eye at the sizes the
    # page shows: the first image should win up close and the second from far away
    # The alignment points are (x, y) picked by hand instead of clicking: two on the first image, then the matching two on the second
    hybrids = [
        ("derek_nutmeg", "DerekPicture.jpg", "nutmeg.jpg", ((296, 344), (444, 332), (600, 289), (749, 367)), 2.5, 12, 2),
        ("labubu", "labubu_pair_1.jpg", "labubu_pair_2.jpg", ((418, 664), (559, 675), (314, 691), (420, 692)), 6, 12, 1.5),
        ("bird", "bird_pair_1.jpg", "bird_pair_2.jpg", ((278, 633), (217, 694), (231, 619), (198, 652)), 3, 8, 1.5),
        ("dessert", "dessert_pair_2.jpg", "dessert_pair_1.jpg", ((350, 236), (350, 636), (357, 214), (436, 671)), 7, 15, 1.5),
    ]
    for name, high_name, low_name, pts, sigma_high, sigma_low, gain in hybrids:
        im1 = load_rgb(os.path.join(path_to_img_folder, high_name))
        im2 = load_rgb(os.path.join(path_to_img_folder, low_name))

        # Align the two images, then crop off the black borders that the alignment leaves
        aligned1, aligned2 = align_images(im1, im2, pts)
        aligned1, aligned2 = crop_to_valid(aligned1, aligned2, im1.shape, im2.shape, pts)

        # High frequencies of the first image (shown around mid gray), low frequencies of the second, and their sum
        high = gain * (aligned1 - gaussian_blur(aligned1, sigma_high))
        low = gaussian_blur(aligned2, sigma_low)
        hybrid = hybrid_image(aligned1, aligned2, sigma_high, sigma_low, gain)

        images = {"aligned1": aligned1, "aligned2": aligned2, "high": high + 0.5, "low": low, "hybrid": hybrid}
        for part, image in images.items():
            save_img(f"{path_to_img_folder}/{name}_{part}.jpg", rgb_to_bgr_uint8(image))

        # Fourier spectra of the inputs, the filtered images and the hybrid, all on the same brightness scale
        spectra = {"aligned1": log_spectrum(aligned1), "aligned2": log_spectrum(aligned2), "high": log_spectrum(high),
                   "low": log_spectrum(low), "hybrid": log_spectrum(hybrid)}
        lowest = min(spectrum.min() for spectrum in spectra.values())
        highest = max(spectrum.max() for spectrum in spectra.values())
        for part, spectrum in spectra.items():
            save_img(f"{path_to_img_folder}/{name}_fft_{part}.jpg", clip_uint8((spectrum - lowest) / (highest - lowest) * 255))
        print(f"{name}: {aligned1.shape[1]}x{aligned1.shape[0]}, sigma {sigma_high} (high pass) and {sigma_low} (low pass), gain {gain}")
