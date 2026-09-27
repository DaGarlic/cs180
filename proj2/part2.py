from helper import unsharp, convol_symm_with_scipy, gaussian_kernel_cv
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

def psnr(image, reference):
    # Peak signal to noise ratio in dB, higher means closer to the reference
    mse = np.mean((image.astype(float) - reference.astype(float)) ** 2)
    return 10 * np.log10(255**2 / mse)

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

    # Blur a sharp image, sharpen it again, and see how close it gets back to the original
    img = cv2.imread(os.path.join(path_to_img_folder, "taj.jpg"), cv2.IMREAD_COLOR)
    blurred = clip_uint8(convol_symm_with_scipy(img, gaussian_kernel_cv(11, 1.5)))
    save_img(f"{path_to_img_folder}/taj_test_blurred.jpg", blurred)
    print(f"blurred (sigma 1.5): PSNR vs original {psnr(blurred, img):.2f} dB")
    for alpha in [0.5, 1, 2, 3, 4]:
        recovered = clip_uint8(unsharp(blurred, 11, 1.5, alpha))
        print(f"sharpened again, alpha {alpha:g}: PSNR vs original {psnr(recovered, img):.2f} dB")
        if alpha == 2:
            save_img(f"{path_to_img_folder}/taj_test_sharpened.jpg", recovered)
