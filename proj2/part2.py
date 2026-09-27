from helper import unsharp, convol_symm_with_scipy, gaussian_kernel_cv, align_images, crop_to_valid, gaussian_blur, hybrid_image, gaussian_stack, laplacian_stack, blend_stacks
from part1 import save_img
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

def to_gray(image):
    # Grayscale version of an RGB image, still with 3 channels (all equal) so the other functions can take it
    gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
    return np.dstack([gray, gray, gray])

def log_spectrum(image):
    # Log magnitude of the 2D Fourier transform of the grayscale image, low frequencies in the center
    gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))))

def run_part2p2(path_to_img_folder):
    # (name, image for the high frequencies, image for the low frequencies, alignment points, sigma of the high pass,
    # sigma of the low pass, gain on the high frequencies). The cutoffs and gain are tuned for each image by eye on the
    # grayscale versions, at the sizes the page shows: the first image should win up close and the second from far away
    # The alignment points are (x, y) picked by hand instead of clicking: two on the first image, then the matching two on the second
    hybrids = [
        ("derek_nutmeg", "DerekPicture.jpg", "nutmeg.jpg", ((296, 344), (444, 332), (600, 289), (749, 367)), 4, 12, 1),
        ("labubu", "labubu_pair_1.jpg", "labubu_pair_2.jpg", ((418, 664), (559, 675), (314, 691), (420, 692)), 5, 18, 2),
        ("bird", "bird_pair_1.jpg", "bird_pair_2.jpg", ((80, 594), (500, 634), (80, 569), (600, 688)), 5, 12, 0.65),
        ("dessert", "dessert_pair_2.jpg", "dessert_pair_1.jpg", ((350, 236), (350, 636), (357, 214), (436, 671)), 7, 16, 1),
    ]
    for name, high_name, low_name, pts, sigma_high, sigma_low, gain in hybrids:
        # Convert both images to grayscale first, so the hybrid only depends on brightness
        im1 = to_gray(load_rgb(os.path.join(path_to_img_folder, high_name)))
        im2 = to_gray(load_rgb(os.path.join(path_to_img_folder, low_name)))

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

def make_montage(tiles, row_labels, tile_labels=None, headers=None, tile_size=300, gap=10, font_size=30):
    # Lay RGB float tiles (values 0 to 1) out on a paper colored canvas, with a label for each row on the left, an optional
    # label under every tile and an optional header over each column. The text is large because the montage is shown small
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    left = int(font_size * 7)
    label_h = int(font_size * 1.6) if tile_labels else 0
    top = int(font_size * 1.6) if headers else 0
    rows, cols = len(tiles), len(tiles[0])
    canvas = Image.new("RGB", (left + cols * tile_size + (cols + 1) * gap, top + rows * (tile_size + label_h + gap) + gap), (250, 247, 241))
    draw = ImageDraw.Draw(canvas)
    for c in range(cols):
        if headers:
            draw.text((left + gap + c * (tile_size + gap), gap), headers[c], fill=(60, 55, 45), font=font)
    for r in range(rows):
        y = top + gap + r * (tile_size + label_h + gap)
        draw.text((gap, y + tile_size // 2 - font_size // 2), row_labels[r], fill=(138, 43, 43), font=font)
        for c in range(cols):
            tile = cv2.resize(clip_uint8(tiles[r][c] * 255), (tile_size, tile_size), interpolation=cv2.INTER_AREA)
            x = left + gap + c * (tile_size + gap)
            canvas.paste(Image.fromarray(tile), (x, y))
            if tile_labels:
                draw.text((x, y + tile_size + font_size // 3), tile_labels[r][c], fill=(60, 55, 45), font=font)
    return np.array(canvas)[:, :, ::-1]

def run_part2p3(path_to_img_folder):
    # (name, first image, second image, names to show, sigma of the first blurred level). The left half of the blend
    # is the first image. The blur doubles at every level, so the smaller Oraple images start at a smaller sigma
    pairs = [
        ("oraple", "apple.jpeg", "orange.jpeg", ("apple", "orange"), 2),
        ("plush", "pikachu.jpg", "charizard.jpg", ("pikachu", "charizard"), 4),
    ]
    levels = 5
    band_gain = 3
    for pair, name1, name2, names, sigma in pairs:
        im1 = load_rgb(os.path.join(path_to_img_folder, name1))
        im2 = load_rgb(os.path.join(path_to_img_folder, name2))
        blur_labels = ["original"] + [f"\u03c3 = {sigma * 2 ** (i - 1)}" for i in range(1, levels)]

        # Gaussian and Laplacian stacks of both images, shown as a montage per image. The band-pass levels of the
        # Laplacian stack are small, so they are scaled up and shifted to mid gray to be visible
        laplacians = []
        for name, image in zip(names, [im1, im2]):
            gaussian = gaussian_stack(image, levels, sigma)
            laplacian = laplacian_stack(gaussian)
            laplacians.append(laplacian)
            print(f"{pair}, {name}: Laplacian levels add back to the image, max difference {np.abs(laplacian.sum(axis=0) - image).max():.1e}")
            shown = [0.5 + band_gain * laplacian[i] for i in range(levels - 1)] + [laplacian[-1]]
            labels = [[f"Level {i}: {blur_labels[i]}" for i in range(levels)],
                      [f"Level {i}: " + ("band" if i < levels - 1 else "low-pass") for i in range(levels)]]
            save_img(f"{path_to_img_folder}/stack_{pair}_{name}.jpg", make_montage([list(gaussian), shown], ["Gaussian", "Laplacian"], tile_labels=labels))

        # Blend with a vertical seam: the mask is 1 on the left half, and its own Gaussian stack sets how wide the seam is at each level
        mask = np.zeros(im1.shape[:2])
        mask[:, :im1.shape[1] // 2] = 1
        mask_stack = gaussian_stack(mask, levels, sigma)
        part1 = mask_stack[..., np.newaxis] * laplacians[0]
        part2 = (1 - mask_stack[..., np.newaxis]) * laplacians[1]
        blended = blend_stacks(laplacians[0], laplacians[1], mask_stack)

        # Szeliski's Figure 3.42: the high, medium and low frequency levels of each image weighted by the mask, their sum,
        # and the totals over all levels
        def show(level, image):
            return image if level == levels - 1 else 0.5 + band_gain * image
        rows = []
        for level in [0, 2, 4]:
            rows.append([show(level, part1[level]), show(level, part2[level]), show(level, blended[level])])
        final = np.clip(blended.sum(axis=0), 0, 1)
        rows.append([part1.sum(axis=0), part2.sum(axis=0), final])
        headers = [f"{names[0].capitalize()} x mask", f"{names[1].capitalize()} x (1 - mask)", "Sum"]
        row_labels = ["Level 0", "Level 2", "Level 4", "All levels"]
        save_img(f"{path_to_img_folder}/stack_{pair}_figure.jpg", make_montage(rows, row_labels, headers=headers, font_size=26))
        save_img(f"{path_to_img_folder}/stack_{pair}_blend.jpg", rgb_to_bgr_uint8(final))
