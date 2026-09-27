from helper import convol_img, convol_img_2loops, convol_with_scipy, box_kernel, gaussian_kernel, gaussian_kernel_cv, convol_symm_with_scipy, Dx, Dy
import cv2
import numpy as np
import os
import time

def timed(func, *args):
    start = time.perf_counter()
    result = func(*args)
    return result, time.perf_counter() - start

def save_img(path, image):
    # Encode by file extension (png for binary edge maps, so thin lines don't get jpg artifacts)
    with open(path, "wb") as f:
        f.write(cv2.imencode(os.path.splitext(path)[1], image)[1].tobytes())

def to_uint8(image, normalize=False):
    # Filter outputs are floats (and can be negative), so take the magnitude and clip before saving
    image = np.abs(image)
    if normalize:
        image = image / image.max() * 255
    return np.clip(np.round(image), 0, 255).astype(np.uint8)

def crop_white_frame(image):
    # Crop off the white frame around the image, it would show up as a strong edge
    rows = np.where((image != 255).any(axis=1))[0]
    cols = np.where((image != 255).any(axis=0))[0]
    return image[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

def filter_to_img(kernel, scale=20):
    # Show a filter with values around 0 as mid gray (positive is lighter, negative is darker), enlarged
    kernel = np.array(kernel)
    shown = np.round(kernel / np.abs(kernel).max() * 127 + 128).astype(np.uint8)
    return cv2.resize(shown, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

def run_part1p1(path_to_img_folder):
    img_path = os.path.join(path_to_img_folder, "paddling.JPG")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    save_img(f"{path_to_img_folder}/paddling_bw.jpg", img)

    # Convolve with a 9x9 box filter and the finite difference operators Dx and Dy
    kernels = {"box_filter": box_kernel(9), "dx": Dx, "dy": Dy}
    for name, kernel in kernels.items():
        for mode in ["same", "full"]:
            cimg, t_four = timed(convol_img, img, kernel, mode)
            c2img, t_two = timed(convol_img_2loops, img, kernel, mode)
            scimg, t_scipy = timed(convol_with_scipy, img, kernel, mode)
            max_diff = max(np.abs(cimg - scimg).max(), np.abs(c2img - scimg).max())
            print(f"{name:10s} {mode:5s} out {cimg.shape}  four loop {t_four:7.2f}s  two loop {t_two:6.2f}s  scipy {t_scipy:6.3f}s  max |diff| {max_diff:.1e}")

            if mode == "same":
                normalize = name != "box_filter"
                save_img(f"{path_to_img_folder}/{name}.jpg", to_uint8(cimg, normalize))
                save_img(f"{path_to_img_folder}/{name}_scipy.jpg", to_uint8(scimg, normalize))

    # Apply Gaussian blur to the image
    cimg = convol_img(img, gaussian_kernel(9, 3))
    scimg = convol_with_scipy(img, gaussian_kernel(9, 3))

    save_img(f"{path_to_img_folder}/blurred_image.jpg", to_uint8(cimg))
    save_img(f"{path_to_img_folder}/blurred_image_scipy.jpg", to_uint8(scimg))

def run_part1p2(path_to_img_folder):
    img_path = os.path.join(path_to_img_folder, "cameraman.png")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = crop_white_frame(img)
    save_img(f"{path_to_img_folder}/cameraman_gray.jpg", img)

    # Partial derivatives in x and y
    dx = convol_img(img, Dx)
    dy = convol_img(img, Dy)

    # Zero padding turns the first column of Dx and the first row of Dy into raw pixel values, not edges
    dx[:, 0] = 0
    dy[0, :] = 0
    save_img(f"{path_to_img_folder}/cameraman_dx.jpg", to_uint8(dx, normalize=True))
    save_img(f"{path_to_img_folder}/cameraman_dy.jpg", to_uint8(dy, normalize=True))

    # Gradient magnitude, then threshold it to get a binary edge image
    magnitude = np.sqrt(dx**2 + dy**2)
    save_img(f"{path_to_img_folder}/cameraman_magnitude.jpg", to_uint8(magnitude, normalize=True))

    for threshold in [15, 30, 45, 80]:
        edges = (magnitude > threshold) * 255
        save_img(f"{path_to_img_folder}/cameraman_edges_{threshold}.png", edges.astype(np.uint8))

def run_part1p3(path_to_img_folder):
    img_path = os.path.join(path_to_img_folder, "cameraman.png")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = crop_white_frame(img).astype(float)
    threshold = 15

    # Blur with a 13x13 Gaussian (sigma 2), then repeat the finite difference procedure from part 1.2
    gaussian = gaussian_kernel_cv(13, 2)
    blurred = convol_symm_with_scipy(img, gaussian)
    dx = convol_symm_with_scipy(blurred, Dx)
    dy = convol_symm_with_scipy(blurred, Dy)
    magnitude = np.sqrt(dx**2 + dy**2)
    edges = (magnitude > threshold) * 255
    save_img(f"{path_to_img_folder}/cameraman_blurred.jpg", to_uint8(blurred))
    save_img(f"{path_to_img_folder}/cameraman_blur_magnitude.jpg", to_uint8(magnitude, normalize=True))
    save_img(f"{path_to_img_folder}/cameraman_blur_edges.png", edges.astype(np.uint8))

    # Derivative of Gaussian filters: convolve the Gaussian with Dx and Dy, then apply them to the original image
    dog_x = convol_img(gaussian, Dx, "full")
    dog_y = convol_img(gaussian, Dy, "full")
    dog_dx = convol_symm_with_scipy(img, dog_x)
    dog_dy = convol_symm_with_scipy(img, dog_y)
    dog_magnitude = np.sqrt(dog_dx**2 + dog_dy**2)
    dog_edges = (dog_magnitude > threshold) * 255
    save_img(f"{path_to_img_folder}/cameraman_dog_x.png", filter_to_img(dog_x))
    save_img(f"{path_to_img_folder}/cameraman_dog_y.png", filter_to_img(dog_y))
    save_img(f"{path_to_img_folder}/cameraman_dog_magnitude.jpg", to_uint8(dog_magnitude, normalize=True))
    save_img(f"{path_to_img_folder}/cameraman_dog_edges.png", dog_edges.astype(np.uint8))

    # The DoG result should match blurring first and then taking the finite differences
    print(f"DoG filter sizes {dog_x.shape} and {dog_y.shape}")
    print(f"max |diff| vs two step: Ix {np.abs(dx - dog_dx).max():.1e}  Iy {np.abs(dy - dog_dy).max():.1e}  magnitude {np.abs(magnitude - dog_magnitude).max():.1e}")
    print(f"edge pixels that differ at threshold {threshold}: {np.sum(edges != dog_edges)} of {edges.size}")

    