from helper import convol_img, convol_img_2loops, convol_with_scipy, box_kernel, gaussian_kernel, Dx, Dy
import cv2
import numpy as np
import os

def save_img(path, image):
    with open(path, "wb") as f:
        f.write(cv2.imencode(".jpg", image)[1].tobytes())

def to_uint8(image, normalize=False):
    # Filter outputs are floats (and can be negative), so take the magnitude and clip before saving
    image = np.abs(image)
    if normalize:
        image = image / image.max() * 255
    return np.clip(np.round(image), 0, 255).astype(np.uint8)

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
            cimg = convol_img(img, kernel, mode)
            c2img = convol_img_2loops(img, kernel, mode)
            scimg = convol_with_scipy(img, kernel, mode)
            assert np.allclose(cimg, scimg), f"four loop {name} ({mode}) does not match scipy"
            assert np.allclose(c2img, scimg), f"two loop {name} ({mode}) does not match scipy"

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
    # Placeholder for part 1, question 2 implementation
    pass
