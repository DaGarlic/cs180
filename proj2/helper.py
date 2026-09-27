from math import pi, exp
import cv2
import numpy as np
import scipy.signal

Dx = [[1, -1]]
Dy = [[1], [-1]]

def gaussian_kernel(size, sigma):
    # Generate a 2D Gaussian kernel
    kernel = []
    for i in range(size):
        row = []
        for j in range(size):
            x = j - size // 2
            y = i - size // 2
            row.append(exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * pi * sigma**2))
        kernel.append(row)
    return kernel

def box_kernel(size):
    # Generate a 2D box filter kernel (every entry is 1 / size^2 so it sums to 1)
    kernel = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(1 / size**2)
        kernel.append(row)
    return kernel

def pad_img(image, kernel, mode="same"):
    # Zero pad the image ("same" keeps the output size, "full" gives every position the kernel overlaps)
    height, width = image.shape[:2]
    kernel_h, kernel_w = len(kernel), len(kernel[0])
    if mode == "full":
        top, left = kernel_h - 1, kernel_w - 1
        bottom, right = kernel_h - 1, kernel_w - 1
    elif mode == "same":
        bottom, right = (kernel_h - 1) // 2, (kernel_w - 1) // 2
        top, left = kernel_h - 1 - bottom, kernel_w - 1 - right
    else:
        raise ValueError(f"Unknown padding mode: {mode}")

    padded_image = np.zeros((height + top + bottom, width + left + right))
    padded_image[top:top + height, left:left + width] = image
    return padded_image

def convol_img(image, kernel, mode="same"):
    # Perform convolution of the image with the given kernel (four for loop implementation)
    kernel_h, kernel_w = len(kernel), len(kernel[0])
    padded_image = pad_img(image, kernel, mode)
    flipped_kernel = np.array(kernel)[::-1, ::-1]
    height = padded_image.shape[0] - kernel_h + 1
    width = padded_image.shape[1] - kernel_w + 1
    output_image = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            sum_val = 0.0
            for ki in range(kernel_h):
                for kj in range(kernel_w):
                    sum_val += padded_image[i + ki][j + kj] * flipped_kernel[ki][kj]
            output_image[i][j] = sum_val

    return output_image

def convol_img_2loops(image, kernel, mode="same"):
    # Perform convolution of the image with the given kernel (two for loop implementation)
    kernel_h, kernel_w = len(kernel), len(kernel[0])
    padded_image = pad_img(image, kernel, mode)
    flipped_kernel = np.array(kernel)[::-1, ::-1]
    height = padded_image.shape[0] - kernel_h + 1
    width = padded_image.shape[1] - kernel_w + 1
    output_image = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            output_image[i][j] = np.sum(padded_image[i:i + kernel_h, j:j + kernel_w] * flipped_kernel)

    return output_image

def convol_with_scipy(image, kernel, mode="same"):
    return scipy.signal.convolve2d(image, kernel, mode=mode, boundary="fill", fillvalue=0)

def gaussian_kernel_cv(size, sigma):
    # Generate a 2D Gaussian kernel as the outer product of cv2's 1D Gaussian with itself
    kernel_1d = cv2.getGaussianKernel(size, sigma)
    return kernel_1d @ kernel_1d.T

def convol_symm_with_scipy(image, kernel):
    # Same as convol_with_scipy, but mirrors the image at the border instead of zero padding it,
    # so blurring doesn't darken the edges and no fake frame shows up in the gradients
    if image.ndim == 3:
        # Color image, convolve2d only takes 2D arrays so convolve each channel on its own
        channels = []
        for c in range(image.shape[2]):
            channels.append(convol_symm_with_scipy(image[:, :, c], kernel))
        return np.dstack(channels)
    return scipy.signal.convolve2d(image, kernel, mode="same", boundary="symm")

def unsharp_kernel(size, sigma, alpha):
    # Unsharp mask as a single kernel. image + alpha * (image - image * G) is linear in the image, so it
    # equals image * ((1 + alpha) * impulse - alpha * G), where the impulse is a 1 at the center of the kernel
    kernel = -alpha * gaussian_kernel_cv(size, sigma)
    kernel[size // 2, size // 2] += 1 + alpha
    return kernel

def unsharp(image, kernel, sigma, alpha=1.0):
    # Apply unsharp masking with a single convolution, alpha is how much of the high frequencies to add back
    return convol_symm_with_scipy(image, unsharp_kernel(kernel, sigma, alpha))
