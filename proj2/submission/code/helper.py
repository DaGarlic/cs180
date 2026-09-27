from math import pi, exp
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage.transform as sktr

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

# Image alignment for hybrid images (from the course starter code)

def get_points(im1, im2):
    # Click two corresponding points on each image (opens a window, so it needs a display)
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

def recenter(im, r, c):
    # Pad one side with zeros so that pixel (r, c) ends up at the center of the image
    R, C = im.shape[:2]
    rpad = int(np.abs(2*r+1 - R))
    cpad = int(np.abs(2*c+1 - C))
    pad_width = [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
                 (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad)]
    if im.ndim == 3:
        pad_width.append((0, 0))
    return np.pad(im, pad_width, 'constant')

def find_centers(p1, p2):
    # Midpoint of the two clicked points
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1, im2, pts):
    # Translate both images so the midpoint of their two points is at the center
    p1, p2, p3, p4 = pts

    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    # Shrink the image with the larger distance between its two points so the distances match
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    channel_axis = -1 if im1.ndim == 3 else None
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale, channel_axis=channel_axis)
    else:
        im2 = sktr.rescale(im2, 1./dscale, channel_axis=channel_axis)
    return im1, im2

def rotate_im1(im1, pts):
    # Rotate the first image so the line between its two points has the same angle as in the second image
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta*180/np.pi)
    return im1, dtheta

def match_img_size(im1, im2):
    # Crop the larger image around its center so both have the same size
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    if h1 < h2:
        im2 = im2[int(np.floor((h2-h1)/2.)) : -int(np.ceil((h2-h1)/2.)), :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1-h2)/2.)) : -int(np.ceil((h1-h2)/2.)), :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2-w1)/2.)) : -int(np.ceil((w2-w1)/2.))]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1-w2)/2.)) : -int(np.ceil((w1-w2)/2.))]
    assert im1.shape == im2.shape
    return im1, im2

def align_images(im1, im2, pts=None):
    # Align im1 to im2 by translation, scale and rotation. pts are the 4 clicked points (p1, p2 on im1 and
    # p3, p4 on im2), if they aren't given you get to click them
    if pts is None:
        pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2

def largest_rectangle(mask):
    # Find the biggest rectangle that is all True in a boolean mask, as (top, bottom, left, right) with bottom
    # and right exclusive. Go row by row, keeping how many True cells are stacked above each column
    heights = np.zeros(mask.shape[1], dtype=int)
    best_area, best = 0, (0, 0, 0, 0)
    for r in range(mask.shape[0]):
        heights = np.where(mask[r], heights + 1, 0)
        stack = []
        for c in range(len(heights) + 1):
            h = heights[c] if c < len(heights) else 0
            start = c
            while stack and stack[-1][1] >= h:
                start, stack_h = stack.pop()
                if stack_h * (c - start) > best_area:
                    best_area = stack_h * (c - start)
                    best = (r - stack_h + 1, r + 1, start, c)
            stack.append((start, h))
    return best

def crop_to_valid(im1, im2, shape1, shape2, pts):
    # Aligning pads and rotates in black borders. Push all ones images through the same alignment to see
    # which pixels are real picture, then crop both images to the biggest rectangle that is real in both
    ones1, ones2 = align_images(np.ones(shape1[:2]), np.ones(shape2[:2]), pts)
    top, bottom, left, right = largest_rectangle((ones1 > 0.999) & (ones2 > 0.999))
    return im1[top:bottom, left:right], im2[top:bottom, left:right]

def gaussian_blur(image, sigma):
    # Blur with a separable Gaussian (two 1D passes), much faster than one 2D kernel when sigma is large
    size = 2 * int(np.ceil(3 * sigma)) + 1
    kernel_1d = cv2.getGaussianKernel(size, sigma)
    return convol_symm_with_scipy(convol_symm_with_scipy(image, kernel_1d), kernel_1d.T)

def hybrid_image(im1, im2, sigma1, sigma2, gain=1.0):
    # High frequencies of im1 (the image minus its blur, times a gain that sets how strong they are next to the
    # low frequencies) plus the low frequencies of im2 (its blur)
    high = gain * (im1 - gaussian_blur(im1, sigma1))
    low = gaussian_blur(im2, sigma2)
    return high + low

def gaussian_stack(image, levels, sigma=4):
    # Gaussian stack: level 0 is the image, and every level after it blurs the image with a Gaussian twice as wide as
    # the level before. Nothing is subsampled, so all levels keep the size of the image
    stack = [image]
    for i in range(1, levels):
        stack.append(gaussian_blur(image, sigma * 2 ** (i - 1)))
    return np.array(stack)

def laplacian_stack(gaussian):
    # Laplacian stack: the difference between neighboring Gaussian levels, with the blurriest Gaussian level kept at the
    # end, so the levels add back up to the image
    stack = []
    for i in range(len(gaussian) - 1):
        stack.append(gaussian[i] - gaussian[i + 1])
    stack.append(gaussian[-1])
    return np.array(stack)

def blend_stacks(laplacian_a, laplacian_b, mask_gaussian):
    # Blend two Laplacian stacks level by level, weighting level i of the first by level i of the mask's Gaussian stack
    # and level i of the second by what is left. Returns every blended level, and their sum is the blended image
    mask = mask_gaussian[..., np.newaxis]
    return mask * laplacian_a + (1 - mask) * laplacian_b

def match_at_seam(image, reference, white=0.94):
    # Scale an image about its center column and shift it up or down so the figure in it covers the same rows at the seam
    # (the center column) as the figure in reference does. Both are on a white background, and new space is filled with white
    def extent(im):
        rows = np.where((im[:, im.shape[1] // 2] < white).any(axis=1))[0]
        return rows.min(), rows.max()
    top_ref, bottom_ref = extent(reference)
    top, bottom = extent(image)
    scale = (bottom_ref - top_ref) / (bottom - top)
    shift = top_ref - scale * top
    matrix = np.array([[scale, 0, image.shape[1] / 2 * (1 - scale)], [0, scale, shift]])
    warped = cv2.warpAffine(image.astype(np.float32), matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1))
    return np.clip(warped, 0, 1).astype(float), scale, shift

def segment_grabcut(image, rect, iterations=10):
    # Cut a rough foreground object out of a BGR uint8 image with GrabCut, seeded with a rectangle around it. Cleaned
    # up with morphological open/close and filled to its external contour, so interior dark details (buttons, a belt)
    # don't punch holes in the mask. Returns a float64 mask of 1s and 0s, the same size as the image
    gc_mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model, fgd_model = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(image, gc_mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    binary = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    solid = np.zeros_like(binary)
    cv2.drawContours(solid, [largest], -1, 255, thickness=cv2.FILLED)
    return (solid > 0).astype(np.float64)

def composite_masked(base, overlay, overlay_mask, top, left):
    # Paste overlay (and its own mask) onto a copy of base at (top, left). Returns the composited image and a mask
    # the same size as base, for blending against the unmodified base with an irregular (non-seam) mask
    canvas = base.copy()
    mask_full = np.zeros(base.shape[:2])
    h, w = overlay.shape[:2]
    region = canvas[top:top + h, left:left + w]
    weight = overlay_mask[..., np.newaxis]
    region[:] = overlay * weight + region * (1 - weight)
    mask_full[top:top + h, left:left + w] = overlay_mask
    return canvas, mask_full
