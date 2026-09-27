CS180 Project 2: Fun with Filters and Frequencies
Ariq Koh

Website: https://dagarlic.github.io/cs180/proj2/

FILES
  main.py   - entry point. Run with `python main.py` from inside this
              code/ folder. Reads from ./images and writes every figure
              back into ./images (about 100 files, a few minutes total,
              mostly Part 1.1's four-loop convolution on a 1280x960 photo).
  helper.py - shared routines: convolution (four-loop and two-loop, with
              zero padding), Gaussian kernels and blur, unsharp masking,
              image alignment (course starter code, adapted to take fixed
              points instead of clicking), Gaussian/Laplacian stacks,
              multiresolution blending, and GrabCut-based segmentation for
              the irregular-mask example.
  part1.py  - Part 1: convolution from scratch, finite differences,
              derivative of Gaussian.
  part2.py  - Part 2: unsharp masking, hybrid images, Gaussian/Laplacian
              stacks, multiresolution blending.
  images/   - only the source images each function reads by name (not the
              generated outputs, which main.py recreates). Some inputs
              (blurred_duck.JPG, blurred_fishes.JPG) are the largest files
              here; everything else is a few hundred KB or less.

WHAT IT DOES
  Part 1.1 (run_part1p1) - convolution written with four nested loops and
    then with two, both with hand-written zero padding ("same" and "full")
    and a flipped kernel, checked against scipy.signal.convolve2d. Run on a
    photo with a 9x9 box filter and the Dx/Dy finite difference operators.

  Part 1.2 (run_part1p2) - Dx and Dy on the cameraman image, combined into
    a gradient magnitude, then thresholded into a binary edge map.

  Part 1.3 (run_part1p3) - a Gaussian built from cv2.getGaussianKernel via
    an outer product, blurred edges compared against unblurred, and DoG
    filters (Gaussian convolved with Dx/Dy) checked against blurring first
    and taking the difference after.

  Part 2.1 (run_part2p1) - unsharp masking collapsed into one convolution,
    (1+alpha)*impulse - alpha*Gaussian, run on taj.jpg (varying alpha) plus
    two personal photos, and a blur-then-sharpen-again comparison.

  Part 2.2 (run_part2p2) - hybrid images: an image's high frequencies
    (itself minus its own blur, scaled by a gain) added to another image's
    low frequencies (its blur). Both images of every pair are converted to
    grayscale first. Four pairs: a dessert (the favorite, with the full
    process and Fourier spectra shown), Derek and Nutmeg, two Labubu dolls,
    and two photos of a gull in different poses, aligned on hand-picked
    points instead of clicking them.

  Part 2.3 (run_part2p3) - Gaussian and Laplacian stacks built from scratch
    (no cv2.pyrDown or skimage.transform.pyramid_gaussian), applied to the
    course's apple/orange pair and to two plush toys, with a straight
    vertical-seam blend recreating Szeliski's Figure 3.42.

  Part 2.4 (run_part2p4) - multiresolution blending with an irregular mask:
    a cartoon character is cut out of one photo with cv2.grabCut and pasted
    into an unrelated scene, then blended in with the same Gaussian/Laplacian
    stacks as Part 2.3, so the mask (not a straight seam) feathers the edge
    instead of pasting it on as a sticker.
