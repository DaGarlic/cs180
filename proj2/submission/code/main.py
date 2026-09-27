# CS180 Project 2: Fun with Filters and Frequencies
#
# Entry point. Run with `python main.py` from inside this code/ folder (it
# expects ./images to exist, and writes its outputs back into ./images).
#
#   Part 1.1  run_part1p1  convolution from scratch (four-loop and two-loop),
#                          zero padding, box filter / Dx / Dy on a photo
#   Part 1.2  run_part1p2  Dx/Dy, gradient magnitude, binarized edges on
#                          the cameraman image
#   Part 1.3  run_part1p3  derivative of Gaussian (DoG) filters
#   Part 2.1  run_part2p1  unsharp masking (taj.jpg + two own photos)
#   Part 2.2  run_part2p2  hybrid images (Derek/Nutmeg + three own pairs)
#   Part 2.3  run_part2p3  Gaussian/Laplacian stacks (Oraple + plush toys)
#   Part 2.4  run_part2p4  multiresolution blending with an irregular mask
#
# helper.py holds the shared numpy/scipy/cv2 routines; part1.py and part2.py
# each call into it to produce one part's figures.

from part1 import run_part1p1, run_part1p2, run_part1p3
from part2 import run_part2p1, run_part2p2, run_part2p3, run_part2p4

if __name__ == "__main__":
    path_to_imgs = "images"
    run_part1p1(path_to_imgs)
    run_part1p2(path_to_imgs)
    run_part1p3(path_to_imgs)
    run_part2p1(path_to_imgs)
    run_part2p2(path_to_imgs)
    run_part2p3(path_to_imgs)
    run_part2p4(path_to_imgs)
