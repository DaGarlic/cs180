CS180 (CS280A) Project 1: Images of the Russian Empire
Ariq Koh

FILES
  main.py   - entry point. Run with `python main.py` from inside this
              code/ folder (it expects ../data, ../data_custom to exist,
              and writes ../output/* and ../results.json).
  utils.py  - alignment routines shared by all three variants below.

WHAT IT DOES
  For each glass-plate scan, splits it into B/G/R thirds and aligns G and R
  onto B by translation only. Both euclidean_align and pyramind_align score
  a candidate shift by L2/SSD on a center square crop only, applied via
  direct index shifting rather than np.roll, so image borders and
  roll-wraparound can never bias the match. Runs three variants, in order,
  so the writeup can compare them:

  1. single_scale - one exhaustive L2/SSD search over a +-15px window on
     the raw pixels. Only run on the small JPEGs.

  2. pyramid_l2 - the required baseline: a coarse-to-fine image pyramid
     (min_size=100), still scoring alignment with raw-pixel L2/SSD. Run on
     every image in data/ and data_custom/.

  3. golden - our best version. Same pyramid, but alignment is scored on
     each channel's Sobel gradient instead of raw pixel values. Raw-pixel
     SSD gets fooled on one of the harder plates (a repeating tiled
     pattern scores deceptively well at the wrong offset); gradients fix
     it without touching anything else. See the writeup for the before
     and after.

  Every run writes its aligned outputs to ../output/<variant>/ and its
  computed (x, y) shifts (plus per-image timing) to ../results.json,
  which the project webpage reads to build its result tables.

DATA
  ../data/        - the 14 example glass plates provided with the assignment.
  ../data_custom/ - 3 additional Prokudin-Gorskii plates downloaded from the
                    Library of Congress collection (loc.gov/collections/
                    prokudin-gorskii) for the deliverable.