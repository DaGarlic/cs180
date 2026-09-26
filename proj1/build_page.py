# Generates ../index.html from ../results.json + image dimensions in
# ../media/. Not part of the graded pipeline -- just a helper so the offset
# tables/galleries in the writeup can't drift out of sync with results.json.
# Run from the code/ folder: `python build_page.py`

import json
import os
from PIL import Image

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CUSTOM_TITLES = {
    'dagestan_group': 'Tipy Dagestana',
    'portrait': 'Ėtiud golovki',
    'ukraine_fence': 'V Maloross i',
}
PROVIDED_TITLES = {
    'cathedral': 'Cathedral',
    'church': 'Church',
    'emir': 'Emir of Bukhara',
    'harvesters': 'Harvesters',
    'icon': 'Icon',
    'ilemselga': 'Ilemselga',
    'melons': 'Melons',
    'monastery': 'Monastery',
    'religous_painting': 'Religious Painting',
    'self_portrait': 'Self Portrait',
    'siren': 'Siren (Lilacs)',
    'three_generations': 'Three Generations',
    'tobolsk': 'Tobolsk',
    'wharf': 'Wharf',
}
JPG_KEYS = {'cathedral', 'monastery', 'tobolsk'}


def dims(rel_path):
    with Image.open(os.path.join(PROJ, rel_path)) as im:
        return im.size


def fname_for(key):
    return key + '.jpg' if key in JPG_KEYS else key + '.tif'


def fmt_shift(s):
    return f'({s[0]}, {s[1]})'


def offsets_caption(rec):
    return f'<span class="offsets">G {fmt_shift(rec["g_shift"])} &middot; R {fmt_shift(rec["r_shift"])}</span>'


def figure(variant, key, title, caption_extra='', alt_title=None):
    src = f'media/{variant}/{key}_aligned.jpg'
    w, h = dims(src)
    alt_title = alt_title or title
    return (f'    <figure>\n'
             f'      <img src="{src}" width="{w}" height="{h}" loading="lazy" '
             f'alt="Aligned Prokudin-Gorskii plate: {alt_title}">\n'
             f'      <figcaption>{title}{caption_extra}</figcaption>\n'
             f'    </figure>\n')


def compare_figure(variant, key, label):
    src = f'media/{variant}/{key}_aligned.jpg'
    w, h = dims(src)
    return (f'      <img src="{src}" width="{w}" height="{h}" loading="lazy" '
             f'alt="{label} alignment of {key}">')


def build():
    results = json.load(open(os.path.join(PROJ, 'results.json')))
    golden, l2, single = results['golden'], results['pyramid_l2'], results['single_scale']

    provided_keys = list(PROVIDED_TITLES)
    custom_keys = list(CUSTOM_TITLES)
    FAIL_KEYS = {'emir'}

    # single-scale section
    single_figs, single_rows = [], []
    for key in ('cathedral', 'monastery', 'tobolsk'):
        rec = single[fname_for(key)]
        title = PROVIDED_TITLES[key]
        single_figs.append(figure('single_scale', key, title, f'<br>{offsets_caption(rec)}'))
        single_rows.append(
            f'          <tr><td>{title}</td><td class="num">{fmt_shift(rec["g_shift"])}</td>'
            f'<td class="num">{fmt_shift(rec["r_shift"])}</td>'
            f'<td class="num">{rec["seconds"]:.2f}s</td></tr>')

    def l2_gallery(keys, section_ref):
        figs, rows = [], []
        for key in keys:
            rec = l2[fname_for(key)]
            is_custom = key in CUSTOM_TITLES
            title = CUSTOM_TITLES[key] if is_custom else PROVIDED_TITLES[key]
            is_fail = key in FAIL_KEYS
            fail_note = f' <b>(misaligned, see Section {section_ref})</b>' if is_fail else ''
            figs.append(figure('pyramid_l2', key, title, f'<br>{offsets_caption(rec)}{fail_note}'))
            row_class = ' class="fail"' if is_fail else ''
            rows.append(
                f'          <tr{row_class}><td>{title}</td>'
                f'<td class="num">{fmt_shift(rec["g_shift"])}</td>'
                f'<td class="num">{fmt_shift(rec["r_shift"])}</td>'
                f'<td class="num">{rec["seconds"]:.1f}s</td></tr>')
        return ''.join(figs), '\n'.join(rows)

    # required-baseline results (L2/SSD on raw RGB pixels, through the
    # pyramid procedure): 14 provided plates, shown separately from the 3
    # self-sourced ones per the assignment's deliverable list
    provided_figs_html, provided_rows_html = l2_gallery(provided_keys, 5)
    custom_figs_html, custom_rows_html = l2_gallery(custom_keys, 5)

    single_figs_html = ''.join(single_figs)
    single_rows_html = '\n'.join(single_rows)

    emir_before = compare_figure('pyramid_l2', 'emir', 'raw-pixel L2 (before)')
    emir_after = compare_figure('golden', 'emir', 'gradient-based (after)')
    emir_l2_r = fmt_shift(l2['emir.tif']['r_shift'])
    emir_golden_r = fmt_shift(golden['emir.tif']['r_shift'])

    crop_before = compare_figure('uncropped', 'church', 'before cropping')
    crop_after = compare_figure('golden', 'church', 'after cropping')

    raw_w, raw_h = dims('media/raw/emir_raw.jpg')

    html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Project 1: Images of the Russian Empire</title>
<meta name="description" content="CS180 Project 1: colorizing the Prokudin-Gorskii glass plate negatives by aligning color channels.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&display=swap">
<link rel="stylesheet" href="../styles.css">
<link rel="stylesheet" href="editorial.css">
</head>
<body>
<main class="doc">

  <h1>Images of the Russian Empire</h1>
  <p class="sub">Project 1 &middot; Ariq Koh &middot; CS180, Fall 2026 &middot; colorizing the Prokudin-Gorskii glass plates</p>

  <p class="toc">
    <span class="nb">1. <a href="#s1">Approach</a></span>
    <span class="sep" aria-hidden="true">&middot;</span>
    <span class="nb">2. <a href="#s2">Single-scale alignment</a></span>
    <span class="sep" aria-hidden="true">&middot;</span>
    <span class="nb">3. <a href="#s3">Multi-scale pyramid results</a></span>
    <span class="sep" aria-hidden="true">&middot;</span>
    <span class="nb">4. <a href="#s4">Custom images</a></span>
    <span class="sep" aria-hidden="true">&middot;</span>
    <span class="nb">5. <a href="#s5">Bells &amp; whistles</a></span>
    <span class="sep" aria-hidden="true">&middot;</span>
    <span class="nb">6. <a href="#s6">Failure cases</a></span>
  </p>


  <h2 id="s1"><span class="secno">1</span> Approach</h2>
  <div class="intro-flex">
    <div class="intro-text">
      <p class="setup">
        Between 1909 and 1915, Sergei Prokudin-Gorskii photographed the Russian
        Empire in color decades before color film existed, by taking three exposures
        of the same scene in quick succession through red, green and blue filters.
        The three exposures were never merged. They survive as a single glass
        plate negative with the blue-filtered exposure on top, green in the middle,
        and red on the bottom. To recover the color photo, I split each plate into
        thirds and align the green and red thirds onto the blue one by translation.
      </p>

      <p class="setup">
        <b>Scoring alignment.</b> For a candidate shift, I score it with the L2
        norm (Euclidean distance, the square root of the sum of squared
        pixel differences) between the shifted channel and the reference channel,
        where smaller is better. I score only a center crop of the image (half
        the width and height, centered), and apply the candidate shift by
        indexing directly into the array rather than <code>np.roll</code>-ing it.
        This means a shift is never scored against pixels that wrapped around
        from the opposite edge, and the plate's border and frame, which is
        often scratched, mismatched between exposures, or simply a different
        shape per channel, can never bias the match.
      </p>
    </div>

    <figure>
      <img src="media/raw/emir_raw.jpg" width="{raw_w}" height="{raw_h}" loading="lazy"
           alt="Raw, unaligned Prokudin-Gorskii glass plate showing three stacked black-and-white exposures">
      <figcaption><b>Figure 1.</b> A raw plate before alignment: blue exposure on top, green in the middle, red on the bottom.</figcaption>
    </figure>
  </div>

  <p class="setup">
    <b>Single-scale search.</b> For the three low-resolution JPEGs, an
    exhaustive search over a small window (&plusmn;15px in x and y) around
    zero is fast enough and wide enough to find the true offset directly.
  </p>

  <p class="setup">
    <b>Multi-scale pyramid.</b> The full-resolution scans are too large
    (roughly 3500&times;9600px per channel) for an exhaustive &plusmn;15px
    search to reliably cover the true offset, which can be much larger at
    full resolution. Instead I build an image pyramid: repeatedly downscale
    both channels by half until the smaller dimension drops under a
    <code>min_size</code> threshold, exhaustively search a small window
    (&plusmn;4px) at that coarse scale, then walk back up the pyramid,
    doubling the shift estimate at each level and refining it with another
    small &plusmn;4px search. This keeps the search cheap at every scale
    while still covering a wide range of true offsets, since the doubling
    compounds across levels.
  </p>

  <h2 id="s2"><span class="secno">2</span> Single-scale alignment</h2>
  <p class="setup">
    Results of the single-scale, raw-pixel L2 search (&plusmn;15px window)
    on the three low-resolution JPEGs.
  </p>

  <div class="resultsgrid">
{single_figs_html}  </div>

  <table class="offsets-table">
    <thead><tr><th>Image</th><th>G offset (x, y)</th><th>R offset (x, y)</th><th>Time</th></tr></thead>
    <tbody>
{single_rows_html}
    </tbody>
  </table>


  <h2 id="s3"><span class="secno">3</span> Multi-scale pyramid results</h2>
  <p class="setup">
    Results of the multi-scale pyramid alignment, with the L2 norm computed
    directly on raw RGB pixel values (as required), on all 14 provided
    example plates. All 14 use the exact same procedure and parameters,
    with nothing tuned per image. One of the 14 (emir) comes out visibly
    misaligned with this metric. See Section 5 for why, and for the fix.
  </p>

  <div class="resultsgrid">
{provided_figs_html}  </div>

  <table class="offsets-table">
    <thead><tr><th>Image</th><th>G offset (x, y)</th><th>R offset (x, y)</th><th>Time</th></tr></thead>
    <tbody>
{provided_rows_html}
    </tbody>
  </table>


  <h2 id="s4"><span class="secno">4</span> Custom images</h2>
  <p class="setup">
    The same procedure (multi-scale pyramid, L2 on raw RGB pixels) applied
    to 3 more plates I downloaded from the
    <a href="https://www.loc.gov/collections/prokudin-gorskii/?st=grid">Library
    of Congress Prokudin-Gorskii collection</a>. All 3 align correctly with
    this metric.
  </p>

  <div class="resultsgrid">
{custom_figs_html}  </div>

  <table class="offsets-table">
    <thead><tr><th>Image</th><th>G offset (x, y)</th><th>R offset (x, y)</th><th>Time</th></tr></thead>
    <tbody>
{custom_rows_html}
    </tbody>
  </table>


  <h2 id="s5"><span class="secno">5</span> Bells &amp; whistles</h2>

  <h3>Better features</h3>
  <p class="setup">
    The required baseline metric is L2/SSD computed directly on raw RGB
    pixel values, shown in Section 3 and Section 4 above. On 16 of the 17
    plates (all 3 custom ones included) this already works fine. But on
    <b>emir</b>, raw-pixel L2 converges to a confidently wrong answer. The
    problem isn't the search window being too narrow (I verified a wider
    window finds the exact same wrong optimum), but that the metric itself
    prefers the wrong shift: the tiled stonework around the doorway repeats
    often enough that a few wrong offsets line it up just as well as the
    correct one, once pixel intensities are compared directly.
  </p>
  <p class="setup">
    The fix: score alignment on each channel's Sobel gradient magnitude
    instead of its raw pixel values (still with L2/SSD as the metric, just
    on a different feature). Edges are far more distinctive than flat
    regions and largely invariant to the brightness differences between
    color channels, which fixes the failure without regressing any of the
    other 16 images.
  </p>

  <div class="bw-block">
    <h3>Emir of Bukhara: before and after</h3>
    <figure>
      <div class="pair">
{emir_before}
{emir_after}
      </div>
      <figcaption>Left: raw-pixel L2 (R offset {emir_l2_r}, clearly wrong). Right: gradient-based (R offset {emir_golden_r}).</figcaption>
    </figure>
  </div>

  <h3>Automatic border cropping</h3>
  <p class="setup">
    The final composite is built by shifting each channel with
    <code>np.roll</code>, which wraps content around from the far edge
    instead of leaving a gap. Whatever we shifted a channel by is exactly
    how much wrapped-in, mismatched content now sits on that edge: a
    positive shift wraps garbage in at the top or left, a negative one at
    the bottom or right. Since the shift is already known, the fix doesn't
    need any edge detection, just crop off the larger of the two channels'
    margins on each side.
  </p>

  <div class="bw-block">
    <h3>Church: before and after</h3>
    <figure>
      <div class="pair">
{crop_before}
{crop_after}
      </div>
      <figcaption>Left: before cropping (the wrapped-in top strip is a scoring artifact, not real content). Right: after cropping.</figcaption>
    </figure>
  </div>


  <h2 id="s6"><span class="secno">6</span> Failure cases</h2>
  <p class="setup">
    With the required baseline (multi-scale pyramid, L2 on raw RGB pixels,
    Sections 3 and 4), 1 of the 17 images, <b>emir</b>, comes out visibly
    misaligned (all 3 custom images and the other 13 provided ones align
    correctly). The search itself isn't the problem (a wider window
    converges to the exact same wrong answer). Raw pixel intensities just
    aren't a distinctive enough signal on this plate: the repeating tiled
    pattern around the doorway lines up just as well at the wrong offset as
    at the correct one. See Section 5 for the fix (gradient features),
    which brings it back into alignment. 0/17 fail with the full pipeline.
  </p>

</main>
</body>
</html>
'''

    out_path = os.path.join(PROJ, 'index.html')
    with open(out_path, 'w') as f:
        f.write(html)
    print('wrote', out_path)


if __name__ == '__main__':
    build()
