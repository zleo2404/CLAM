"""
Fit a Macenko stain reference averaged over many patches sampled from many slides.

The default path in this repo fits the normalizer on a single target patch
(`--stain_norm_target target_patches/xxx.png`), so the reference stain matrix HERef
and the reference concentrations maxCRef of the whole cohort come from one arbitrary
patch of one slide. This script estimates them instead as the average over P patches
drawn from S slides, which is both more representative of the cohort's staining
spread and reportable ("reference estimated over P patches from S slides").

Patches are sampled from the coordinates already produced by create_patches_fp.py,
so they are guaranteed to be tissue and are read at exactly the patch_level and
patch_size the feature extractor will use.

Output is an .npz with HERef (3x2) and maxCRef (2,), consumed directly by
`--stain_norm_target ref.npz` in extract_features_fp.py.

Usage (on the cluster, from the CLAM directory):
    python scripts/fit_stain_reference.py \
        --data_h5_dir patching_results_20x \
        --data_slide_dir /scratch.hpc/sabrina.tassinari/ProgettoTesi/wsi_organizzate \
        --out target_patches/stain_ref_20x.npz \
        --n_slides 60 --patches_per_slide 20

Pass --csv_path to restrict the slide list (e.g. to training slides only, if you
want to rule out any information flowing from the test slides into the reference).
"""
import argparse
import glob
import os

import h5py
import numpy as np
import openslide
import torch
import torchstain

# Macenko's published reference matrix, the torchstain default. It is used ONLY to fix
# the H/E column order of each per-patch fit before averaging; it is never averaged in.
ANCHOR_HE = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]], dtype=np.float64)


def orient(he):
    """Put the hematoxylin vector in column 0.

    torchstain's __find_HE orders the two columns with the heuristic `vMin[0] > vMax[0]`,
    which can flip H and E on atypical patches. Averaging a flipped matrix together with
    correctly oriented ones yields a physically meaningless mixture, so compare both
    orderings against the anchor (columns are unit vectors, so the dot product is the
    cosine) and keep the better one.
    """
    keep = np.dot(he[:, 0], ANCHOR_HE[:, 0]) + np.dot(he[:, 1], ANCHOR_HE[:, 1])
    swap = np.dot(he[:, 1], ANCHOR_HE[:, 0]) + np.dot(he[:, 0], ANCHOR_HE[:, 1])
    return he if keep >= swap else he[:, ::-1].copy()


def tissue_fraction(rgb, io=240, beta=0.15):
    """Fraction of pixels torchstain would keep as non-transparent.

    Mirrors __convert_rgb2od: a pixel survives when every channel has optical density
    at least beta. A patch that is mostly background gives Macenko too few pixels to
    estimate the stain plane from, and produces an unstable or degenerate fit.
    """
    od = -np.log((rgb.reshape(-1, 3).astype(np.float64) + 1) / io)
    return float(np.mean(np.all(od >= beta, axis=1)))


def sample_coords(h5_path, n_patches, rng):
    """Draw n_patches coordinates from a patches .h5, with its level and size."""
    with h5py.File(h5_path, 'r') as f:
        dset = f['coords']
        level = int(dset.attrs['patch_level'])
        size = int(dset.attrs['patch_size'])
        n = len(dset)
        if n == 0:
            return [], level, size
        idx = rng.choice(n, size=min(n_patches, n), replace=False)
        coords = dset[np.sort(idx)]        # h5py needs an increasing index list
    return coords, level, size


def main():
    parser = argparse.ArgumentParser(description='Average a Macenko stain reference over many slides')
    parser.add_argument('--data_h5_dir', type=str, required=True,
                        help='patching output directory, the one containing patches/')
    parser.add_argument('--data_slide_dir', type=str, required=True,
                        help='directory containing the WSIs')
    parser.add_argument('--slide_ext', type=str, default='.svs')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='optional csv with a slide_id column, to restrict which slides '
                        +'are sampled (default: every slide with a patches .h5)')
    parser.add_argument('--out', type=str, required=True,
                        help='output .npz path')
    parser.add_argument('--n_slides', type=int, default=60,
                        help='how many slides to sample (default: 60)')
    parser.add_argument('--patches_per_slide', type=int, default=20,
                        help='how many patches per slide (default: 20)')
    parser.add_argument('--reduce', type=str, choices=['mean', 'median'], default='mean',
                        help='how to combine the per-patch fits (default: mean; median is '
                        +'robust to pen marks and blood-heavy patches)')
    parser.add_argument('--min_tissue_frac', type=float, default=0.25,
                        help='skip patches where fewer than this fraction of pixels pass the '
                        +'optical-density threshold (default: 0.25)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    h5_files = sorted(glob.glob(os.path.join(args.data_h5_dir, 'patches', '*.h5')))
    if len(h5_files) == 0:
        raise FileNotFoundError('no .h5 found in {}'.format(os.path.join(args.data_h5_dir, 'patches')))

    if args.csv_path is not None:
        import pandas as pd
        wanted = set(pd.read_csv(args.csv_path)['slide_id'].astype(str)
                     .str.replace(args.slide_ext, '', regex=False))
        h5_files = [f for f in h5_files
                    if os.path.splitext(os.path.basename(f))[0] in wanted]
        print('{} slides kept out of the csv list'.format(len(h5_files)))

    if len(h5_files) > args.n_slides:
        h5_files = [h5_files[i] for i in sorted(rng.choice(len(h5_files), args.n_slides, replace=False))]

    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    all_he, all_mc = [], []
    n_slides_used = 0
    n_skipped_bg, n_failed = 0, 0

    print('sampling up to {} patches from each of {} slides\n'.format(
        args.patches_per_slide, len(h5_files)))

    for i, h5_path in enumerate(h5_files):
        slide_id = os.path.splitext(os.path.basename(h5_path))[0]
        slide_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        if not os.path.isfile(slide_path):
            print('[{}/{}] {}: WSI not found, skipped'.format(i + 1, len(h5_files), slide_id))
            continue

        try:
            wsi = openslide.open_slide(slide_path)
            coords, level, size = sample_coords(h5_path, args.patches_per_slide, rng)
        except Exception as e:
            print('[{}/{}] {}: could not read ({}), skipped'.format(i + 1, len(h5_files), slide_id, e))
            continue

        kept_here = 0
        for (x, y) in coords:
            rgb = np.array(wsi.read_region((int(x), int(y)), level, (size, size)).convert('RGB'))
            if tissue_fraction(rgb) < args.min_tissue_frac:
                n_skipped_bg += 1
                continue
            try:
                normalizer.fit(torch.from_numpy(rgb).permute(2, 0, 1))
                he = normalizer.HERef.numpy().astype(np.float64)
                mc = normalizer.maxCRef.numpy().astype(np.float64)
                if not (np.all(np.isfinite(he)) and np.all(np.isfinite(mc)) and np.all(mc > 0)):
                    raise ValueError('non-finite fit')
            except Exception:
                n_failed += 1
                continue
            all_he.append(orient(he))
            all_mc.append(mc)
            kept_here += 1

        if kept_here > 0:
            n_slides_used += 1
        print('[{}/{}] {}: {} patches kept (level {}, size {})'.format(
            i + 1, len(h5_files), slide_id, kept_here, level, size))

    if len(all_he) < 10:
        raise RuntimeError('only {} usable patches, refusing to build a reference from that few'.format(len(all_he)))

    HE = np.stack(all_he)          # [P, 3, 2]
    MC = np.stack(all_mc)          # [P, 2]
    reduce_fn = np.mean if args.reduce == 'mean' else np.median

    heref = reduce_fn(HE, axis=0)
    # the columns are unit vectors in OD space; the elementwise average is not, so
    # put it back on the unit sphere before using it as a mixing matrix
    heref = heref / np.linalg.norm(heref, axis=0, keepdims=True)
    maxcref = reduce_fn(MC, axis=0)

    print('\n{} patches from {} slides used ({} background, {} failed fits)'.format(
        len(all_he), n_slides_used, n_skipped_bg, n_failed))
    print('\nHERef ({} over patches, columns renormalised):'.format(args.reduce))
    for row in heref:
        print('  [{:7.4f} {:7.4f}]'.format(row[0], row[1]))
    print('maxCRef: [{:.4f} {:.4f}]  (per-patch std [{:.4f} {:.4f}])'.format(
        maxcref[0], maxcref[1], MC[:, 0].std(), MC[:, 1].std()))

    # angular spread of the individual fits around the averaged direction: a compact
    # measure of how much the staining actually varies across the cohort
    print('\nangular deviation of the per-patch fits from the average:')
    for j, stain in enumerate(('hematoxylin', 'eosin')):
        cos = np.clip(HE[:, :, j] @ heref[:, j], -1.0, 1.0)
        ang = np.degrees(np.arccos(cos))
        print('  {:<12} mean {:5.2f} deg, median {:5.2f}, p95 {:5.2f}, max {:5.2f}'.format(
            stain, ang.mean(), np.median(ang), np.percentile(ang, 95), ang.max()))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out,
             HERef=heref.astype(np.float32),
             maxCRef=maxcref.astype(np.float32),
             n_patches=len(all_he),
             n_slides=n_slides_used,
             reduce=args.reduce,
             seed=args.seed)
    print('\nsaved {}'.format(args.out))
    print('use it with: extract_features_fp.py --stain_norm_target {}'.format(args.out))


if __name__ == '__main__':
    main()
