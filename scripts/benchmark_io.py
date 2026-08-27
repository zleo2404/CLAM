"""
Measure how much of a training epoch is spent just reading the .pt feature files.

Generic_MIL_Dataset.__getitem__ does a torch.load() per slide on every access, so the
whole training set is re-read from disk every epoch. This script times that read loop
alone, with no model attached, and repeats it so the second pass shows whether the OS
page cache is already absorbing the cost -- if pass 2 is much faster than pass 1, the
filesystem is caching the features and an explicit in-RAM cache would add little.

Usage (on the cluster, from the CLAM directory):
    python scripts/benchmark_io.py \
        --feat_dir /scratch.hpc/leonardo.meloni/CLAM/features_univ1_20x \
        --splits_csv splits/task_1_tumor_vs_normal_100_kfold/splits_0.csv \
        --passes 2
"""
import argparse
import os
import time

import pandas as pd
import torch


def load_one(pt_path):
    """Same read Generic_MIL_Dataset performs, dict or bare-tensor format."""
    data = torch.load(pt_path, map_location='cpu')
    if isinstance(data, dict):
        return data['features']
    return data


def run_pass(paths, label):
    n_patches = 0
    n_bytes = 0
    start = time.time()
    for p in paths:
        feats = load_one(p)
        n_patches += feats.shape[0]
        n_bytes += feats.numel() * feats.element_size()
        del feats
    elapsed = time.time() - start

    gb = n_bytes / 1e9
    print('{:<10} {:6.1f}s | {:5.2f} GB | {:6.1f} MB/s | {:.3f}s per slide | {:,} patch'.format(
        label, elapsed, gb, gb * 1000 / elapsed, elapsed / max(len(paths), 1), n_patches))
    return elapsed


def main():
    parser = argparse.ArgumentParser(description='Time the per-epoch feature read')
    parser.add_argument('--feat_dir', type=str, required=True,
                        help='feature directory containing pt_files/')
    parser.add_argument('--splits_csv', type=str, required=True,
                        help='splits_{i}.csv; the train column is used')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--passes', type=int, default=2,
                        help='how many times to read the whole split (default: 2)')
    args = parser.parse_args()

    split = pd.read_csv(args.splits_csv)[args.split].dropna().tolist()
    paths = [os.path.join(args.feat_dir, 'pt_files', '{}.pt'.format(s)) for s in split]

    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        print('WARNING: {} of {} .pt files are missing, e.g. {}'.format(
            len(missing), len(paths), missing[0]))
        paths = [p for p in paths if os.path.isfile(p)]

    print('{} slides in the {} split of {}'.format(len(paths), args.split, os.path.basename(args.splits_csv)))
    print()

    times = []
    for i in range(args.passes):
        label = 'pass {}'.format(i + 1)
        times.append(run_pass(paths, label))

    print()
    if len(times) > 1 and times[0] > 0:
        speedup = times[0] / times[-1]
        print('pass 1 -> pass {}: {:.1f}x faster'.format(len(times), speedup))
        if speedup > 3:
            print('The OS page cache is already absorbing most of the read cost, so an '
                  'explicit in-RAM cache would buy little beyond the first epoch.')
        else:
            print('Reads stay expensive across passes: an in-RAM cache would remove this '
                  'cost from every epoch after the first.')
    print()
    print('Multiply the steady-state time above by the number of epochs to get the I/O '
          'share of a fold, then compare it with the wall time of a real epoch.')


if __name__ == '__main__':
    main()