#!/usr/bin/env python
"""
Staged ablation for CLAM-SB over the k-fold splits, with automatic selection of the
winning configuration.

Why staged and not a full grid: the axes below multiply out to 1728 configurations, each
costing 5 folds. Beyond the compute, searching that many configurations makes the
selection WORSE, not better: with 5 folds the mean val_auc carries a standard error near
0.03, and the maximum over n noisy estimates is inflated by roughly sqrt(2 ln n) standard
errors -- about 0.08 at n=26 but 0.12 at n=1728, by which point the winner is mostly luck.

So one axis is varied at a time and the winner is frozen, EXCEPT for capacity and
regularisation, which are explored jointly because they genuinely interact. 26 runs
instead of 1728, at the cost of assuming the remaining axes interact weakly -- an
assumption worth stating in the thesis rather than hiding.

THE RULE THIS FILE ENFORCES: configurations are ranked on val_auc. The test columns of
summary.csv are not read at all unless --reveal-test is passed, which is meant to be
used exactly once, on the final winner. Selecting on test turns the test estimate into a
training estimate and invalidates every number reported from it.

Workflow
--------
    # 1. write the configurations and the SLURM array for the first stage
    python ablation_clam_sb.py plan --stage 0 --feat_dir features_univ1_20x

    # 2. launch it
    sbatch ablation/stage0.sh

    # 3. rank the finished runs; the winner is stored and carried into the next stage
    python ablation_clam_sb.py report --stage 0

    # 4. next stage, already built on top of the stage-0 winner
    python ablation_clam_sb.py plan --stage 1
    sbatch ablation/stage1.sh
    ...

    # once every stage is decided: repeat the winner under several seeds
    python ablation_clam_sb.py final --seeds 42,1,7
    sbatch ablation/final.sh

    # at the very end, once and only once:
    python ablation_clam_sb.py report --final --reveal-test
"""
import argparse
import glob
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ABL_DIR = 'ablation'
MANIFEST = os.path.join(ABL_DIR, 'manifest.json')
WINNERS = os.path.join(ABL_DIR, 'winners.json')
# kept out of the top-level results/, which is also used for manual runs: 26 ablation
# folders would otherwise bury them, and a manual run reusing an exp_code would collide
RESULTS_ROOT = os.path.join(ABL_DIR, 'results')

# ---------------------------------------------------------------------------------
# Starting configuration. Everything a stage does not vary is held at these values, so
# that within a stage the only difference between runs is the axis under test.
# ---------------------------------------------------------------------------------
BASE = {
    'task': 'task_1_tumor_vs_normal',
    'model_type': 'clam_sb',
    'model_size': 'small',
    'embed_dim': 1024,
    'k': 5,
    'seed': 42,
    'max_epochs': 120,
    'drop_out': 0.25,
    'lr': 1e-4,
    'reg': 1e-4,
    'opt': 'adamw',
    'scheduler': 'plateau',
    'scheduler_patience': 5,
    'scheduler_gamma': 0.5,
    'scheduler_min_lr': 1e-6,
    'warmup_epochs': 5,
    'early_stopping': True,
    'early_stopping_metric': 'auc',
    'patience': 12,
    'stop_epoch': 0,
    'min_delta': 0.001,
    'bag_loss': 'ce',
    'inst_loss': 'ce',
    'bag_weight': 0.7,
    'B': 8,
    'weighted_sample': True,
    'patch_drop': 0.0,
    'log_data': True,
}

# ---------------------------------------------------------------------------------
# The stages. Each is (name, description, [list of overrides]). Order matters: capacity
# first because it has the largest effect on an overfitting model, and everything
# downstream is cheaper to tune once the model is the right size.
# ---------------------------------------------------------------------------------
def grid(**axes):
    """Cartesian product of the given axes, as a list of override dicts."""
    keys = sorted(axes)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(axes[k] for k in keys))]


STAGES = [
    ('capacity', 'trunk width x regularisation, explored JOINTLY rather than greedily: '
                 'these two axes interact, since a supertiny model is already regularised '
                 'by its size while a small one needs the dropout and weight decay. Picking '
                 'the width first at a fixed dropout can discard a width that would have won '
                 'with the right regularisation',
     grid(model_size=['small', 'tiny', 'supertiny'],
          drop_out=[0.25, 0.5],
          reg=[1e-4, 1e-2])),
    ('optimisation', 'learning rate x schedule', [
        {'lr': 2e-4, 'scheduler': 'plateau'},
        {'lr': 1e-4, 'scheduler': 'plateau'},
        {'lr': 5e-5, 'scheduler': 'plateau'},
        {'lr': 1e-4, 'scheduler': 'cosine'},
    ]),
    ('clam', 'instance branch: bag/instance loss balance and patches sampled per bag', [
        {'bag_weight': 0.7, 'B': 8},
        {'bag_weight': 0.9, 'B': 8},
        {'bag_weight': 0.7, 'B': 16},
        {'bag_weight': 1.0, 'B': 8, 'no_inst_cluster': True},   # instance branch off
    ]),
    ('imbalance', 'how the 81/19 split is compensated', [
        {'weighted_sample': True, 'bag_loss': 'ce'},
        {'weighted_sample': False, 'bag_loss': 'focal', 'focal_alpha': 'auto', 'focal_gamma': 2.0},
        {'weighted_sample': False, 'bag_loss': 'ce'},           # no compensation, the baseline
    ]),
    ('augmentation', 'bag-level patch dropping, the only augmentation left', [
        {'patch_drop': 0.0},
        {'patch_drop': 0.3},
        {'patch_drop': 0.5},
    ]),
]

RANK_METRIC = 'val_auc'
VAL_COLS = ['val_auc', 'val_f1_macro', 'val_bal_acc', 'val_acc']
TEST_COLS = ['test_auc', 'test_f1_macro', 'test_bal_acc', 'test_acc']

# Smallest AUC difference worth acting on. Two configurations closer than this are
# treated as tied however consistent the difference is across folds: with ~29 positives
# per fold, a 0.005 AUC gap is a handful of slides swapping rank and will not survive a
# different seed. Without this floor the paired standard error alone can declare a
# meaningless 0.002 "significant" simply because it repeats in every fold.
TIE_THRESHOLD = 0.01


# ---------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------
def load_json(path, default):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def cfg_to_cli(cfg):
    """Render a config dict as main.py arguments. True means a store_true flag."""
    parts = []
    for key in sorted(cfg):
        val = cfg[key]
        if isinstance(val, bool):
            if val:
                parts.append('--{}'.format(key))
        elif val is not None:
            parts.append('--{} {}'.format(key, val))
    return ' '.join(parts)


def results_dir_for(cfg, exp_code, results_root=RESULTS_ROOT):
    """Mirror the folder main.py builds, so report can find the run without guessing."""
    return os.path.join(results_root, '{}_{}_{}_s{}'.format(
        exp_code, cfg['model_type'], cfg['model_size'], cfg['seed']))


def label_for(overrides):
    """Short human-readable name for one point of an axis."""
    return ' '.join('{}={}'.format(k, v) for k, v in sorted(overrides.items()))


def accumulated_base():
    """BASE plus every winner accepted so far, so each stage builds on the last."""
    cfg = dict(BASE)
    for stage_name, entry in sorted(load_json(WINNERS, {}).items()):
        cfg.update(entry['overrides'])
    return cfg


# ---------------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------------
SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=HER2_ABL_S{stage}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonardo.meloni@unibo.it
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem={mem}
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --chdir={chdir}
#SBATCH --output={abl}/logs/stage{stage}_%A_%a.txt
#SBATCH --error={abl}/logs/stage{stage}_%A_%a.err
#SBATCH --array=0-{last}%{concurrent}

source {venv}

# record which code produced these results: with auto_skip and requeues it is otherwise
# impossible to tell afterwards whether two runs came from the same source tree
echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "uncommitted files: $(git status --porcelain 2>/dev/null | wc -l)"
echo "stage {stage} ({name}), task $SLURM_ARRAY_TASK_ID"
date

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" {abl}/stage{stage}.txt)
echo "config: $CONFIG"
python main.py $CONFIG

date
"""


def cmd_plan(args):
    if not 0 <= args.stage < len(STAGES):
        sys.exit('stage must be between 0 and {}'.format(len(STAGES) - 1))

    name, description, variants = STAGES[args.stage]
    base = accumulated_base()
    base['data_root_dir'] = args.data_root_dir
    base['feat_dir'] = args.feat_dir
    base['split_dir'] = args.split_dir
    base['results_dir'] = args.results_root
    if args.embed_dim is not None:
        base['embed_dim'] = args.embed_dim

    winners = load_json(WINNERS, {})
    if winners:
        print('carrying forward {} accepted winner(s):'.format(len(winners)))
        for k, v in sorted(winners.items()):
            print('  stage {}: {}'.format(k, label_for(v['overrides'])))
        print()

    os.makedirs(os.path.join(ABL_DIR, 'logs'), exist_ok=True)
    os.makedirs(args.results_root, exist_ok=True)   # main.py only does a single-level mkdir
    manifest = load_json(MANIFEST, {})
    lines = []

    print('stage {} -- {}\n{}\n'.format(args.stage, name, description))
    for idx, overrides in enumerate(variants):
        cfg = dict(base)
        cfg.update(overrides)
        exp_code = 'ABL_S{}_{}'.format(args.stage, idx)
        cfg['exp_code'] = exp_code

        rdir = results_dir_for(cfg, exp_code, args.results_root)
        lines.append(cfg_to_cli(cfg))
        manifest[exp_code] = {'stage': args.stage, 'stage_name': name, 'index': idx,
                              'overrides': overrides, 'label': label_for(overrides),
                              'results_dir': rdir, 'k': cfg['k']}
        flag = ' (already has results)' if os.path.isfile(os.path.join(rdir, 'summary.csv')) else ''
        print('  [{}] {:<44} -> {}{}'.format(idx, label_for(overrides), rdir, flag))

    txt_path = os.path.join(ABL_DIR, 'stage{}.txt'.format(args.stage))
    with open(txt_path, 'w', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')

    sh_path = os.path.join(ABL_DIR, 'stage{}.sh'.format(args.stage))
    with open(sh_path, 'w', newline='\n') as f:
        f.write(SBATCH_TEMPLATE.format(
            stage=args.stage, name=name, last=len(variants) - 1,
            concurrent=args.concurrent, abl=ABL_DIR, chdir=args.chdir,
            venv=args.venv, time=args.time, mem=args.mem))

    save_json(MANIFEST, manifest)
    print('\nwrote {} ({} configs) and {}'.format(txt_path, len(lines), sh_path))
    print('launch with:  sbatch {}'.format(sh_path))


# ---------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------
def collect(exp_codes, manifest, reveal_test):
    """One row per configuration: mean and std over the folds of summary.csv."""
    rows, per_fold = [], {}
    cols = VAL_COLS + (TEST_COLS if reveal_test else [])

    for exp_code in exp_codes:
        entry = manifest[exp_code]
        path = os.path.join(entry['results_dir'], 'summary.csv')
        if not os.path.isfile(path):
            rows.append({'exp_code': exp_code, 'label': entry['label'],
                         'stage': entry['stage'], 'folds': 0, 'status': 'not run'})
            continue

        df = pd.read_csv(path)
        expected = entry.get('k', 5)
        status = 'ok' if len(df) == expected else 'PARTIAL {}/{}'.format(len(df), expected)

        row = {'exp_code': exp_code, 'label': entry['label'], 'stage': entry['stage'],
               'folds': len(df), 'status': status}
        for c in cols:
            if c in df.columns:
                row[c] = df[c].mean()
                row[c + '_std'] = df[c].std(ddof=1) if len(df) > 1 else np.nan
        rows.append(row)
        if RANK_METRIC in df.columns:
            per_fold[exp_code] = df.set_index('folds')[RANK_METRIC]

    return pd.DataFrame(rows), per_fold


def is_tie(delta, standard_error):
    """
    Two configurations are tied when the difference is either too small to matter
    (below TIE_THRESHOLD) or too noisy to trust (within two paired standard errors).
    Both conditions are needed: the statistical one alone calls a perfectly consistent
    0.002 significant, the practical one alone ignores how reproducible it is.
    """
    return abs(delta) < TIE_THRESHOLD or abs(delta) < 2 * standard_error


def paired_deltas(per_fold, best_code):
    """
    Difference against the best configuration, computed fold by fold.

    Every configuration is evaluated on the same folds, so a paired comparison removes
    the fold-to-fold variance that dominates the raw standard deviations -- some folds
    are simply harder than others. Without pairing, two configurations differing by a
    real 0.02 can look indistinguishable behind a 0.06 between-fold std.
    """
    if best_code not in per_fold:
        return {}
    ref = per_fold[best_code]
    out = {}
    for code, series in per_fold.items():
        common = ref.index.intersection(series.index)
        if len(common) < 2:
            continue
        d = (series.loc[common] - ref.loc[common]).astype(float)
        out[code] = (d.mean(), d.std(ddof=1) / np.sqrt(len(d)))   # mean diff, its standard error
    return out


def cmd_report(args):
    manifest = load_json(MANIFEST, None)
    if manifest is None:
        sys.exit('no {} -- run "plan" first'.format(MANIFEST))

    if args.final:
        return report_final(manifest, args)

    if args.all:
        codes = sorted(manifest, key=lambda c: (manifest[c]['stage'], manifest[c]['index']))
        title = 'every configuration planned so far'
    else:
        codes = sorted([c for c in manifest if manifest[c]['stage'] == args.stage],
                       key=lambda c: manifest[c]['index'])
        if not codes:
            sys.exit('nothing planned for stage {}'.format(args.stage))
        title = 'stage {} -- {}'.format(args.stage, STAGES[args.stage][0])

    df, per_fold = collect(codes, manifest, args.reveal_test)
    done = df[df['status'] == 'ok'].copy()
    missing = df[df['status'] != 'ok']

    print('\n' + title)
    print('ranked on {} (validation), {} of {} configurations complete\n'.format(
        RANK_METRIC, len(done), len(df)))

    if len(done) == 0:
        print('no completed run yet.')
        for _, r in missing.iterrows():
            print('  {:<14} {:<40} {}'.format(r['exp_code'], r['label'], r['status']))
        return

    done = done.sort_values(RANK_METRIC, ascending=False).reset_index(drop=True)
    best_code = done.loc[0, 'exp_code']
    deltas = paired_deltas(per_fold, best_code)

    header = '{:<4}{:<14}{:<38}{:>15}{:>15}{:>19}{:>6}'.format(
        '#', 'exp_code', 'configuration', 'val_auc', 'val_f1_macro', 'vs best (paired)', '')
    print(header)
    print('-' * len(header))
    for i, r in done.iterrows():
        mark = ''
        if r['exp_code'] == best_code:
            delta = '(best)'
        elif r['exp_code'] in deltas:
            d, se = deltas[r['exp_code']]
            mark = 'tie' if is_tie(d, se) else ''
            delta = '{:+.4f} +-{:.4f}'.format(d, se)
        else:
            delta = ''
        print('{:<4}{:<14}{:<38}{:>8.4f} +-{:.3f}{:>8.4f} +-{:.3f}{:>19}{:>6}'.format(
            i, r['exp_code'], r['label'][:37],
            r[RANK_METRIC], r[RANK_METRIC + '_std'],
            r['val_f1_macro'], r['val_f1_macro_std'], delta, mark))
    print('\n"tie" = difference below {:.3f} AUC, or within two paired standard errors of '
          'zero.'.format(TIE_THRESHOLD))

    if len(missing):
        print('\nincomplete:')
        for _, r in missing.iterrows():
            print('  {:<14} {:<40} {}'.format(r['exp_code'], r['label'], r['status']))

    if args.reveal_test:
        print('\n--- TEST metrics, for the final winner only ---')
        for _, r in done.iterrows():
            print('  {:<14} test_auc {:.4f}+-{:.3f}   test_f1_macro {:.4f}+-{:.3f}   '
                  'test_bal_acc {:.4f}+-{:.3f}'.format(
                      r['exp_code'], r['test_auc'], r['test_auc_std'],
                      r['test_f1_macro'], r['test_f1_macro_std'],
                      r['test_bal_acc'], r['test_bal_acc_std']))
        print('\nThese are valid ONLY for the configuration chosen on validation. Quoting the '
              'best test number across this table would be selecting on test.')

    out_csv = os.path.join(ABL_DIR, 'report_{}.csv'.format('all' if args.all else 'stage{}'.format(args.stage)))
    df.to_csv(out_csv, index=False)
    print('\nwrote {}'.format(out_csv))

    if args.plot:
        make_plot(done, deltas, best_code, title,
                  os.path.join(ABL_DIR, 'report_{}.png'.format('all' if args.all else 'stage{}'.format(args.stage))))

    # accept the winner into the running configuration
    if not args.all and not args.no_accept:
        winners = load_json(WINNERS, {})
        entry = manifest[best_code]
        n_tied = sum(1 for c, (d, se) in deltas.items() if c != best_code and is_tie(d, se))
        winners[str(args.stage)] = {'exp_code': best_code, 'overrides': entry['overrides'],
                                    'val_auc': float(done.loc[0, RANK_METRIC]), 'n_tied': n_tied}
        save_json(WINNERS, winners)
        print('\nwinner of stage {}: {}  ({} = {:.4f})'.format(
            args.stage, label_for(entry['overrides']), RANK_METRIC, done.loc[0, RANK_METRIC]))
        if n_tied:
            print('NOTE: {} other configuration(s) are statistically tied with it. The choice is '
                  'arbitrary among them -- prefer the smaller/simpler one and say so in the '
                  'thesis. Override with: plan --stage {} after editing {}'.format(
                      n_tied, args.stage + 1, WINNERS))
        print('carried into every later stage. Next:  python {} plan --stage {}'.format(
            os.path.basename(__file__), args.stage + 1))


def make_plot(done, deltas, best_code, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = done.iloc[::-1]                       # best at the top
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(d) + 2.2))
    colours = ['#1f77b4' if c == best_code else
               ('#9ecae1' if c in deltas and is_tie(*deltas[c]) else '#c6c6c6')
               for c in d['exp_code']]
    ax.barh(y, d[RANK_METRIC], xerr=d[RANK_METRIC + '_std'], color=colours,
            error_kw={'ecolor': '#555', 'capsize': 3, 'lw': 1})
    ax.set_yticks(y)
    ax.set_yticklabels(['{}\n{}'.format(c, l[:44]) for c, l in zip(d['exp_code'], d['label'])], fontsize=7)
    lo = max(0.4, float(np.nanmin(d[RANK_METRIC] - d[RANK_METRIC + '_std'])) - 0.03)
    ax.set_xlim(lo, min(1.0, float(np.nanmax(d[RANK_METRIC] + d[RANK_METRIC + '_std'])) + 0.03))
    ax.axvline(0.5, color='crimson', ls=':', lw=1)
    ax.text(0.5, len(d) - 0.4, ' chance', color='crimson', fontsize=7, va='top')
    ax.set_xlabel('{} (mean over folds, error bars = std between folds)'.format(RANK_METRIC))
    ax.set_title(title, fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote {}'.format(path))


# ---------------------------------------------------------------------------------
# final: the winning configuration re-run under several seeds
# ---------------------------------------------------------------------------------
def cmd_final(args):
    """
    Repeat the accumulated winner with several training seeds.

    The splits are NOT regenerated: --split_dir still points at the same k-fold
    partition, so the folds are identical across seeds. What the seed changes is
    seed_torch() -- weight initialisation, dropout masks and the order the weighted
    sampler draws bags in. The spread across seeds is therefore optimisation variance
    alone, which is the quantity to put a +- on when reporting a single model.

    Data-partition variance is a different and larger quantity; regenerating the splits
    with a different seed would mix the two and make the folds incomparable between runs.
    """
    winners = load_json(WINNERS, {})
    if not winners and not args.force:
        sys.exit('no stage has been decided yet -- run the stages first, or pass --force')

    seeds = [int(s) for s in args.seeds.split(',')]
    base = accumulated_base()
    base['data_root_dir'] = args.data_root_dir
    base['feat_dir'] = args.feat_dir
    base['split_dir'] = args.split_dir
    base['results_dir'] = args.results_root
    if args.embed_dim is not None:
        base['embed_dim'] = args.embed_dim
    base['exp_code'] = args.exp_code

    print('final configuration ({} stage winner(s) applied):'.format(len(winners)))
    for k, v in sorted(winners.items()):
        print('  stage {}: {}'.format(k, label_for(v['overrides'])))
    print()

    os.makedirs(os.path.join(ABL_DIR, 'logs'), exist_ok=True)
    os.makedirs(args.results_root, exist_ok=True)
    manifest = load_json(MANIFEST, {})
    lines = []
    for seed in seeds:
        cfg = dict(base)
        cfg['seed'] = seed
        lines.append(cfg_to_cli(cfg))
        key = '{}_s{}'.format(args.exp_code, seed)
        rdir = results_dir_for(cfg, args.exp_code, args.results_root)
        manifest[key] = {'stage': 'final', 'stage_name': 'final', 'index': seed,
                         'overrides': {'seed': seed}, 'label': 'seed {}'.format(seed),
                         'results_dir': rdir, 'k': cfg['k']}
        print('  seed {:<6} -> {}'.format(seed, rdir))

    txt_path = os.path.join(ABL_DIR, 'final.txt')
    with open(txt_path, 'w', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')

    sh_path = os.path.join(ABL_DIR, 'final.sh')
    with open(sh_path, 'w', newline='\n') as f:
        f.write(SBATCH_TEMPLATE.format(
            stage='_final', name='final, {} seeds'.format(len(seeds)), last=len(seeds) - 1,
            concurrent=args.concurrent, abl=ABL_DIR, chdir=args.chdir,
            venv=args.venv, time=args.time, mem=args.mem
        ).replace('stage_final.txt', 'final.txt'))

    save_json(MANIFEST, manifest)
    print('\nwrote {} ({} seeds) and {}'.format(txt_path, len(lines), sh_path))
    print('launch with:  sbatch {}'.format(sh_path))
    print('then:         python {} report --final --reveal-test'.format(os.path.basename(__file__)))


def report_final(manifest, args):
    codes = sorted([c for c in manifest if manifest[c]['stage'] == 'final'],
                   key=lambda c: manifest[c]['index'])
    if not codes:
        sys.exit('no final run planned -- run "final" first')

    cols = VAL_COLS + (TEST_COLS if args.reveal_test else [])
    per_seed, pooled = [], []
    for code in codes:
        entry = manifest[code]
        path = os.path.join(entry['results_dir'], 'summary.csv')
        if not os.path.isfile(path):
            print('  seed {}: not run'.format(entry['index']))
            continue
        df = pd.read_csv(path)
        if len(df) != entry.get('k', 5):
            print('  seed {}: PARTIAL {}/{} folds, excluded'.format(
                entry['index'], len(df), entry.get('k', 5)))
            continue
        row = {'seed': entry['index'], 'folds': len(df)}
        for c in cols:
            if c in df.columns:
                row[c] = df[c].mean()
                row[c + '_std'] = df[c].std(ddof=1)
        per_seed.append(row)
        pooled.append(df)

    if not per_seed:
        sys.exit('no complete seed yet')

    ps = pd.DataFrame(per_seed)
    allfolds = pd.concat(pooled, ignore_index=True)

    print('\nfinal configuration, {} seed(s) x {} folds = {} runs'.format(
        len(ps), int(ps['folds'].iloc[0]), len(allfolds)))
    print('splits identical across seeds, so the spread below is optimisation variance\n')

    print('{:<8}{:>18}{:>18}'.format('seed', 'val_auc (5 folds)', 'val_f1_macro'))
    print('-' * 44)
    for _, r in ps.iterrows():
        # int(): iterrows() upcasts the whole row to one dtype, so seed arrives as a float
        print('{:<8}{:>11.4f} +-{:.3f}{:>11.4f} +-{:.3f}'.format(
            int(r['seed']), r['val_auc'], r['val_auc_std'], r['val_f1_macro'], r['val_f1_macro_std']))

    print('\nacross seeds (each seed = mean of its 5 folds):')
    for c in VAL_COLS:
        if c in ps.columns:
            sd = ps[c].std(ddof=1) if len(ps) > 1 else float('nan')
            print('  {:<16} {:.4f} +- {:.4f}'.format(c, ps[c].mean(), sd))

    if args.reveal_test:
        print('\n--- TEST, the number to report ---')
        for c in TEST_COLS:
            if c in ps.columns:
                sd = ps[c].std(ddof=1) if len(ps) > 1 else float('nan')
                print('  {:<16} {:.4f} +- {:.4f}   (across seeds)'.format(c, ps[c].mean(), sd))
        print('\n  pooled over all {} fold-runs: test_auc {:.4f} +- {:.4f}'.format(
            len(allfolds), allfolds['test_auc'].mean(), allfolds['test_auc'].std(ddof=1)))
        print('\nReport the across-seed figure: it separates optimisation variance from the '
              'fold-to-fold variance the pooled number conflates it with.')
    else:
        print('\n(test metrics hidden; add --reveal-test -- this run is where that is legitimate)')

    out = os.path.join(ABL_DIR, 'report_final.csv')
    ps.to_csv(out, index=False)
    print('\nwrote {}'.format(out))


# ---------------------------------------------------------------------------------
def cmd_status(args):
    winners = load_json(WINNERS, {})
    print('\naccumulated configuration (BASE + accepted winners)\n')
    base, acc = dict(BASE), accumulated_base()
    for k in sorted(acc):
        changed = ' <- stage winner' if base.get(k) != acc[k] else ''
        print('  {:<24} {}{}'.format(k, acc[k], changed))
    print('\nstages')
    for i, (name, desc, variants) in enumerate(STAGES):
        w = winners.get(str(i))
        state = 'won by {} ({} {:.4f})'.format(w['exp_code'], RANK_METRIC, w['val_auc']) if w else 'pending'
        print('  [{}] {:<16} {:>2} configs   {}'.format(i, name, len(variants), state))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    pl = sub.add_parser('plan', help='write the configurations and the SLURM array for one stage')
    pl.add_argument('--stage', type=int, required=True)
    pl.add_argument('--data_root_dir', type=str, default='/scratch.hpc/leonardo.meloni/CLAM')
    pl.add_argument('--feat_dir', type=str, default='features_univ1_20x')
    pl.add_argument('--split_dir', type=str, default='task_1_tumor_vs_normal_100_kfold')
    pl.add_argument('--results_root', type=str, default=RESULTS_ROOT,
                    help='where main.py writes the runs; kept out of the top-level results/ '
                    +'so manual experiments stay separate (default: {})'.format(RESULTS_ROOT))
    pl.add_argument('--embed_dim', type=int, default=None, help='1024 for uni_v1/resnet, 1536 for uni_v2')
    pl.add_argument('--concurrent', type=int, default=2, help='SLURM array concurrency cap (default: 2)')
    pl.add_argument('--time', type=str, default='12:00:00')
    pl.add_argument('--mem', type=str, default='32G')
    pl.add_argument('--chdir', type=str, default='/scratch.hpc/leonardo.meloni/CLAM')
    pl.add_argument('--venv', type=str, default='/scratch.hpc/leonardo.meloni/clam_env/bin/activate')
    pl.set_defaults(func=cmd_plan)

    fn = sub.add_parser('final', help='re-run the accumulated winner under several seeds')
    fn.add_argument('--seeds', type=str, default='42,1,7', help='comma separated (default: 42,1,7)')
    fn.add_argument('--exp_code', type=str, default='FINAL')
    fn.add_argument('--force', action='store_true', help='allow it before any stage has been decided')
    fn.add_argument('--data_root_dir', type=str, default='/scratch.hpc/leonardo.meloni/CLAM')
    fn.add_argument('--feat_dir', type=str, default='features_univ1_20x')
    fn.add_argument('--split_dir', type=str, default='task_1_tumor_vs_normal_100_kfold')
    fn.add_argument('--results_root', type=str, default=RESULTS_ROOT)
    fn.add_argument('--embed_dim', type=int, default=None)
    fn.add_argument('--concurrent', type=int, default=2)
    fn.add_argument('--time', type=str, default='12:00:00')
    fn.add_argument('--mem', type=str, default='32G')
    fn.add_argument('--chdir', type=str, default='/scratch.hpc/leonardo.meloni/CLAM')
    fn.add_argument('--venv', type=str, default='/scratch.hpc/leonardo.meloni/clam_env/bin/activate')
    fn.set_defaults(func=cmd_final)

    rp = sub.add_parser('report', help='rank the finished runs of a stage and accept the winner')
    rp.add_argument('--stage', type=int, default=None)
    rp.add_argument('--all', action='store_true', help='every stage at once, without accepting a winner')
    rp.add_argument('--final', action='store_true', help='aggregate the multi-seed final run')
    rp.add_argument('--reveal-test', dest='reveal_test', action='store_true',
                    help='also print the test metrics. Use once, on the final winner')
    rp.add_argument('--no-accept', dest='no_accept', action='store_true',
                    help='rank without recording the winner')
    rp.add_argument('--plot', action='store_true', default=True)
    rp.set_defaults(func=cmd_report)

    sub.add_parser('status', help='show the accumulated configuration and stage progress').set_defaults(func=cmd_status)

    args = p.parse_args()
    if args.cmd == 'report' and not args.all and not args.final and args.stage is None:
        p.error('report needs --stage N, --all or --final')
    args.func(args)


if __name__ == '__main__':
    main()
