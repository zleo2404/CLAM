#!/usr/bin/env python
"""
Optuna hyperparameter search for CLAM-SB over the k-fold splits.

Complementary to ablation_clam_sb.py, not a replacement. The ablation answers "which
choices matter", one interpretable axis at a time, and produces statements for the
thesis. This answers "what is the best this model can do", and produces a single point
in a ten-dimensional space plus an importance ranking.

How a trial works
-----------------
One trial = one configuration = the k folds of the split directory. The folds are run
ONE AT A TIME through main.py --k_start f --k_end f+1, and the running mean is reported
to Optuna after each. That is what makes pruning possible: MedianPruner abandons a trial
whose partial mean is already below the median of the completed ones, reclaiming its
remaining folds. How much that saves depends entirely on how spread out the objective is
-- report() prints the fold-runs actually reclaimed, so the saving is measured rather
than assumed. Nothing is pruned until 10 trials have finished (the medians would be
meaningless) nor before the third fold (a trial should not be judged on one lucky fold).

Selection discipline
--------------------
The objective is a VALIDATION metric. The test columns of summary.csv are never read by
this script, at all -- there is no flag to make it. Optuna maximises whatever it is
given, so pointing it at test would not merely bias the estimate, it would optimise
directly against it.

Note on multiple comparisons: the winner of n trials is inflated by roughly
sqrt(2 ln n) standard errors of the objective. At 100 trials that is about 3.0, so with
a between-fold standard error near 0.03 the best val_auc reads ~0.09 high by luck alone.
Optuna finds the top of a noisy surface faster; it does not make the surface less noisy.
Re-run the winner under several seeds before believing the margin.

Usage
-----
    pip install optuna                       # not in env.yml

    # run trials; several workers can share one study through the sqlite storage
    python optuna_clam_sb.py run --n_trials 40 --feat_dir features_univ1_20x

    # or on the cluster, as parallel workers
    python optuna_clam_sb.py sbatch --workers 3 --n_trials 15
    sbatch optuna/optuna.sh

    # the winner, the importances and the ready-to-run command
    python optuna_clam_sb.py report
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

OPT_DIR = 'optuna'
# forward slash on purpose, not os.path.join: this string is emitted into best_command.sh
# and into main.py's argv, and the search may well be driven from Windows while the job
# runs on Linux, where a backslash is a literal character in a path
RESULTS_ROOT = OPT_DIR + '/results'
STORAGE = 'sqlite:///{}'.format(os.path.join(OPT_DIR, 'study.db').replace('\\', '/'))
STUDY_NAME = 'clam_sb_her2'

# Held fixed across every trial: these are not hyperparameters of the model, they define
# the experiment. Anything varied belongs in suggest() instead.
FIXED = {
    'task': 'task_1_tumor_vs_normal',
    'model_type': 'clam_sb',
    'embed_dim': 1024,
    'k': 5,
    'seed': 42,
    'max_epochs': 120,
    'opt': 'adamw',
    'inst_loss': 'ce',
    'early_stopping': True,
    # early_stopping_metric is NOT set here: it is derived from --objective, so the epoch
    # and the configuration are never chosen on two different criteria
    'patience': 20,
    'stop_epoch': 0,
    'min_delta': 0.001,
    'scheduler_gamma': 0.5,
    'scheduler_min_lr': 1e-6,
    'scheduler_patience': 5,
}


def suggest(trial):
    """
    The search space.

    Ten dimensions is a lot for a few dozen trials, which is exactly why the importance
    plot report() writes matters: it tells you which of these Optuna actually needed to
    move, and the rest can be frozen in a second, narrower study.
    """
    cfg = {
        # capacity and regularisation -- the axes that interact most on a small cohort
        'model_size': trial.suggest_categorical('model_size', ['small', 'tiny', 'supertiny']),
        'drop_out':   trial.suggest_float('drop_out', 0.1, 0.6, step=0.05),
        'reg':        trial.suggest_float('reg', 1e-5, 1e-1, log=True),

        # optimisation
        'lr':             trial.suggest_float('lr', 1e-5, 5e-4, log=True),
        'scheduler':      trial.suggest_categorical('scheduler', ['plateau', 'cosine']),
        'warmup_epochs':  trial.suggest_int('warmup_epochs', 0, 10),

        # clam's instance branch
        'bag_weight': trial.suggest_float('bag_weight', 0.5, 1.0, step=0.1),
        'B':          trial.suggest_categorical('B', [8, 16, 32]),

        # the only augmentation available with precomputed features
        'patch_drop': trial.suggest_float('patch_drop', 0.0, 0.5, step=0.1),
    }

    # How the class imbalance is compensated. One categorical rather than two independent
    # flags, because weighted sampling and focal alpha both correct for the same thing and
    # combining them corrects for it twice.
    imbalance = trial.suggest_categorical('imbalance', ['weighted', 'focal', 'none'])
    if imbalance == 'weighted':
        cfg.update({'weighted_sample': True, 'bag_loss': 'ce'})
    elif imbalance == 'focal':
        cfg.update({'weighted_sample': False, 'bag_loss': 'focal',
                    'focal_alpha': 'auto',
                    'focal_gamma': trial.suggest_float('focal_gamma', 0.5, 4.0, step=0.5)})
    else:
        cfg.update({'weighted_sample': False, 'bag_loss': 'ce'})

    if cfg['bag_weight'] >= 0.999:
        cfg['no_inst_cluster'] = True      # bag_weight 1.0 means the instance branch is dead weight
    return cfg


# Which per-epoch criterion EarlyStopping should monitor, given what the search is
# maximising. Selecting the epoch on one metric and the configuration on another means
# the best epoch of a trial is not the epoch the trial is scored on. bal_acc and acc have
# no EarlyStopping equivalent, so they fall back to auc, the closest threshold-free one.
OBJECTIVE_TO_ES = {'val_auc': 'auc', 'val_auprc': 'auprc', 'val_f1_macro': 'f1_macro',
                   'val_bal_acc': 'auc', 'val_acc': 'auc'}


# ---------------------------------------------------------------------------------
def cfg_to_argv(cfg):
    """Render a config dict as main.py argv. True means a store_true flag."""
    argv = []
    for key in sorted(cfg):
        val = cfg[key]
        if isinstance(val, bool):
            if val:
                argv.append('--{}'.format(key))
        elif isinstance(val, float):
            # '{:.10g}' rather than str(): a step of 0.05 lands on 0.45000000000000007,
            # which is the same number but unreadable in the command the thesis quotes
            argv += ['--{}'.format(key), '{:.10g}'.format(val)]
        elif val is not None:
            argv += ['--{}'.format(key), str(val)]
    return argv


def results_dir_for(cfg, exp_code):
    return '{}/{}_{}_{}_s{}'.format(RESULTS_ROOT, exp_code, cfg['model_type'],
                                    cfg['model_size'], cfg['seed'])


def run_one_fold(cfg, fold, timeout=None):
    """Run a single fold through main.py and return its row of the partial summary."""
    argv = [sys.executable, 'main.py'] + cfg_to_argv(cfg) + [
        '--k_start', str(fold), '--k_end', str(fold + 1)]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError('fold {} failed:\n{}\n{}'.format(
            fold, proc.stdout[-3000:], proc.stderr[-3000:]))

    path = os.path.join(results_dir_for(cfg, cfg['exp_code']),
                        'summary_partial_{}_{}.csv'.format(fold, fold + 1))
    if not os.path.isfile(path):
        raise RuntimeError('fold {} produced no {}'.format(fold, path))
    return pd.read_csv(path).iloc[0]


def make_objective(args):
    import optuna

    def objective(trial):
        cfg = dict(FIXED)
        cfg.update(suggest(trial))
        cfg.update({'data_root_dir': args.data_root_dir,
                    'feat_dir': args.feat_dir,
                    'split_dir': args.split_dir,
                    'early_stopping_metric': OBJECTIVE_TO_ES[args.objective],
                    'results_dir': RESULTS_ROOT,
                    'exp_code': 'OPT_T{}'.format(trial.number)})
        if args.embed_dim is not None:
            cfg['embed_dim'] = args.embed_dim

        trial.set_user_attr('results_dir', results_dir_for(cfg, cfg['exp_code']))
        print('\n=== trial {} ===\n{}'.format(trial.number, ' '.join(cfg_to_argv(cfg))))

        scores = []
        for fold in range(cfg['k']):
            t0 = time.time()
            row = run_one_fold(cfg, fold, args.fold_timeout)
            scores.append(float(row[args.objective]))
            running = float(np.mean(scores))
            print('  fold {}: {} {:.4f}   running mean {:.4f}   ({:.0f}s)'.format(
                fold, args.objective, scores[-1], running, time.time() - t0))

            # the running mean is what the pruner compares against the other trials
            trial.report(running, fold)
            if trial.should_prune():
                trial.set_user_attr('folds_completed', len(scores))
                print('  pruned after {} folds'.format(len(scores)))
                raise optuna.TrialPruned()

        trial.set_user_attr('folds_completed', len(scores))
        trial.set_user_attr('std', float(np.std(scores, ddof=1)))
        return float(np.mean(scores))

    return objective


def get_study(create=True):
    import optuna
    os.makedirs(OPT_DIR, exist_ok=True)
    if not create:
        try:
            return optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
        except Exception:
            sys.exit('no study at {} yet -- run "python {} run" first'.format(
                STORAGE, os.path.basename(__file__)))
    return optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction='maximize',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
        # do not prune until 10 trials have finished, or the medians are meaningless;
        # and never before 2 folds, or a trial is judged on one lucky fold
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    )


# ---------------------------------------------------------------------------------
def cmd_run(args):
    if args.objective.startswith('test'):
        sys.exit('the objective must be a validation metric. Optuna maximises what it is '
                 'given, so a test objective would be optimised against directly, not merely '
                 'biased -- there would be nothing left to report.')
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    study = get_study()
    print('study "{}" at {} ({} trials so far)\n'.format(STUDY_NAME, STORAGE, len(study.trials)))
    study.optimize(make_objective(args), n_trials=args.n_trials,
                   catch=(RuntimeError, subprocess.TimeoutExpired))
    print('\ndone: {} trials in the study'.format(len(study.trials)))
    cmd_report(args)


SBATCH = """#!/bin/bash
#SBATCH --job-name=HER2_OPTUNA
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
#SBATCH --output={opt}/logs/worker_%A_%a.txt
#SBATCH --error={opt}/logs/worker_%A_%a.err
#SBATCH --array=0-{last}%{workers}

source {venv}
echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "optuna worker $SLURM_ARRAY_TASK_ID"
date

# every worker attaches to the same sqlite study and pulls the next trial, so the
# trials are distributed without any coordination beyond the database file
python optuna_clam_sb.py run --n_trials {n_trials} \\
    --data_root_dir {data_root_dir} --feat_dir {feat_dir} --split_dir {split_dir} \\
    --objective {objective}

date
"""


def cmd_sbatch(args):
    os.makedirs(os.path.join(OPT_DIR, 'logs'), exist_ok=True)
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    path = os.path.join(OPT_DIR, 'optuna.sh')
    with open(path, 'w', newline='\n') as f:
        f.write(SBATCH.format(
            time=args.time, mem=args.mem, chdir=args.chdir, venv=args.venv, opt=OPT_DIR,
            last=args.workers - 1, workers=args.workers, n_trials=args.n_trials,
            data_root_dir=args.data_root_dir, feat_dir=args.feat_dir,
            split_dir=args.split_dir, objective=args.objective))
    print('wrote {}: {} workers x {} trials = up to {} trials'.format(
        path, args.workers, args.n_trials, args.workers * args.n_trials))
    print('launch with:  sbatch {}'.format(path))


# ---------------------------------------------------------------------------------
def cmd_report(args):
    import optuna
    study = get_study(create=False)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print('\n{} trials: {} complete, {} pruned, {} failed'.format(
        len(study.trials), len(done), len(pruned), len(failed)))
    if pruned:
        saved = sum(5 - (t.user_attrs.get('folds_completed') or 0) for t in pruned)
        print('pruning saved about {} fold-runs'.format(saved))
    if not done:
        sys.exit('no completed trial yet')

    rows = []
    for t in study.trials:
        row = {'trial': t.number, 'state': t.state.name, 'value': t.value,
               'folds': t.user_attrs.get('folds_completed'),
               'std': t.user_attrs.get('std'), 'results_dir': t.user_attrs.get('results_dir')}
        row.update(t.params)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values('value', ascending=False, na_position='last')
    df.to_csv(os.path.join(OPT_DIR, 'trials.csv'), index=False)

    best = study.best_trial
    print('\n=== best trial: #{}  {} = {:.4f} (+-{:.4f} between folds) ==='.format(
        best.number, args.objective, best.value, best.user_attrs.get('std') or float('nan')))
    for k, v in sorted(best.params.items()):
        print('  {:<16} {}'.format(k, v))

    # the top few, so a margin inside the noise is visible rather than hidden
    print('\ntop {} trials:'.format(min(5, len(done))))
    for _, r in df[df['state'] == 'COMPLETE'].head(5).iterrows():
        print('  #{:<4} {:.4f}   model_size={} lr={:.2e} reg={:.2e} drop_out={} imbalance={}'.format(
            int(r['trial']), r['value'], r.get('model_size'), r.get('lr'), r.get('reg'),
            r.get('drop_out'), r.get('imbalance')))

    # which hyperparameters Optuna actually needed to move
    importances = {}
    if len(done) >= 4:
        try:
            importances = optuna.importance.get_param_importances(study)
            print('\nparameter importance (fraction of the objective variance explained):')
            for k, v in importances.items():
                print('  {:<16} {:.3f}  {}'.format(k, v, '#' * int(round(v * 40))))
        except Exception as e:
            print('\ncould not compute importances: {}'.format(e))

    cfg = dict(FIXED)
    cfg.update(suggest(optuna.trial.FixedTrial(best.params)))
    cfg.update({'data_root_dir': args.data_root_dir, 'feat_dir': args.feat_dir,
                'split_dir': args.split_dir, 'results_dir': RESULTS_ROOT,
                'early_stopping_metric': OBJECTIVE_TO_ES[args.objective],
                'exp_code': 'OPT_BEST'})

    with open(os.path.join(OPT_DIR, 'best_config.json'), 'w') as f:
        json.dump({'trial': best.number, 'objective': args.objective, 'value': best.value,
                   'params': best.params, 'full_config': cfg,
                   'importances': dict(importances)}, f, indent=2)

    command = 'python main.py ' + ' '.join(cfg_to_argv(cfg))
    with open(os.path.join(OPT_DIR, 'best_command.sh'), 'w', newline='\n') as f:
        f.write('#!/bin/bash\n# best configuration found by Optuna, trial {} ({} {:.4f})\n'
                '# selected on validation; the test metrics of this run are the ones to report,\n'
                '# and only after re-running it under several seeds\n{}\n'.format(
                    best.number, args.objective, best.value, command))

    print('\nwrote:')
    for name in ('trials.csv', 'best_config.json', 'best_command.sh'):
        print('  {}'.format(os.path.join(OPT_DIR, name)))

    if args.plot:
        make_plots(study, df, importances, args.objective)

    n = len(done)
    print('\nWith {} completed trials the winner is inflated by roughly {:.2f} standard errors '
          'of the objective purely by selection. Re-run it under several seeds before '
          'believing the margin over the runner-up.'.format(n, np.sqrt(2 * np.log(max(n, 2)))))


def make_plots(study, df, importances, objective):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    comp = df[df['state'] == 'COMPLETE'].sort_values('trial')
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(comp['trial'], comp['value'], s=22, label='trial', color='#9ecae1')
    ax.plot(comp['trial'], comp['value'].cummax(), color='#1f77b4', lw=2, label='best so far')
    pr = df[df['state'] == 'PRUNED']
    if len(pr):
        ax.scatter(pr['trial'], [comp['value'].min()] * len(pr), marker='x', s=18,
                   color='#c6c6c6', label='pruned')
    ax.set_xlabel('trial'); ax.set_ylabel(objective)
    ax.set_title('Optuna search history'); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OPT_DIR, 'history.png'), dpi=150); plt.close(fig)

    if importances:
        keys = list(importances)[::-1]
        vals = [importances[k] for k in keys]
        fig, ax = plt.subplots(figsize=(7, 0.42 * len(keys) + 1.6))
        ax.barh(range(len(keys)), vals, color='#1f77b4')
        ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys, fontsize=8)
        ax.set_xlabel('fraction of the objective variance explained')
        ax.set_title('Which hyperparameters mattered'); ax.grid(axis='x', alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OPT_DIR, 'importance.png'), dpi=150)
        plt.close(fig)
    print('  {}  {}'.format(os.path.join(OPT_DIR, 'history.png'),
                            os.path.join(OPT_DIR, 'importance.png')))


# ---------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    def shared(sp):
        sp.add_argument('--data_root_dir', type=str, default='/scratch.hpc/leonardo.meloni/CLAM')
        sp.add_argument('--feat_dir', type=str, default='features_univ1_20x')
        sp.add_argument('--split_dir', type=str, default='task_1_tumor_vs_normal_100_kfold')
        sp.add_argument('--objective', type=str, default='val_auc',
                        choices=['val_auc', 'val_auprc', 'val_f1_macro', 'val_bal_acc', 'val_acc'],
                        help='column of summary.csv to maximise. Validation only, by '
                        +'construction. val_auc is symmetric between the classes and free of '
                        +'the decision cut-off; val_auprc targets the minority class. NEVER '
                        +'val_acc in practice: predicting the majority class alone already '
                        +'scores 0.80 (default: val_auc)')
        return sp

    r = shared(sub.add_parser('run', help='run trials; several workers may share the study'))
    r.add_argument('--n_trials', type=int, default=40)
    r.add_argument('--embed_dim', type=int, default=None, help='1024 uni_v1/resnet, 1536 uni_v2')
    r.add_argument('--fold_timeout', type=int, default=None, help='seconds before a fold is abandoned')
    r.add_argument('--plot', action='store_true', default=True)
    r.set_defaults(func=cmd_run)

    s = shared(sub.add_parser('sbatch', help='write the SLURM array of parallel workers'))
    s.add_argument('--workers', type=int, default=3, help='parallel workers (default: 3)')
    s.add_argument('--n_trials', type=int, default=15, help='trials per worker (default: 15)')
    s.add_argument('--time', type=str, default='24:00:00')
    s.add_argument('--mem', type=str, default='32G')
    s.add_argument('--chdir', type=str, default='/scratch.hpc/leonardo.meloni/CLAM')
    s.add_argument('--venv', type=str, default='/scratch.hpc/leonardo.meloni/clam_env/bin/activate')
    s.set_defaults(func=cmd_sbatch)

    rp = shared(sub.add_parser('report', help='best trial, importances, ready-to-run command'))
    rp.add_argument('--plot', action='store_true', default=True)
    rp.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
