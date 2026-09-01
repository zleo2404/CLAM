import numpy as np
import torch
from utils.utils import *
import os
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from models.model_abmil import ABMIL
from models.model_transmil import TransMIL
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc as calc_auc

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics(labels, preds, n_classes):
    """
    Threshold-dependent metrics computed on hard predictions (Y_hat), to complement
    the AUC computed on probabilities.

    F1 is macro-averaged rather than binary-on-the-positive-class: this repo's
    task_1 label_dict is inverted ({'normal_tissue': 1, 'tumor_tissue': 0}), so
    sklearn's default pos_label=1 would silently score the wrong class. Per-class
    F1 is returned alongside so nothing is hidden by the averaging.
    zero_division=0 keeps folds that miss a class from raising.
    """
    metrics = {'acc':      accuracy_score(labels, preds),
               'bal_acc':  balanced_accuracy_score(labels, preds),
               'f1_macro': f1_score(labels, preds, average='macro', zero_division=0)}
    metrics['f1_per_class'] = f1_score(labels, preds, average=None,
                                       labels=list(range(n_classes)), zero_division=0)
    metrics['confusion_matrix'] = confusion_matrix(labels, preds,
                                                   labels=list(range(n_classes)))
    return metrics

def compute_auprc(labels, probs, n_classes):
    """
    Area under the precision-recall curve (average precision) for the MINORITY class.

    ROC AUC dilutes the false positive rate in the large negative class, so on a 21/79
    cohort it reads optimistically: a model can score a respectable AUC while being poor
    at the thing that matters here, finding the HER2+ slides. Average precision has no
    such dilution -- its baseline is the positive prevalence itself, 0.21 rather than
    0.5, leaving far more room between a model that finds the positives and one that
    does not.

    The positive class is taken to be the rarer one in the labels of the split being
    scored, not a hardcoded index: this repo's label_dict is inverted, and average
    precision for the majority class would silently return a near-useless number with a
    0.79 baseline. Returns (auprc, positive_class_index).
    """
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs)

    if n_classes == 2:
        counts = np.bincount(labels, minlength=2)
        if counts.min() == 0:                       # a fold that happens to miss a class
            return float('nan'), int(np.argmin(counts))
        pos = int(np.argmin(counts))
        return float(average_precision_score((labels == pos).astype(int), probs[:, pos])), pos

    aps = [average_precision_score((labels == c).astype(int), probs[:, c])
           for c in range(n_classes) if (labels == c).any()]
    return (float(np.mean(aps)) if aps else float('nan')), -1

def find_threshold(labels, probs, criterion='f1_macro', target_sensitivity=0.90):
    """
    Fit the decision cut-off on the VALIDATION split. Binary tasks only.

    Everything else in this file predicts with argmax, which is a hard 0.5 on the class
    probability. That is optimal only when the prior the model trained under matches the
    prior it is scored under, and here it does not: --weighted_sample feeds the model a
    rebalanced 50/50 stream while validation and test stay at 21/79, so the probabilities
    are shifted toward the minority class and 0.5 does not sit where it should.

    The cut-off is a parameter like any other: fitted on validation, then applied
    unchanged to test. Candidates are the observed probabilities themselves, so the
    search is exact rather than a grid approximation.

    The score is always the MINORITY class probability, matching compute_auprc, so
    "sensitivity" here means sensitivity to HER2+ regardless of the inverted label_dict.

        f1_macro     maximise macro F1 -- balances both classes, assumes equal costs
        youden       maximise tpr - fpr, the ROC point furthest from chance
        prior        predict the minority class as often as it actually occurs; changes
                     no ranking, only re-centres the decision. The most stable of the
                     four, because it optimises nothing
        sensitivity  the cut-off that reaches target_sensitivity on the minority class.
                     The clinically meaningful one for a triage model: fix the miss rate
                     you are willing to accept for HER2+, then report the specificity it
                     costs

    Returns (threshold, positive_class_index).
    """
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs)
    counts = np.bincount(labels, minlength=probs.shape[1])
    pos = int(np.argmin(counts))                       # minority class = HER2+
    y = (labels == pos).astype(int)
    s = probs[:, pos]

    if counts.min() == 0:                              # nothing to fit against
        return 0.5, pos

    if criterion == 'prior':
        rate = float(y.mean())
        return float(np.quantile(s, 1.0 - rate)), pos

    if criterion == 'youden':
        fpr, tpr, thr = roc_curve(y, s)
        return float(thr[int(np.argmax(tpr - fpr))]), pos

    if criterion == 'sensitivity':
        # the largest cut-off that still recovers target_sensitivity of the positives
        return float(np.quantile(s[y == 1], 1.0 - target_sensitivity)), pos

    if criterion == 'f1_macro':
        uniq = np.unique(s)
        candidates = (uniq[:-1] + uniq[1:]) / 2.0 if len(uniq) > 1 else np.array([0.5])
        best_t, best_score = 0.5, -1.0
        for t in candidates:
            preds = np.where(s >= t, pos, 1 - pos)
            score = f1_score(labels, preds, average='macro', zero_division=0)
            if score > best_score:
                best_t, best_score = float(t), score
        return best_t, pos

    raise ValueError('unknown threshold criterion: {}'.format(criterion))

def metrics_at_threshold(labels, probs, threshold, pos, n_classes):
    """
    Recompute the threshold-dependent metrics at a given cut-off, reusing the
    probabilities summary() already returned rather than re-running the model.
    """
    probs = np.asarray(probs)
    preds = np.where(probs[:, pos] >= threshold, pos, 1 - pos)
    metrics = compute_metrics(np.asarray(labels).astype(int), preds, n_classes)
    metrics['auprc'], _ = compute_auprc(labels, probs, n_classes)
    metrics['labels'], metrics['probs'] = labels, probs
    metrics['threshold'], metrics['threshold_class'] = float(threshold), int(pos)
    return metrics

def save_loss_curve(train_losses, val_losses, save_path, title='Training / validation loss'):
    """Save the per-epoch train and validation bag loss as a png."""
    if len(train_losses) == 0:
        print('No epochs recorded, skipping loss curve')
        return

    import matplotlib
    matplotlib.use('Agg')  # cluster nodes are headless
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_losses, label='train', marker='o', markersize=3)
    ax.plot(epochs, val_losses, label='validation', marker='o', markersize=3)

    best = int(np.argmin(val_losses))
    ax.axvline(epochs[best], color='grey', linestyle='--', linewidth=1)
    ax.annotate('best val loss {:.4f} @ epoch {}'.format(val_losses[best], epochs[best]),
                xy=(epochs[best], val_losses[best]), xytext=(4, 8),
                textcoords='offset points', fontsize=8, color='grey')

    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # epochs are discrete
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print('Saved loss curve to {}'.format(save_path))

def save_confusion_matrix(cm, save_path, class_names=None, title='Confusion matrix'):
    """Save a confusion matrix as a png, annotated with counts and row-normalized rates."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    # row-normalize for the colour scale so a dominant class does not flatten the map
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(1.8 * n_classes + 2, 1.8 * n_classes + 1.5))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label='fraction of true class')

    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, '{}\n({:.1%})'.format(cm[i, j], cm_norm[i, j]),
                    ha='center', va='center', fontsize=10,
                    color='white' if cm_norm[i, j] > 0.5 else 'black')

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print('Saved confusion matrix to {}'.format(save_path))

def class_names_from_label_dict(label_dict, n_classes, display_names=None):
    """
    Class names ordered by class index, e.g. {'normal_tissue': 1, 'tumor_tissue': 0}
    becomes ['tumor_tissue', 'normal_tissue']. Falls back to the index itself for any
    class the dict does not name, so a missing label_dict degrades to '0', '1', ...

    display_names remaps those labels to what they actually mean. The strings in the csv
    are leftovers from the upstream tumor-vs-normal task and stand in for HER2 status, so
    a confusion matrix labelled 'tumor_tissue' / 'normal_tissue' claims something the
    experiment never tested. Passing {'tumor_tissue': 'HER2+', 'normal_tissue': 'HER2-'}
    makes the figure say what the model was actually trained to predict.
    """
    names = [str(i) for i in range(n_classes)]
    for name, idx in (label_dict or {}).items():
        if isinstance(idx, int) and 0 <= idx < n_classes:
            names[idx] = str((display_names or {}).get(name, name))
    return names

def log_metrics(metrics, writer, split, epoch, n_classes):
    """Print and (optionally) log to tensorboard the metrics from compute_metrics."""
    print('{} acc: {:.4f}, balanced acc: {:.4f}, macro F1: {:.4f}, AUPRC: {:.4f}'.format(
        split, metrics['acc'], metrics['bal_acc'], metrics['f1_macro'], metrics.get('auprc', float('nan'))))
    for i in range(n_classes):
        print('class {}: F1 {:.4f}'.format(i, metrics['f1_per_class'][i]))

    if writer:
        writer.add_scalar('{}/acc'.format(split), metrics['acc'], epoch)
        writer.add_scalar('{}/bal_acc'.format(split), metrics['bal_acc'], epoch)
        writer.add_scalar('{}/f1_macro'.format(split), metrics['f1_macro'], epoch)
        if 'auprc' in metrics and np.isfinite(metrics['auprc']):
            writer.add_scalar('{}/auprc'.format(split), metrics['auprc'], epoch)
        for i in range(n_classes):
            writer.add_scalar('{}/class_{}_f1'.format(split, i), metrics['f1_per_class'][i], epoch)

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

def early_stopping_value(metric_name, val_loss, auc, metrics):
    """Pick the validation quantity EarlyStopping monitors, from what validate() already computed."""
    if metric_name == 'loss':
        return val_loss
    if metric_name == 'auc':
        return auc
    if metric_name == 'auprc':
        # nan when a validation fold happens to contain a single class; treat it as no
        # improvement rather than letting nan poison the comparison in EarlyStopping
        value = metrics['auprc']
        return -np.inf if not np.isfinite(value) else value
    if metric_name == 'f1_macro':
        return metrics['f1_macro']
    raise ValueError('unknown early stopping metric: {}'.format(metric_name))

class EarlyStopping:
    """
    Early stops training when the monitored validation metric stops improving, and keeps
    the checkpoint of its best epoch.

    min_delta guards the opposite failure: without it an improvement of 1e-6 resets the
    counter, and on a small validation split noise alone keeps training alive indefinitely.
    """
    def __init__(self, patience=20, stop_epoch=50, verbose=False, mode='min',
                 min_delta=0., metric_name='loss'):
        """
        Args:
            patience (int): epochs without improvement before stopping
            stop_epoch (int): earliest epoch at which stopping is allowed
            verbose (bool): print a message on every improvement
            mode (str): 'min' if lower is better (loss), 'max' otherwise (auc, f1)
            min_delta (float): improvement below this does not count as an improvement
            metric_name (str): 'loss', 'auc' or 'f1_macro'; also used for logging
        """
        assert mode in ('min', 'max'), mode
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.mode = mode
        self.min_delta = abs(min_delta)
        self.metric_name = metric_name
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metric = np.inf if mode == 'min' else -np.inf

    def __call__(self, epoch, metric, model, ckpt_name = 'checkpoint.pt'):
        # fold both directions into a single "higher score is better" convention
        score = -metric if self.mode == 'min' else metric

        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.save_checkpoint(metric, model, ckpt_name)
            self.counter = 0
        else:
            self.counter += 1
            print('EarlyStopping counter: {} out of {} (best val {} {:.6f})'.format(
                self.counter, self.patience, self.metric_name, self.best_metric))
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True

    def save_checkpoint(self, metric, model, ckpt_name):
        '''Saves the model whenever the monitored metric improves.'''
        if self.verbose:
            print('Validation {} improved ({:.6f} --> {:.6f}).  Saving model ...'.format(
                self.metric_name, self.best_metric, metric))
        torch.save(model.state_dict(), ckpt_name)
        self.best_metric = metric

def drop_patches(data, drop_frac, min_keep=1):
    """
    Randomly drop a fraction of a bag's instances. Training only.

    The kept indices are sorted rather than left shuffled: attention pooling is
    permutation-invariant, but TransMIL's PPEG folds the sequence into a 2D grid, so the
    raster order the coordinates were written in should be preserved.
    """
    if drop_frac <= 0:
        return data
    n = data.size(0)
    keep = max(min_keep, int(round(n * (1.0 - drop_frac))))
    if keep >= n:
        return data
    idx = torch.randperm(n, device=data.device)[:keep]
    return data[idx.sort().values]

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    elif args.bag_loss == 'focal':
        alpha = parse_focal_alpha(args.focal_alpha, train_split, args.n_classes)
        print('focal loss (gamma={}, alpha={})'.format(args.focal_gamma, alpha), end=' ')
        if alpha is not None and args.weighted_sample:
            print('\nWarning: --weighted_sample already rebalances the sampler; '
                  'combining it with focal alpha weights corrects for imbalance twice. '
                  'Consider --focal_alpha none.', end=' ')
        loss_fn = FocalLoss(gamma=args.focal_gamma, alpha=alpha).to(device)
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    
    # mil ignores size_arg (upstream behaviour); every other model type honours it
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})

    if args.model_type == 'abmil':
        model = ABMIL(**model_dict, gate=not args.no_gate)

    elif args.model_type == 'transmil':
        model = TransMIL(**model_dict)

    elif args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    _ = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    scheduler = get_scheduler(optimizer, args)
    # built after the main scheduler on purpose: LinearLR lowers the lr on construction,
    # and that lowered value is what epoch 0 must train with
    warmup_scheduler = get_warmup_scheduler(optimizer, args)
    if warmup_scheduler is not None:
        print('warmup: linear ramp over {} epochs, starting at lr {:.2e}'.format(
            args.warmup_epochs, optimizer.param_groups[0]['lr']), end=' ')
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        es_metric = getattr(args, 'early_stopping_metric', 'loss')
        early_stopping = EarlyStopping(patience=getattr(args, 'patience', 20),
                                       stop_epoch=getattr(args, 'stop_epoch', 50),
                                       verbose=True,
                                       mode='min' if es_metric == 'loss' else 'max',
                                       min_delta=getattr(args, 'min_delta', 0.),
                                       metric_name=es_metric)
        print('monitoring val {} (mode {}), patience {}, stop_epoch {}, min_delta {}'.format(
            es_metric, early_stopping.mode, early_stopping.patience,
            early_stopping.stop_epoch, early_stopping.min_delta), end=' ')
        if args.scheduler == 'plateau' and args.scheduler_patience >= early_stopping.patience:
            print('\nWarning: --scheduler_patience ({}) >= --patience ({}), so training will '
                  'stop before the plateau scheduler ever drops the lr.'.format(
                      args.scheduler_patience, early_stopping.patience), end=' ')
    else:
        early_stopping = None
    print('Done!')

    train_losses = []
    val_losses = []
    warmup_epochs = getattr(args, 'warmup_epochs', 0)
    patch_drop = getattr(args, 'patch_drop', 0.)

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:
            train_loss = train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn, patch_drop)
            stop, val_loss = validate_clam(cur, epoch, model, val_loader, args.n_classes,
                early_stopping, writer, loss_fn, args.results_dir)

        else:
            train_loss = train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, patch_drop)
            stop, val_loss = validate(cur, epoch, model, val_loader, args.n_classes,
                early_stopping, writer, loss_fn, args.results_dir)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # during the ramp only the warmup scheduler advances; the main one is held so its
        # own schedule (and, for plateau, its patience counter) starts from the full lr
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        elif scheduler is not None:
            # plateau reacts to the validation loss, the others advance on epoch count
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if scheduler is not None or warmup_epochs > 0:
            current_lr = optimizer.param_groups[0]['lr']
            print('lr: {:.2e}'.format(current_lr))
            if writer:
                writer.add_scalar('train/lr', current_lr, epoch)

        if stop:
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _, val_metrics = summary(model, val_loader, args.n_classes)
    results_dict, test_error, test_auc, acc_logger, test_metrics = summary(model, test_loader, args.n_classes)

    # Fit the decision cut-off on validation, then apply it unchanged to test. Only the
    # threshold-dependent metrics move (acc, bal_acc, F1, confusion matrix); auc and
    # auprc are threshold-free and are unaffected.
    threshold_metric = getattr(args, 'threshold_metric', 'argmax')
    if args.n_classes == 2 and threshold_metric != 'argmax':
        threshold, pos = find_threshold(val_metrics['labels'], val_metrics['probs'],
                                        threshold_metric,
                                        getattr(args, 'threshold_sensitivity', 0.90))
        print('\nDecision threshold fitted on validation ({}): {:.4f} on P(class {})'.format(
            threshold_metric, threshold, pos))
        val_metrics = metrics_at_threshold(val_metrics['labels'], val_metrics['probs'],
                                           threshold, pos, args.n_classes)
        test_metrics = metrics_at_threshold(test_metrics['labels'], test_metrics['probs'],
                                            threshold, pos, args.n_classes)
        # error was computed against argmax; keep it consistent with the new cut-off
        val_error, test_error = 1.0 - val_metrics['acc'], 1.0 - test_metrics['acc']
    else:
        val_metrics['threshold'] = test_metrics['threshold'] = 0.5

    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))
    log_metrics(val_metrics, writer, 'final/val', 0, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))
    log_metrics(test_metrics, writer, 'final/test', 0, args.n_classes)

    class_names = class_names_from_label_dict(getattr(args, 'label_dict', None), args.n_classes,
                                              getattr(args, 'class_display_names', None))

    save_loss_curve(train_losses, val_losses,
                    os.path.join(args.results_dir, 'loss_curve_{}.png'.format(cur)),
                    title='Bag loss, fold {}'.format(cur))
    save_confusion_matrix(test_metrics['confusion_matrix'],
                          os.path.join(args.results_dir, 'confusion_matrix_test_{}.png'.format(cur)),
                          class_names=class_names,
                          title='Test confusion matrix, fold {}'.format(cur))
    save_confusion_matrix(val_metrics['confusion_matrix'],
                          os.path.join(args.results_dir, 'confusion_matrix_val_{}.png'.format(cur)),
                          class_names=class_names,
                          title='Val confusion matrix, fold {}'.format(cur))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error, test_metrics, val_metrics


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None, patch_drop = 0.):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        # clam's instance branch takes the k_sample highest and k_sample lowest attended
        # instances, so the bag must never be shrunk below 2*k_sample or topk raises
        data = drop_patches(data, patch_drop, min_keep=2 * model.k_sample)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

    return train_loss

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None, patch_drop = 0.):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        data = drop_patches(data, patch_drop)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

    return train_loss


def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    preds = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            preds[batch_idx] = Y_hat.item()

            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')

    metrics = compute_metrics(labels, preds, n_classes)
    metrics['auprc'], auprc_class = compute_auprc(labels, prob, n_classes)

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    log_metrics(metrics, writer, 'val', epoch, n_classes)
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    if early_stopping:
        assert results_dir
        monitored = early_stopping_value(early_stopping.metric_name, val_loss, auc, metrics)
        early_stopping(epoch, monitored, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True, val_loss

    return False, val_loss

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    preds = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            preds[batch_idx] = Y_hat.item()

            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    metrics = compute_metrics(labels, preds, n_classes)
    metrics['auprc'], auprc_class = compute_auprc(labels, prob, n_classes)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    log_metrics(metrics, writer, 'val', epoch, n_classes)
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        monitored = early_stopping_value(early_stopping.metric_name, val_loss, auc, metrics)
        early_stopping(epoch, monitored, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True, val_loss

    return False, val_loss

def summary(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    metrics = compute_metrics(all_labels, all_preds, n_classes)
    metrics['auprc'], _ = compute_auprc(all_labels, all_probs, n_classes)
    # carried out so a cut-off fitted on validation can be applied to these same
    # predictions without a second forward pass over the split
    metrics['labels'], metrics['probs'] = all_labels, all_probs

    return patient_results, test_error, auc, acc_logger, metrics
