import argparse
import csv
import os
import time

import numpy as np
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from experiment_utils import (
    apply_perturbation,
    build_model,
    build_optimizer,
    confusion_matrix_from_logits,
    dense_target_strengths,
    ensure_dir,
    evaluate_model,
    evaluate_under_perturbation,
    format_inputs,
    get_model_summary,
    load_mnist,
    load_model_from_checkpoint,
    matrix_to_rows,
    non_target_representatives,
    per_class_accuracy_from_logits,
    robustness_diagnosis,
    same_type_neighbor,
    save_csv,
    save_history_csv,
    select_target_perturbation,
    train_model,
    train_val_split,
)
import mynn as nn


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset', 'MNIST')
TRAIN_IMAGES_PATH = os.path.join(DATASET_DIR, 'train-images-idx3-ubyte.gz')
TRAIN_LABELS_PATH = os.path.join(DATASET_DIR, 'train-labels-idx1-ubyte.gz')
TEST_IMAGES_PATH = os.path.join(DATASET_DIR, 't10k-images-idx3-ubyte.gz')
TEST_LABELS_PATH = os.path.join(DATASET_DIR, 't10k-labels-idx1-ubyte.gz')


def build_parser():
    parser = argparse.ArgumentParser(description='End-to-end project pipeline.')
    parser.add_argument('--output-dir', default='outputs/project_run')
    parser.add_argument('--seed', type=int, default=309)
    parser.add_argument('--eval-seed', type=int, default=2026)
    parser.add_argument('--val-size', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--max-finetune-epochs', type=int, default=5)
    parser.add_argument('--mlp-lr', type=float, default=0.05)
    parser.add_argument('--cnn-lr', type=float, default=0.02)
    parser.add_argument('--rgft-lr', type=float, default=0.005)
    parser.add_argument('--optimizer', choices=['sgd', 'momentum'], default='sgd')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', action='store_true')
    parser.add_argument('--weight-decay-lambda', type=float, default=1e-4)
    parser.add_argument('--clean-mix-ratio', type=float, default=0.5)
    parser.add_argument('--subset-train', type=int, default=None)
    parser.add_argument('--subset-val', type=int, default=None)
    parser.add_argument('--subset-test', type=int, default=None)
    parser.add_argument('--smoke', action='store_true')
    return parser


def architecture_string(model_name):
    if model_name == 'MLP-clean':
        return '784-256-10 with ReLU'
    return 'Conv(1,8,3)-ReLU-Pool-Conv(8,16,3)-ReLU-Pool-FC(784,128)-ReLU-FC(128,10)'


def load_data(args):
    train_images, train_labels = load_mnist(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, flatten=False, normalize=True)
    train_images, train_labels, valid_images, valid_labels, _ = train_val_split(
        train_images,
        train_labels,
        val_size=args.val_size,
        seed=args.seed,
        split_path=os.path.join(args.output_dir, 'splits', 'train_val_idx.npy'),
    )
    test_images, test_labels = load_mnist(TEST_IMAGES_PATH, TEST_LABELS_PATH, flatten=False, normalize=True)

    if args.smoke:
        args.subset_train = args.subset_train or 128
        args.subset_val = args.subset_val or 64
        args.subset_test = args.subset_test or 64
        args.max_epochs = 1
        args.max_finetune_epochs = 1

    if args.subset_train is not None:
        train_images = train_images[:args.subset_train]
        train_labels = train_labels[:args.subset_train]
    if args.subset_val is not None:
        valid_images = valid_images[:args.subset_val]
        valid_labels = valid_labels[:args.subset_val]
    if args.subset_test is not None:
        test_images = test_images[:args.subset_test]
        test_labels = test_labels[:args.subset_test]

    return train_images, train_labels, valid_images, valid_labels, test_images, test_labels


def maybe_plot_learning(history, save_path, title, include_target=False):
    if plt is None:
        return
    epochs = [row['epoch'] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row['train_loss'] for row in history], marker='o', label='Train loss')
    axes[0].plot(epochs, [row['clean_val_loss'] for row in history], marker='s', label='Val loss')
    axes[0].set_title(f'{title} Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [row['train_acc'] for row in history], marker='o', label='Train acc')
    axes[1].plot(epochs, [row['clean_val_acc'] for row in history], marker='s', label='Clean val acc')
    if include_target:
        target_vals = [row['target_val_acc'] for row in history]
        axes[1].plot(epochs, target_vals, marker='^', label='Target val acc')
    axes[1].set_title(f'{title} Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_part_b_comparison(mlp_history, cnn_history, save_path):
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([x['epoch'] for x in mlp_history], [x['train_loss'] for x in mlp_history], marker='o', label='MLP train loss')
    axes[0].plot([x['epoch'] for x in cnn_history], [x['train_loss'] for x in cnn_history], marker='s', label='CNN train loss')
    axes[0].set_title('Part B Train Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot([x['epoch'] for x in mlp_history], [x['clean_val_acc'] for x in mlp_history], marker='o', label='MLP val acc')
    axes[1].plot([x['epoch'] for x in cnn_history], [x['clean_val_acc'] for x in cnn_history], marker='s', label='CNN val acc')
    axes[1].set_title('Part B Validation Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def evaluate_clean_metrics(model, model_kind, train_images, train_labels, valid_images, valid_labels, test_images, test_labels):
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    train_metrics = evaluate_model(model, model_kind, train_images, train_labels, loss_fn)
    val_metrics = evaluate_model(model, model_kind, valid_images, valid_labels, loss_fn)
    test_metrics = evaluate_model(model, model_kind, test_images, test_labels, loss_fn)
    return train_metrics, val_metrics, test_metrics


def train_clean_model(
    model_kind,
    model_name,
    learning_rate,
    args,
    train_images,
    train_labels,
    valid_images,
    valid_labels,
    test_images,
    test_labels,
    epoch_callback=None,
):
    model = build_model(model_kind, weight_decay=args.weight_decay, weight_decay_lambda=args.weight_decay_lambda)
    optimizer = build_optimizer(args.optimizer, model, learning_rate, momentum=args.momentum)
    checkpoint_path = os.path.join(args.output_dir, 'checkpoints', f'{model_name.lower().replace("-", "_")}.pickle')
    start = time.time()
    result = train_model(
        model=model,
        model_kind=model_kind,
        optimizer=optimizer,
        train_images=train_images,
        train_labels=train_labels,
        valid_images=valid_images,
        valid_labels=valid_labels,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        checkpoint_path=checkpoint_path,
        selection_mode='clean',
        seed=args.seed,
        eval_seed=args.eval_seed,
        epoch_callback=epoch_callback,
    )
    total_time = time.time() - start
    best_model = load_model_from_checkpoint(model_kind, checkpoint_path, weight_decay=args.weight_decay, weight_decay_lambda=args.weight_decay_lambda)
    train_metrics, val_metrics, test_metrics = evaluate_clean_metrics(best_model, model_kind, train_images, train_labels, valid_images, valid_labels, test_images, test_labels)
    return {
        'checkpoint': checkpoint_path,
        'history': result['history'],
        'best_state': result['best_state'],
        'model': best_model,
        'total_time_sec': total_time,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'summary': get_model_summary(
            model_name=model_name,
            model=best_model,
            optimizer_name=args.optimizer,
            batch_size=args.batch_size,
            learning_rate=learning_rate,
            epochs=args.max_epochs,
            final_train_acc=train_metrics['acc'],
            final_val_acc=val_metrics['acc'],
            final_test_acc=test_metrics['acc'],
            architecture=architecture_string(model_name),
        )
    }


def train_target_model(
    mode,
    checkpoint_name,
    learning_rate,
    args,
    target,
    train_images,
    train_labels,
    valid_images,
    valid_labels,
    baseline_checkpoint=None,
    baseline_clean_val_acc=None,
    epoch_callback=None,
):
    if mode == 'augscratch':
        model = build_model('cnn', weight_decay=args.weight_decay, weight_decay_lambda=args.weight_decay_lambda)
        selection_mode = 'target'
        max_epochs = args.max_epochs
    else:
        model = load_model_from_checkpoint('cnn', baseline_checkpoint, weight_decay=args.weight_decay, weight_decay_lambda=args.weight_decay_lambda)
        selection_mode = 'rgft'
        max_epochs = args.max_finetune_epochs
    optimizer = build_optimizer(args.optimizer, model, learning_rate, momentum=args.momentum)
    checkpoint_path = os.path.join(args.output_dir, 'checkpoints', checkpoint_name)
    start = time.time()
    result = train_model(
        model=model,
        model_kind='cnn',
        optimizer=optimizer,
        train_images=train_images,
        train_labels=train_labels,
        valid_images=valid_images,
        valid_labels=valid_labels,
        batch_size=args.batch_size,
        max_epochs=max_epochs,
        checkpoint_path=checkpoint_path,
        selection_mode=selection_mode,
        target_perturbation=target,
        clean_mix_ratio=args.clean_mix_ratio,
        baseline_clean_val_acc=baseline_clean_val_acc,
        clean_drop_limit=0.01,
        seed=args.seed,
        eval_seed=args.eval_seed,
        epoch_callback=epoch_callback,
    )
    total_time = time.time() - start
    best_model = load_model_from_checkpoint('cnn', checkpoint_path, weight_decay=args.weight_decay, weight_decay_lambda=args.weight_decay_lambda)
    return {
        'checkpoint': checkpoint_path,
        'history': result['history'],
        'best_state': result['best_state'],
        'model': best_model,
        'total_time_sec': total_time,
    }


def plot_robustness_curves(curve_rows, save_path, include_mlp=False):
    if plt is None:
        return
    perturbation_types = ['noise', 'translation', 'rotation']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    models = ['CNN-clean', 'CNN-AugScratch-target', 'CNN-RGFT-target']
    if include_mlp:
        models = ['MLP-clean'] + models
    for ax, ptype in zip(axes, perturbation_types):
        for model_name in models:
            rows = [row for row in curve_rows if row['perturbation_type'] == ptype and row['model_name'] == model_name]
            rows = sorted(rows, key=lambda r: r['severity'])
            ax.plot([row['severity'] for row in rows], [row['accuracy'] for row in rows], marker='o', label=model_name)
        ax.set_title(ptype)
        ax.set_xlabel('Severity')
        ax.set_ylabel('Accuracy')
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_dense_target_curve(curve_rows, target, save_path):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for model_name in ['CNN-clean', 'CNN-AugScratch-target', 'CNN-RGFT-target']:
        rows = [row for row in curve_rows if row['model_name'] == model_name]
        rows = sorted(rows, key=lambda r: r['severity'])
        ax.plot([row['severity'] for row in rows], [row['accuracy'] for row in rows], marker='o', label=model_name)
    ax.set_title(f'Dense Target Curve: {target["type"]}')
    ax.set_xlabel('Severity')
    ax.set_ylabel('Accuracy')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_example_grid(example_rows, images, perturbed_images, preds_a, preds_b, labels, title, save_path, max_items=10):
    if plt is None or len(example_rows) == 0:
        return
    rng = np.random.default_rng(2026)
    chosen = rng.choice(example_rows, size=min(max_items, len(example_rows)), replace=False)
    fig, axes = plt.subplots(len(chosen), 2, figsize=(4, 2 * len(chosen)))
    if len(chosen) == 1:
        axes = np.array([axes])
    for row_idx, sample_idx in enumerate(chosen):
        axes[row_idx, 0].imshow(images[sample_idx, 0], cmap='gray')
        axes[row_idx, 0].set_title(f'Clean y={labels[sample_idx]}')
        axes[row_idx, 1].imshow(perturbed_images[sample_idx, 0], cmap='gray')
        axes[row_idx, 1].set_title(f'A:{preds_a[sample_idx]} B:{preds_b[sample_idx]}')
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])
        axes[row_idx, 1].set_xticks([])
        axes[row_idx, 1].set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def append_live_progress(progress_path, row):
    ensure_dir(os.path.dirname(progress_path) or '.')
    fieldnames = [
        'stage',
        'epoch',
        'train_loss',
        'train_acc',
        'clean_val_loss',
        'clean_val_acc',
        'target_val_acc',
        'epoch_time_sec',
        'learning_rate',
        'selected_as_best',
        'selection_reason',
    ]
    exists = os.path.exists(progress_path)
    with open(progress_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def make_epoch_callback(history_path, plot_path, title, progress_path, stage_name, include_target=False):
    def _callback(history, epoch_metrics, _best_state):
        save_history_csv(history_path, history)
        maybe_plot_learning(history, plot_path, title, include_target=include_target)
        append_live_progress(progress_path, {
            'stage': stage_name,
            **epoch_metrics,
        })
    return _callback


def main():
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, 'checkpoints'))

    train_images, train_labels, valid_images, valid_labels, test_images, test_labels = load_data(args)

    part_b_dir = os.path.join(args.output_dir, 'part_b')
    part_c_dir = os.path.join(args.output_dir, 'part_c')
    ensure_dir(part_b_dir)
    ensure_dir(part_c_dir)
    live_progress_path = os.path.join(args.output_dir, 'live_progress.csv')

    mlp_run = train_clean_model(
        'mlp',
        'MLP-clean',
        args.mlp_lr,
        args,
        train_images,
        train_labels,
        valid_images,
        valid_labels,
        test_images,
        test_labels,
        epoch_callback=make_epoch_callback(
            os.path.join(part_b_dir, 'mlp_history.csv'),
            os.path.join(part_b_dir, 'mlp_learning_curve.png'),
            'MLP-clean',
            live_progress_path,
            'MLP-clean',
        ),
    )
    cnn_run = train_clean_model(
        'cnn',
        'CNN-clean',
        args.cnn_lr,
        args,
        train_images,
        train_labels,
        valid_images,
        valid_labels,
        test_images,
        test_labels,
        epoch_callback=make_epoch_callback(
            os.path.join(part_b_dir, 'cnn_history.csv'),
            os.path.join(part_b_dir, 'cnn_learning_curve.png'),
            'CNN-clean',
            live_progress_path,
            'CNN-clean',
        ),
    )

    save_history_csv(os.path.join(part_b_dir, 'mlp_history.csv'), mlp_run['history'])
    save_history_csv(os.path.join(part_b_dir, 'cnn_history.csv'), cnn_run['history'])
    maybe_plot_learning(mlp_run['history'], os.path.join(part_b_dir, 'mlp_learning_curve.png'), 'MLP-clean')
    maybe_plot_learning(cnn_run['history'], os.path.join(part_b_dir, 'cnn_learning_curve.png'), 'CNN-clean')
    plot_part_b_comparison(mlp_run['history'], cnn_run['history'], os.path.join(part_b_dir, 'mlp_vs_cnn_learning_curve.png'))
    save_csv(os.path.join(part_b_dir, 'model_summary.csv'), [mlp_run['summary'], cnn_run['summary']])

    diagnosis_grid = (
        [{'type': 'noise', 'severity': sigma} for sigma in [0.05, 0.10, 0.20]]
        + [{'type': 'translation', 'severity': shift} for shift in [1, 2, 3]]
        + [{'type': 'rotation', 'severity': angle} for angle in [5, 10, 15]]
    )
    diagnosis_rows, diagnosis_per_class = robustness_diagnosis(
        cnn_run['model'], 'cnn', valid_images, valid_labels, diagnosis_grid, clean_seed=args.eval_seed
    )
    save_csv(os.path.join(part_c_dir, 'validation_robustness_overall.csv'), diagnosis_rows)
    save_csv(os.path.join(part_c_dir, 'validation_robustness_per_class.csv'), diagnosis_per_class)
    target = select_target_perturbation(diagnosis_rows)
    save_csv(os.path.join(part_c_dir, 'selected_target.csv'), [{
        'target_type': target['type'],
        'target_severity': target['severity'],
    }])

    aug_run = train_target_model(
        'augscratch',
        'cnn_augscratch_target.pickle',
        args.cnn_lr,
        args,
        target,
        train_images,
        train_labels,
        valid_images,
        valid_labels,
        epoch_callback=make_epoch_callback(
            os.path.join(part_c_dir, 'cnn_augscratch_history.csv'),
            os.path.join(part_c_dir, 'cnn_augscratch_learning_curve.png'),
            'CNN-AugScratch-target',
            live_progress_path,
            'CNN-AugScratch-target',
            include_target=True,
        ),
    )
    rgft_run = train_target_model(
        'rgft',
        'cnn_rgft_target.pickle',
        args.rgft_lr,
        args,
        target,
        train_images,
        train_labels,
        valid_images,
        valid_labels,
        baseline_checkpoint=cnn_run['checkpoint'],
        baseline_clean_val_acc=cnn_run['val_metrics']['acc'],
        epoch_callback=make_epoch_callback(
            os.path.join(part_c_dir, 'cnn_rgft_history.csv'),
            os.path.join(part_c_dir, 'cnn_rgft_learning_curve.png'),
            'CNN-RGFT-target',
            live_progress_path,
            'CNN-RGFT-target',
            include_target=True,
        ),
    )

    save_history_csv(os.path.join(part_c_dir, 'cnn_augscratch_history.csv'), aug_run['history'])
    save_history_csv(os.path.join(part_c_dir, 'cnn_rgft_history.csv'), rgft_run['history'])
    maybe_plot_learning(aug_run['history'], os.path.join(part_c_dir, 'cnn_augscratch_learning_curve.png'), 'CNN-AugScratch-target', include_target=True)
    maybe_plot_learning(rgft_run['history'], os.path.join(part_c_dir, 'cnn_rgft_learning_curve.png'), 'CNN-RGFT-target', include_target=True)

    models = {
        'MLP-clean': ('mlp', mlp_run['model']),
        'CNN-clean': ('cnn', cnn_run['model']),
        'CNN-AugScratch-target': ('cnn', aug_run['model']),
        'CNN-RGFT-target': ('cnn', rgft_run['model']),
    }

    main_eval_perturbations = [
        ('clean', None),
        ('target', target),
        ('same_type_neighbor', same_type_neighbor(target)),
    ]
    non_targets = non_target_representatives(target)
    main_eval_perturbations.extend([
        (f'non_target_{idx+1}', perturbation) for idx, perturbation in enumerate(non_targets)
    ])

    overall_rows = []
    per_class_rows = []
    for model_name, (model_kind, model) in models.items():
        clean_metrics = evaluate_model(model, model_kind, test_images, test_labels, nn.op.MultiCrossEntropyLoss(model=model, max_classes=10))
        clean_acc = clean_metrics['acc']
        for setting_name, perturbation in main_eval_perturbations:
            if perturbation is None:
                metrics = clean_metrics
            else:
                metrics, _ = evaluate_under_perturbation(model, model_kind, test_images, test_labels, perturbation, seed=args.eval_seed)
            overall_rows.append({
                'model_name': model_name,
                'setting': setting_name,
                'perturbation_type': 'clean' if perturbation is None else perturbation['type'],
                'severity': 0 if perturbation is None else perturbation['severity'],
                'accuracy': metrics['acc'],
                'accuracy_drop': clean_acc - metrics['acc'],
            })
            for row in per_class_accuracy_from_logits(metrics['logits'], test_labels):
                per_class_rows.append({
                    'model_name': model_name,
                    'setting': setting_name,
                    'perturbation_type': 'clean' if perturbation is None else perturbation['type'],
                    'severity': 0 if perturbation is None else perturbation['severity'],
                    **row,
                })
    save_csv(os.path.join(part_c_dir, 'test_overall_results.csv'), overall_rows)
    save_csv(os.path.join(part_c_dir, 'test_per_class_results.csv'), per_class_rows)
    save_csv(os.path.join(part_c_dir, 'accuracy_drop_results.csv'), overall_rows)

    curve_rows = []
    curve_grid = diagnosis_grid
    for model_name in ['CNN-clean', 'CNN-AugScratch-target', 'CNN-RGFT-target']:
        model_kind, model = models[model_name]
        for perturbation in curve_grid:
            metrics, _ = evaluate_under_perturbation(model, model_kind, test_images, test_labels, perturbation, seed=args.eval_seed)
            curve_rows.append({
                'model_name': model_name,
                'perturbation_type': perturbation['type'],
                'severity': perturbation['severity'],
                'accuracy': metrics['acc'],
            })
    for perturbation in curve_grid:
        metrics, _ = evaluate_under_perturbation(mlp_run['model'], 'mlp', test_images, test_labels, perturbation, seed=args.eval_seed)
        curve_rows.append({
            'model_name': 'MLP-clean',
            'perturbation_type': perturbation['type'],
            'severity': perturbation['severity'],
            'accuracy': metrics['acc'],
        })
    save_csv(os.path.join(part_c_dir, 'robustness_curves.csv'), curve_rows)
    plot_robustness_curves(curve_rows, os.path.join(part_c_dir, 'robustness_curves.png'))

    dense_rows = []
    for severity in dense_target_strengths(target):
        perturbation = {'type': target['type'], 'severity': severity}
        for model_name in ['CNN-clean', 'CNN-AugScratch-target', 'CNN-RGFT-target']:
            model_kind, model = models[model_name]
            metrics, _ = evaluate_under_perturbation(model, model_kind, test_images, test_labels, perturbation, seed=args.eval_seed)
            dense_rows.append({
                'model_name': model_name,
                'severity': severity,
                'accuracy': metrics['acc'],
            })
    save_csv(os.path.join(part_c_dir, 'target_only_dense_curve.csv'), dense_rows)
    plot_dense_target_curve(dense_rows, target, os.path.join(part_c_dir, 'target_only_dense_curve.png'))

    target_metrics = {}
    target_images = {}
    for model_name in ['CNN-clean', 'CNN-AugScratch-target', 'CNN-RGFT-target']:
        model_kind, model = models[model_name]
        metrics, perturbed = evaluate_under_perturbation(model, model_kind, test_images, test_labels, target, seed=args.eval_seed)
        target_metrics[model_name] = metrics
        target_images[model_name] = perturbed
        matrix = confusion_matrix_from_logits(metrics['logits'], test_labels)
        save_csv(os.path.join(part_c_dir, f'{model_name.lower().replace("-", "_")}_target_confusion_matrix.csv'), matrix_to_rows(matrix))

    clean_matrix = confusion_matrix_from_logits(target_metrics['CNN-clean']['logits'], test_labels)
    rgft_matrix = confusion_matrix_from_logits(target_metrics['CNN-RGFT-target']['logits'], test_labels)
    confusion_shift_rows = []
    for i in range(clean_matrix.shape[0]):
        for j in range(clean_matrix.shape[1]):
            if i == j:
                continue
            clean_count = int(clean_matrix[i, j])
            rgft_count = int(rgft_matrix[i, j])
            confusion_shift_rows.append({
                'true_class': i,
                'pred_class': j,
                'clean_count': clean_count,
                'rgft_count': rgft_count,
                'absolute_reduction': clean_count - rgft_count,
                'relative_reduction': ((clean_count - rgft_count) / clean_count) if clean_count > 0 else 0.0,
            })
    confusion_shift_rows.sort(key=lambda row: row['absolute_reduction'], reverse=True)
    save_csv(os.path.join(part_c_dir, 'confusion_shift.csv'), confusion_shift_rows)

    clean_preds = np.argmax(target_metrics['CNN-clean']['logits'], axis=1)
    rgft_preds = np.argmax(target_metrics['CNN-RGFT-target']['logits'], axis=1)
    repaired_indices = np.where((clean_preds != test_labels) & (rgft_preds == test_labels))[0]
    new_error_indices = np.where((clean_preds == test_labels) & (rgft_preds != test_labels))[0]
    save_csv(os.path.join(part_c_dir, 'repaired_examples.csv'), [{'index': int(idx)} for idx in repaired_indices])
    save_csv(os.path.join(part_c_dir, 'new_errors.csv'), [{'index': int(idx)} for idx in new_error_indices])
    plot_example_grid(repaired_indices, test_images, target_images['CNN-clean'], clean_preds, rgft_preds, test_labels, 'Repaired Examples', os.path.join(part_c_dir, 'repaired_examples.png'))
    plot_example_grid(new_error_indices, test_images, target_images['CNN-clean'], clean_preds, rgft_preds, test_labels, 'New Errors', os.path.join(part_c_dir, 'new_errors.png'))

    economy_rows = [
        {
            'model_name': 'CNN-AugScratch-target',
            'source_checkpoint': 'random_init',
            'epochs': args.max_epochs,
            'extra_epochs': args.max_epochs,
            'time_sec': aug_run['total_time_sec'],
        },
        {
            'model_name': 'CNN-RGFT-target',
            'source_checkpoint': 'CNN-clean',
            'epochs': args.max_finetune_epochs,
            'extra_epochs': args.max_finetune_epochs,
            'time_sec': rgft_run['total_time_sec'],
        },
    ]
    save_csv(os.path.join(part_c_dir, 'economy_comparison.csv'), economy_rows)


if __name__ == '__main__':
    main()
