import argparse
import csv
import os

import numpy as np
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from experiment_utils import (
    build_model,
    build_optimizer,
    ensure_dir,
    evaluate_under_perturbation,
    format_inputs,
    get_model_summary,
    load_mnist,
    load_model_from_checkpoint,
    save_csv,
    save_history_csv,
    subset_by_indices,
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


def architecture_string(model_kind):
    if model_kind == 'mlp':
        return '784-256-10 with ReLU'
    if model_kind == 'cnn':
        return 'Conv(1,8,3)-ReLU-Pool-Conv(8,16,3)-ReLU-Pool-FC(784,128)-ReLU-FC(128,10)'
    raise ValueError(model_kind)


def plot_learning_curve(history, save_path, title):
    if plt is None:
        return
    epochs = [item['epoch'] for item in history]
    train_loss = [item['train_loss'] for item in history]
    train_acc = [item['train_acc'] for item in history]
    clean_val_acc = [item['clean_val_acc'] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, marker='o', label='Train loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{title} Loss')
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_acc, marker='o', label='Train acc')
    axes[1].plot(epochs, clean_val_acc, marker='s', label='Val acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{title} Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    ensure_dir(os.path.dirname(save_path) or '.')
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


def make_epoch_callback(history_path, plot_path, title, progress_path, stage_name):
    def _callback(history, epoch_metrics, _best_state):
        save_history_csv(history_path, history)
        plot_learning_curve(history, plot_path, title)
        append_live_progress(progress_path, {
            'stage': stage_name,
            **epoch_metrics,
        })
    return _callback


def load_history_csv(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'epoch': int(row['epoch']),
                'train_loss': float(row['train_loss']),
                'train_acc': float(row['train_acc']),
                'clean_val_loss': float(row['clean_val_loss']),
                'clean_val_acc': float(row['clean_val_acc']),
                'target_val_acc': '' if row['target_val_acc'] == '' else float(row['target_val_acc']),
                'epoch_time_sec': float(row['epoch_time_sec']),
                'learning_rate': float(row['learning_rate']),
                'selected_as_best': row['selected_as_best'] == 'True',
                'selection_reason': row['selection_reason'],
            })
    return rows


def infer_clean_best_state(history):
    if not history:
        return None
    best_row = max(history, key=lambda item: item['clean_val_acc'])
    return {
        'epoch': best_row['epoch'],
        'score': best_row['clean_val_acc'],
        'clean_drop': 0.0,
        'metrics': dict(best_row),
        'reason': 'clean_val_acc',
    }


def infer_target_best_state(history):
    if not history:
        return None
    valid_rows = [item for item in history if item['target_val_acc'] != '']
    if not valid_rows:
        return None
    best_row = max(valid_rows, key=lambda item: item['target_val_acc'])
    return {
        'epoch': best_row['epoch'],
        'score': best_row['target_val_acc'],
        'clean_drop': 0.0,
        'metrics': dict(best_row),
        'reason': 'target_val_acc',
    }


def infer_rgft_best_state(history, baseline_clean_val_acc, clean_drop_limit=0.01):
    if not history:
        return None
    best_state = None
    for item in history:
        if item['target_val_acc'] == '':
            continue
        clean_drop = baseline_clean_val_acc - item['clean_val_acc']
        if clean_drop > clean_drop_limit:
            continue
        if best_state is None:
            best_state = {
                'epoch': item['epoch'],
                'score': item['target_val_acc'],
                'clean_drop': clean_drop,
                'metrics': dict(item),
                'reason': 'target_val_acc_with_clean_constraint',
            }
            continue
        if item['target_val_acc'] > best_state['score']:
            best_state = {
                'epoch': item['epoch'],
                'score': item['target_val_acc'],
                'clean_drop': clean_drop,
                'metrics': dict(item),
                'reason': 'target_val_acc_with_clean_constraint',
            }
        elif np.isclose(item['target_val_acc'], best_state['score']):
            if clean_drop < best_state['clean_drop']:
                best_state = {
                    'epoch': item['epoch'],
                    'score': item['target_val_acc'],
                    'clean_drop': clean_drop,
                    'metrics': dict(item),
                    'reason': 'smaller_clean_drop',
                }
            elif np.isclose(clean_drop, best_state['clean_drop']) and item['epoch'] < best_state['epoch']:
                best_state = {
                    'epoch': item['epoch'],
                    'score': item['target_val_acc'],
                    'clean_drop': clean_drop,
                    'metrics': dict(item),
                    'reason': 'earlier_epoch_tie_break',
                }
    return best_state


def build_parser():
    parser = argparse.ArgumentParser(description='Training entry for the MNIST project.')
    parser.add_argument('--experiment', required=True, choices=[
        'smoke',
        'mlp_clean',
        'cnn_clean',
        'cnn_augscratch',
        'cnn_rgft',
    ])
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--checkpoint-dir', default='outputs/checkpoints')
    parser.add_argument('--split-path', default='outputs/splits/train_val_idx.npy')
    parser.add_argument('--seed', type=int, default=309)
    parser.add_argument('--eval-seed', type=int, default=2026)
    parser.add_argument('--val-size', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--max-finetune-epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight-decay', action='store_true')
    parser.add_argument('--weight-decay-lambda', type=float, default=1e-4)
    parser.add_argument('--optimizer', choices=['sgd', 'momentum'], default='sgd')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--subset-train', type=int, default=None)
    parser.add_argument('--subset-val', type=int, default=None)
    parser.add_argument('--subset-test', type=int, default=None)
    parser.add_argument('--perturbation-type', choices=['noise', 'translation', 'rotation'], default=None)
    parser.add_argument('--perturbation-severity', type=float, default=None)
    parser.add_argument('--clean-mix-ratio', type=float, default=0.5)
    parser.add_argument('--baseline-checkpoint', default=None)
    parser.add_argument('--baseline-clean-val-acc', type=float, default=None)
    parser.add_argument('--subset-indices', default=None)
    parser.add_argument('--freeze-policy', choices=['all', 'head_only', 'conv2_head'], default='all')
    parser.add_argument('--drop-last', action='store_true')
    parser.add_argument('--resume-checkpoint', default=None)
    parser.add_argument('--resume-history', default=None)
    parser.add_argument('--resume-to-total-epochs', type=int, default=None)
    return parser


def default_lr(experiment):
    if experiment == 'mlp_clean':
        return 0.05
    if experiment == 'cnn_clean':
        return 0.02
    if experiment == 'cnn_augscratch':
        return 0.02
    if experiment == 'cnn_rgft':
        return 0.005
    if experiment == 'smoke':
        return 0.01
    raise ValueError(experiment)


def load_data(args):
    train_images, train_labels = load_mnist(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, flatten=False, normalize=True)
    train_images, train_labels, valid_images, valid_labels, _ = train_val_split(
        train_images,
        train_labels,
        val_size=args.val_size,
        seed=args.seed,
        split_path=args.split_path,
    )
    test_images, test_labels = load_mnist(TEST_IMAGES_PATH, TEST_LABELS_PATH, flatten=False, normalize=True)

    if args.subset_train is not None:
        train_images = train_images[:args.subset_train]
        train_labels = train_labels[:args.subset_train]
    if args.subset_val is not None:
        valid_images = valid_images[:args.subset_val]
        valid_labels = valid_labels[:args.subset_val]
    if args.subset_test is not None:
        test_images = test_images[:args.subset_test]
        test_labels = test_labels[:args.subset_test]

    if args.subset_indices is not None:
        subset_indices = np.load(args.subset_indices, allow_pickle=False).astype(np.int64)
        split_indices = np.load(args.split_path, allow_pickle=False).astype(np.int64)
        original_to_local = {int(orig_idx): local_idx for local_idx, orig_idx in enumerate(split_indices[args.val_size:])}
        local_indices = np.array([original_to_local[int(idx)] for idx in subset_indices], dtype=np.int64)
        train_images, train_labels = subset_by_indices(train_images, train_labels, local_indices)

    return train_images, train_labels, valid_images, valid_labels, test_images, test_labels


def experiment_to_model_kind(experiment):
    return 'mlp' if experiment == 'mlp_clean' else 'cnn'


def maybe_make_perturbation(args):
    if args.perturbation_type is None:
        return None
    if args.perturbation_severity is None:
        raise ValueError('Need --perturbation-severity when perturbation type is set.')
    severity = int(args.perturbation_severity) if args.perturbation_type in {'translation', 'rotation'} else float(args.perturbation_severity)
    return {
        'type': args.perturbation_type,
        'severity': severity,
    }


def run_smoke(args):
    args.subset_train = args.subset_train or 64
    args.subset_val = args.subset_val or 32
    args.subset_test = args.subset_test or 32
    args.max_epochs = 1
    args.experiment = 'cnn_clean'
    result = run_clean_experiment(args)
    print('Smoke test finished.')
    return result


def run_clean_experiment(args):
    model_kind = experiment_to_model_kind(args.experiment)
    train_images, train_labels, valid_images, valid_labels, test_images, test_labels = load_data(args)
    lr = args.lr if args.lr is not None else default_lr(args.experiment)

    if args.resume_checkpoint is not None:
        model = load_model_from_checkpoint(model_kind, args.resume_checkpoint, weight_decay=args.weight_decay, weight_decay_lambda=args.weight_decay_lambda, freeze_policy=(args.freeze_policy if model_kind == 'cnn' else None))
        existing_history = load_history_csv(args.resume_history) if args.resume_history is not None else []
        initial_best_state = infer_clean_best_state(existing_history)
        start_epoch = existing_history[-1]['epoch'] if existing_history else 0
        if args.resume_to_total_epochs is None:
            raise ValueError('Need --resume-to-total-epochs when resuming.')
        additional_epochs = args.resume_to_total_epochs - start_epoch
        if additional_epochs <= 0:
            raise ValueError('resume_to_total_epochs must be greater than the last recorded epoch.')
    else:
        model = build_model(model_kind, weight_decay=args.weight_decay, weight_decay_lambda=args.weight_decay_lambda, freeze_policy=(args.freeze_policy if model_kind == 'cnn' else None))
        existing_history = []
        initial_best_state = None
        start_epoch = 0
        additional_epochs = args.max_epochs

    optimizer = build_optimizer(args.optimizer, model, lr, momentum=args.momentum)

    output_dir = os.path.join(args.output_dir, args.experiment)
    checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.experiment}_best_model.pickle')
    progress_path = os.path.join(args.output_dir, 'live_progress.csv')

    result = train_model(
        model=model,
        model_kind=model_kind,
        optimizer=optimizer,
        train_images=train_images,
        train_labels=train_labels,
        valid_images=valid_images,
        valid_labels=valid_labels,
        batch_size=args.batch_size,
        max_epochs=additional_epochs,
        checkpoint_path=checkpoint_path,
        selection_mode='clean',
        seed=args.seed,
        eval_seed=args.eval_seed,
        epoch_callback=make_epoch_callback(
            os.path.join(output_dir, 'history.csv'),
            os.path.join(output_dir, f'{args.experiment}_learning_curve.png'),
            args.experiment,
            progress_path,
            args.experiment,
        ),
        initial_history=existing_history,
        initial_best_state=initial_best_state,
        start_epoch=start_epoch,
        drop_last=args.drop_last,
    )

    best_model = load_model_from_checkpoint(model_kind, checkpoint_path, weight_decay=args.weight_decay, weight_decay_lambda=args.weight_decay_lambda)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=best_model, max_classes=10)
    train_metrics = nn.metric.accuracy(best_model(format_inputs(train_images, model_kind)), train_labels)
    val_metrics = nn.metric.accuracy(best_model(format_inputs(valid_images, model_kind)), valid_labels)
    test_metrics = nn.metric.accuracy(best_model(format_inputs(test_images, model_kind)), test_labels)

    ensure_dir(output_dir)
    save_history_csv(os.path.join(output_dir, 'history.csv'), result['history'])
    plot_learning_curve(result['history'], os.path.join(output_dir, f'{args.experiment}_learning_curve.png'), args.experiment)

    summary = get_model_summary(
        model_name=args.experiment.replace('_', '-'),
        model=best_model,
        optimizer_name=args.optimizer,
        batch_size=args.batch_size,
        learning_rate=lr,
        epochs=(args.resume_to_total_epochs if args.resume_checkpoint is not None else args.max_epochs),
        final_train_acc=float(train_metrics),
        final_val_acc=float(val_metrics),
        final_test_acc=float(test_metrics),
        architecture=architecture_string(model_kind),
    )
    save_csv(os.path.join(output_dir, 'model_summary.csv'), [summary])
    return result


def run_cnn_augscratch(args):
    train_images, train_labels, valid_images, valid_labels, _, _ = load_data(args)
    perturbation = maybe_make_perturbation(args)
    lr = args.lr if args.lr is not None else default_lr(args.experiment)
    if args.resume_checkpoint is not None:
        model = load_model_from_checkpoint('cnn', args.resume_checkpoint, weight_decay=args.weight_decay, weight_decay_lambda=args.weight_decay_lambda, freeze_policy=args.freeze_policy)
        existing_history = load_history_csv(args.resume_history) if args.resume_history is not None else []
        initial_best_state = infer_target_best_state(existing_history)
        start_epoch = existing_history[-1]['epoch'] if existing_history else 0
        if args.resume_to_total_epochs is None:
            raise ValueError('Need --resume-to-total-epochs when resuming.')
        additional_epochs = args.resume_to_total_epochs - start_epoch
        if additional_epochs <= 0:
            raise ValueError('resume_to_total_epochs must be greater than the last recorded epoch.')
    else:
        model = build_model('cnn', weight_decay=args.weight_decay, weight_decay_lambda=args.weight_decay_lambda, freeze_policy=args.freeze_policy)
        existing_history = []
        initial_best_state = None
        start_epoch = 0
        additional_epochs = args.max_epochs
    optimizer = build_optimizer(args.optimizer, model, lr, momentum=args.momentum)

    output_dir = os.path.join(args.output_dir, args.experiment)
    checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.experiment}_best_model.pickle')
    progress_path = os.path.join(args.output_dir, 'live_progress.csv')
    result = train_model(
        model=model,
        model_kind='cnn',
        optimizer=optimizer,
        train_images=train_images,
        train_labels=train_labels,
        valid_images=valid_images,
        valid_labels=valid_labels,
        batch_size=args.batch_size,
        max_epochs=additional_epochs,
        checkpoint_path=checkpoint_path,
        selection_mode='target',
        target_perturbation=perturbation,
        clean_mix_ratio=args.clean_mix_ratio,
        seed=args.seed,
        eval_seed=args.eval_seed,
        epoch_callback=make_epoch_callback(
            os.path.join(output_dir, 'history.csv'),
            os.path.join(output_dir, f'{args.experiment}_learning_curve.png'),
            args.experiment,
            progress_path,
            args.experiment,
        ),
        initial_history=existing_history,
        initial_best_state=initial_best_state,
        start_epoch=start_epoch,
        drop_last=args.drop_last,
    )
    ensure_dir(output_dir)
    save_history_csv(os.path.join(output_dir, 'history.csv'), result['history'])
    plot_learning_curve(result['history'], os.path.join(output_dir, f'{args.experiment}_learning_curve.png'), args.experiment)
    return result


def run_cnn_rgft(args):
    if args.baseline_checkpoint is None or args.baseline_clean_val_acc is None:
        raise ValueError('RGFT needs --baseline-checkpoint and --baseline-clean-val-acc')

    train_images, train_labels, valid_images, valid_labels, _, _ = load_data(args)
    perturbation = maybe_make_perturbation(args)
    lr = args.lr if args.lr is not None else default_lr(args.experiment)
    if args.resume_checkpoint is not None:
        model = load_model_from_checkpoint('cnn', args.resume_checkpoint, weight_decay=args.weight_decay, weight_decay_lambda=args.weight_decay_lambda, freeze_policy=args.freeze_policy)
        existing_history = load_history_csv(args.resume_history) if args.resume_history is not None else []
        initial_best_state = infer_rgft_best_state(existing_history, args.baseline_clean_val_acc, clean_drop_limit=0.01)
        start_epoch = existing_history[-1]['epoch'] if existing_history else 0
        if args.resume_to_total_epochs is None:
            raise ValueError('Need --resume-to-total-epochs when resuming.')
        additional_epochs = args.resume_to_total_epochs - start_epoch
        if additional_epochs <= 0:
            raise ValueError('resume_to_total_epochs must be greater than the last recorded epoch.')
    else:
        model = load_model_from_checkpoint('cnn', args.baseline_checkpoint, weight_decay=args.weight_decay, weight_decay_lambda=args.weight_decay_lambda, freeze_policy=args.freeze_policy)
        existing_history = []
        initial_best_state = None
        start_epoch = 0
        additional_epochs = args.max_finetune_epochs
    optimizer = build_optimizer(args.optimizer, model, lr, momentum=args.momentum)

    output_dir = os.path.join(args.output_dir, args.experiment)
    checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.experiment}_best_model.pickle')
    progress_path = os.path.join(args.output_dir, 'live_progress.csv')
    result = train_model(
        model=model,
        model_kind='cnn',
        optimizer=optimizer,
        train_images=train_images,
        train_labels=train_labels,
        valid_images=valid_images,
        valid_labels=valid_labels,
        batch_size=args.batch_size,
        max_epochs=additional_epochs,
        checkpoint_path=checkpoint_path,
        selection_mode='rgft',
        target_perturbation=perturbation,
        clean_mix_ratio=args.clean_mix_ratio,
        baseline_clean_val_acc=args.baseline_clean_val_acc,
        clean_drop_limit=0.01,
        seed=args.seed,
        eval_seed=args.eval_seed,
        epoch_callback=make_epoch_callback(
            os.path.join(output_dir, 'history.csv'),
            os.path.join(output_dir, f'{args.experiment}_learning_curve.png'),
            args.experiment,
            progress_path,
            args.experiment,
        ),
        initial_history=existing_history,
        initial_best_state=initial_best_state,
        start_epoch=start_epoch,
        drop_last=args.drop_last,
    )
    ensure_dir(output_dir)
    save_history_csv(os.path.join(output_dir, 'history.csv'), result['history'])
    plot_learning_curve(result['history'], os.path.join(output_dir, f'{args.experiment}_learning_curve.png'), args.experiment)
    return result


def main():
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    ensure_dir(args.output_dir)
    ensure_dir(args.checkpoint_dir)

    if args.experiment == 'smoke':
        run_smoke(args)
    elif args.experiment in {'mlp_clean', 'cnn_clean'}:
        run_clean_experiment(args)
    elif args.experiment == 'cnn_augscratch':
        run_cnn_augscratch(args)
    elif args.experiment == 'cnn_rgft':
        run_cnn_rgft(args)


if __name__ == '__main__':
    main()
