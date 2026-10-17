import argparse
import os

import numpy as np
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from experiment_utils import (
    ensure_dir,
    load_mnist,
    load_model_from_checkpoint,
    robustness_diagnosis,
    save_csv,
    train_val_split,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset', 'MNIST')
TRAIN_IMAGES_PATH = os.path.join(DATASET_DIR, 'train-images-idx3-ubyte.gz')
TRAIN_LABELS_PATH = os.path.join(DATASET_DIR, 'train-labels-idx1-ubyte.gz')
TEST_IMAGES_PATH = os.path.join(DATASET_DIR, 't10k-images-idx3-ubyte.gz')
TEST_LABELS_PATH = os.path.join(DATASET_DIR, 't10k-labels-idx1-ubyte.gz')


def build_parser():
    parser = argparse.ArgumentParser(description='Analysis helpers for the MNIST project.')
    parser.add_argument('--task', required=True, choices=['robustness_diagnosis'])
    parser.add_argument('--model-kind', required=True, choices=['mlp', 'cnn'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--split-path', default='outputs/splits/train_val_idx.npy')
    parser.add_argument('--seed', type=int, default=309)
    parser.add_argument('--eval-seed', type=int, default=2026)
    parser.add_argument('--val-size', type=int, default=10000)
    parser.add_argument('--output-dir', default='outputs/analysis')
    parser.add_argument('--subset', type=int, default=None)
    return parser


def default_perturbation_grid():
    return (
        [{'type': 'noise', 'severity': sigma} for sigma in [0.05, 0.10, 0.20]]
        + [{'type': 'translation', 'severity': shift} for shift in [1, 2, 3]]
        + [{'type': 'rotation', 'severity': angle} for angle in [5, 10, 15]]
    )


def load_split_images(split, split_path, val_size, seed, subset):
    if split == 'validation':
        train_images, train_labels = load_mnist(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, flatten=False, normalize=True)
        _, _, valid_images, valid_labels, _ = train_val_split(
            train_images,
            train_labels,
            val_size=val_size,
            seed=seed,
            split_path=split_path,
        )
        images, labels = valid_images, valid_labels
    else:
        images, labels = load_mnist(TEST_IMAGES_PATH, TEST_LABELS_PATH, flatten=False, normalize=True)

    if subset is not None:
        images = images[:subset]
        labels = labels[:subset]
    return images, labels


def maybe_plot_diagnosis(rows, save_path):
    if plt is None:
        return

    grouped = {}
    for row in rows:
        grouped.setdefault(row['perturbation_type'], []).append(row)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, perturbation_type in zip(axes, ['noise', 'translation', 'rotation']):
        subrows = sorted(grouped.get(perturbation_type, []), key=lambda item: item['severity'])
        ax.plot([item['severity'] for item in subrows], [item['accuracy_drop'] for item in subrows], marker='o')
        ax.set_title(perturbation_type)
        ax.set_xlabel('Severity')
        ax.set_ylabel('Accuracy drop')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def run_robustness_diagnosis(args):
    images, labels = load_split_images(args.split, args.split_path, args.val_size, args.seed, args.subset)
    model = load_model_from_checkpoint(args.model_kind, args.checkpoint)
    rows, per_class_rows = robustness_diagnosis(
        model=model,
        model_kind=args.model_kind,
        images=images,
        labels=labels,
        perturbation_grid=default_perturbation_grid(),
        clean_seed=args.eval_seed,
    )

    ensure_dir(args.output_dir)
    overall_path = os.path.join(args.output_dir, f'{args.split}_robustness_overall.csv')
    per_class_path = os.path.join(args.output_dir, f'{args.split}_robustness_per_class.csv')
    save_csv(overall_path, rows)
    save_csv(per_class_path, per_class_rows)
    maybe_plot_diagnosis(rows, os.path.join(args.output_dir, f'{args.split}_robustness_curve.png'))


def main():
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    if args.task == 'robustness_diagnosis':
        run_robustness_diagnosis(args)


if __name__ == '__main__':
    main()
