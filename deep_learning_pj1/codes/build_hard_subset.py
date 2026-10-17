import argparse
import csv
import os

import numpy as np

from experiment_utils import (
    ensure_dir,
    format_inputs,
    load_mnist,
    load_model_from_checkpoint,
    train_val_split,
    translate_by_offset,
)
import mynn as nn


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset', 'MNIST')
TRAIN_IMAGES_PATH = os.path.join(DATASET_DIR, 'train-images-idx3-ubyte.gz')
TRAIN_LABELS_PATH = os.path.join(DATASET_DIR, 'train-labels-idx1-ubyte.gz')


def build_parser():
    parser = argparse.ArgumentParser(description='Build hard and random subsets for extended RGFT experiments.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--subset-fraction', type=float, default=0.10)
    parser.add_argument('--target-type', required=True, choices=['translation'])
    parser.add_argument('--target-severity', required=True, type=int)
    parser.add_argument('--val-size', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--batch-size', type=int, default=2048)
    return parser


def cross_entropy_per_sample(logits, labels):
    probs = nn.op.softmax(logits)
    probs = np.clip(probs[np.arange(labels.shape[0]), labels], 1e-12, 1.0)
    return -np.log(probs)


def save_scores_csv(path, rows):
    fieldnames = [
        'index',
        'label',
        'clean_correct',
        'hard_score',
        'target_loss',
        'selection_rank',
        'selection_reason',
    ]
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_progress(path, row):
    fieldnames = ['stage', 'direction', 'batch_start', 'batch_end']
    ensure_dir(os.path.dirname(path) or '.')
    exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def batched_logits(model, images, batch_size):
    outputs = []
    for start in range(0, images.shape[0], batch_size):
        outputs.append(model(format_inputs(images[start:start + batch_size], 'cnn')))
    return np.concatenate(outputs, axis=0)


def main():
    args = build_parser().parse_args()
    if args.target_type != 'translation':
        raise NotImplementedError('Current hard-subset builder is intentionally scoped to translation target only.')

    train_images, train_labels = load_mnist(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, flatten=False, normalize=True)
    train_images, train_labels, _, _, indices = train_val_split(
        train_images,
        train_labels,
        val_size=args.val_size,
        seed=309,
        split_path=args.split_path,
    )
    original_train_indices = indices[args.val_size:]
    progress_path = os.path.join(args.output_dir, 'subset_build_progress.csv')

    model = load_model_from_checkpoint('cnn', args.checkpoint)
    append_progress(progress_path, {'stage': 'clean_eval', 'direction': 'clean', 'batch_start': 0, 'batch_end': 0})
    clean_logits = batched_logits(model, train_images, args.batch_size)
    clean_preds = np.argmax(clean_logits, axis=1)
    clean_correct = clean_preds == train_labels

    k = int(args.target_severity)
    directions = [
        ('left', 0, -k),
        ('right', 0, k),
        ('up', -k, 0),
        ('down', k, 0),
    ]

    wrong_counts = np.zeros(train_images.shape[0], dtype=np.int64)
    loss_sum = np.zeros(train_images.shape[0], dtype=np.float64)
    for direction_name, dx, dy in directions:
        shifted = translate_by_offset(train_images, dx, dy)
        direction_logits = []
        for start in range(0, shifted.shape[0], args.batch_size):
            end = min(start + args.batch_size, shifted.shape[0])
            append_progress(progress_path, {
                'stage': 'target_eval',
                'direction': direction_name,
                'batch_start': start,
                'batch_end': end,
            })
            direction_logits.append(model(format_inputs(shifted[start:end], 'cnn')))
        logits = np.concatenate(direction_logits, axis=0)
        preds = np.argmax(logits, axis=1)
        wrong_counts += (preds != train_labels).astype(np.int64)
        loss_sum += cross_entropy_per_sample(logits, train_labels)

    hard_score = wrong_counts / len(directions)
    avg_target_loss = loss_sum / len(directions)

    quota = int(round(train_images.shape[0] * args.subset_fraction))
    quota = max(quota, 1)

    candidate_indices = np.where(clean_correct & (hard_score > 0))[0]
    sorted_candidates = sorted(
        candidate_indices.tolist(),
        key=lambda idx: (-hard_score[idx], -avg_target_loss[idx], int(idx)),
    )

    selected_hard = []
    for idx in sorted_candidates[:quota]:
        selected_hard.append((idx, 'strict_hard'))

    if len(selected_hard) < quota:
        remaining = [
            idx for idx in np.where(clean_correct)[0].tolist()
            if idx not in {item[0] for item in selected_hard}
        ]
        remaining = sorted(remaining, key=lambda idx: (-avg_target_loss[idx], int(idx)))
        for idx in remaining[:quota - len(selected_hard)]:
            selected_hard.append((idx, 'filled_by_target_loss'))

    selected_hard_indices = np.array([idx for idx, _ in selected_hard], dtype=np.int64)

    rng = np.random.default_rng(args.seed)
    clean_correct_pool = np.where(clean_correct)[0]
    random_subset_indices = rng.choice(clean_correct_pool, size=len(selected_hard_indices), replace=False)
    random_subset_indices = np.sort(random_subset_indices.astype(np.int64))

    rows = []
    for rank, (local_idx, reason) in enumerate(selected_hard, start=1):
        rows.append({
            'index': int(original_train_indices[local_idx]),
            'label': int(train_labels[local_idx]),
            'clean_correct': bool(clean_correct[local_idx]),
            'hard_score': float(hard_score[local_idx]),
            'target_loss': float(avg_target_loss[local_idx]),
            'selection_rank': rank,
            'selection_reason': reason,
        })

    ensure_dir(args.output_dir)
    np.save(os.path.join(args.output_dir, 'hard10_indices.npy'), original_train_indices[selected_hard_indices], allow_pickle=False)
    np.save(os.path.join(args.output_dir, 'random10_indices.npy'), original_train_indices[random_subset_indices], allow_pickle=False)
    save_scores_csv(os.path.join(args.output_dir, 'hard10_scores.csv'), rows)


if __name__ == '__main__':
    main()
