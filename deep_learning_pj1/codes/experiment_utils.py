import csv
import gzip
import os
import shutil
import time
from struct import unpack

import numpy as np

import mynn as nn


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_mnist(images_path, labels_path, flatten=False, normalize=True):
    with gzip.open(images_path, 'rb') as f:
        _, num, rows, cols = unpack('>4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols).astype(np.float32)

    with gzip.open(labels_path, 'rb') as f:
        _, num = unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    if normalize:
        images = images / 255.0

    if flatten:
        images = images.reshape(images.shape[0], -1)
    else:
        images = images[:, None, :, :]
    return images, labels


def create_or_load_split(num_samples, val_size, seed, split_path):
    if os.path.exists(split_path):
        with open(split_path, 'rb') as f:
            return np.array(np.load(f, allow_pickle=False))

    rng = np.random.default_rng(seed)
    indices = rng.permutation(np.arange(num_samples)).astype(np.int64)
    ensure_dir(os.path.dirname(split_path) or '.')
    with open(split_path, 'wb') as f:
        np.save(f, indices, allow_pickle=False)
    return indices


def train_val_split(images, labels, val_size, seed, split_path):
    indices = create_or_load_split(images.shape[0], val_size, seed, split_path)
    images = images[indices]
    labels = labels[indices]
    valid_images = images[:val_size]
    valid_labels = labels[:val_size]
    train_images = images[val_size:]
    train_labels = labels[val_size:]
    return train_images, train_labels, valid_images, valid_labels, indices


def format_inputs(images, model_kind):
    if model_kind == 'mlp':
        return images.reshape(images.shape[0], -1)
    if model_kind == 'cnn':
        return images
    raise ValueError(f'Unknown model kind: {model_kind}')


def count_trainable_params(model):
    total = 0
    for layer in model.layers:
        if layer.optimizable:
            total += int(np.prod(layer.params['W'].shape))
            total += int(np.prod(layer.params['b'].shape))
    return total


def get_model_summary(model_name, model, optimizer_name, batch_size, learning_rate, epochs, final_train_acc, final_val_acc, final_test_acc, architecture):
    return {
        'model_name': model_name,
        'architecture': architecture,
        'num_trainable_parameters': count_trainable_params(model),
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'optimizer': optimizer_name,
        'final_train_accuracy': final_train_acc,
        'final_validation_accuracy': final_val_acc,
        'final_test_accuracy': final_test_acc,
    }


def save_csv(path, rows, fieldnames=None):
    ensure_dir(os.path.dirname(path) or '.')
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def confusion_matrix_from_logits(logits, labels, num_classes=10):
    preds = np.argmax(logits, axis=1)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels, preds):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def matrix_to_rows(matrix):
    rows = []
    for i in range(matrix.shape[0]):
        row = {'row': i}
        for j in range(matrix.shape[1]):
            row[f'col_{j}'] = int(matrix[i, j])
        rows.append(row)
    return rows


def per_class_accuracy_from_logits(logits, labels, num_classes=10):
    preds = np.argmax(logits, axis=1)
    rows = []
    for class_id in range(num_classes):
        mask = labels == class_id
        total = int(np.sum(mask))
        correct = int(np.sum(preds[mask] == class_id)) if total > 0 else 0
        rows.append({
            'class_id': class_id,
            'num_samples': total,
            'num_correct': correct,
            'accuracy': (correct / total) if total > 0 else 0.0,
        })
    return rows


def build_model(model_kind, weight_decay=False, weight_decay_lambda=1e-4, freeze_policy=None):
    if model_kind == 'mlp':
        lambda_list = [weight_decay_lambda, weight_decay_lambda] if weight_decay else None
        return nn.models.Model_MLP([28 * 28, 256, 10], 'ReLU', lambda_list=lambda_list)
    if model_kind == 'cnn':
        model = nn.models.Model_CNN()
        if weight_decay:
            for layer in model.layers:
                if layer.optimizable:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = weight_decay_lambda
        if freeze_policy is not None:
            model.set_trainable_policy(freeze_policy)
        return model
    raise ValueError(f'Unknown model kind: {model_kind}')


def load_model_from_checkpoint(model_kind, checkpoint_path, weight_decay=False, weight_decay_lambda=1e-4, freeze_policy=None):
    model = build_model(model_kind, weight_decay=weight_decay, weight_decay_lambda=weight_decay_lambda, freeze_policy=freeze_policy)
    model.load_model(checkpoint_path)
    if model_kind == 'cnn' and freeze_policy is not None:
        model.set_trainable_policy(freeze_policy)
    return model


def subset_by_indices(images, labels, indices):
    return images[indices], labels[indices]


def build_optimizer(optimizer_name, model, learning_rate, momentum=0.9):
    if optimizer_name == 'sgd':
        return nn.optimizer.SGD(init_lr=learning_rate, model=model)
    if optimizer_name == 'momentum':
        return nn.optimizer.MomentGD(init_lr=learning_rate, model=model, mu=momentum)
    raise ValueError(f'Unknown optimizer: {optimizer_name}')


def evaluate_model(model, model_kind, images, labels, loss_fn):
    model_inputs = format_inputs(images, model_kind)
    logits = model(model_inputs)
    loss = loss_fn(logits, labels)
    acc = nn.metric.accuracy(logits, labels)
    return {
        'loss': float(loss),
        'acc': float(acc),
        'logits': logits,
    }


def translation_transform(images, max_shift, rng):
    transformed = np.zeros_like(images)
    for idx in range(images.shape[0]):
        dx = int(rng.integers(-max_shift, max_shift + 1))
        dy = int(rng.integers(-max_shift, max_shift + 1))
        src_x0 = max(0, -dx)
        src_x1 = min(images.shape[-2], images.shape[-2] - dx)
        src_y0 = max(0, -dy)
        src_y1 = min(images.shape[-1], images.shape[-1] - dy)
        dst_x0 = max(0, dx)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y0 = max(0, dy)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        transformed[idx, :, dst_x0:dst_x1, dst_y0:dst_y1] = images[idx, :, src_x0:src_x1, src_y0:src_y1]
    return transformed


def translate_by_offset(images, dx, dy):
    transformed = np.zeros_like(images)
    src_x0 = max(0, -dx)
    src_x1 = min(images.shape[-2], images.shape[-2] - dx)
    src_y0 = max(0, -dy)
    src_y1 = min(images.shape[-1], images.shape[-1] - dy)
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    transformed[:, :, dst_x0:dst_x1, dst_y0:dst_y1] = images[:, :, src_x0:src_x1, src_y0:src_y1]
    return transformed


def gaussian_noise_transform(images, sigma, rng):
    noise = rng.normal(loc=0.0, scale=sigma, size=images.shape)
    return np.clip(images + noise, 0.0, 1.0).astype(images.dtype)


def _rotation_source_indices(height, width, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    cos_v = np.cos(angle_rad)
    sin_v = np.sin(angle_rad)
    cx = (height - 1) / 2.0
    cy = (width - 1) / 2.0

    yy, xx = np.meshgrid(np.arange(width), np.arange(height))
    x_shift = xx - cx
    y_shift = yy - cy

    src_x = cos_v * x_shift + sin_v * y_shift + cx
    src_y = -sin_v * x_shift + cos_v * y_shift + cy
    src_x = np.rint(src_x).astype(np.int64)
    src_y = np.rint(src_y).astype(np.int64)
    valid = (src_x >= 0) & (src_x < height) & (src_y >= 0) & (src_y < width)
    return src_x, src_y, valid


def rotation_transform(images, angle_deg):
    batch_size, channels, height, width = images.shape
    src_x, src_y, valid = _rotation_source_indices(height, width, angle_deg)
    transformed = np.zeros_like(images)
    for idx in range(batch_size):
        for channel in range(channels):
            rotated = np.zeros((height, width), dtype=images.dtype)
            rotated[valid] = images[idx, channel, src_x[valid], src_y[valid]]
            transformed[idx, channel] = rotated
    return transformed


def apply_perturbation(images, perturbation, rng=None):
    if perturbation is None:
        return images.copy()

    perturbation_type = perturbation['type']
    severity = perturbation['severity']
    rng = np.random.default_rng() if rng is None else rng

    if perturbation_type == 'translation':
        return translation_transform(images, severity, rng)
    if perturbation_type == 'noise':
        return gaussian_noise_transform(images, severity, rng)
    if perturbation_type == 'rotation':
        return rotation_transform(images, severity)
    raise ValueError(f'Unknown perturbation type: {perturbation_type}')


def make_mixed_batch(images, perturbation, clean_ratio, rng):
    if perturbation is None:
        return images.copy()

    batch_size = images.shape[0]
    num_clean = int(round(batch_size * clean_ratio))
    indices = rng.permutation(batch_size)
    clean_indices = indices[:num_clean]
    perturbed_indices = indices[num_clean:]

    mixed = images.copy()
    if perturbed_indices.size > 0:
        mixed[perturbed_indices] = apply_perturbation(images[perturbed_indices], perturbation, rng)
    if clean_indices.size > 0:
        mixed[clean_indices] = images[clean_indices]
    return mixed


def make_family_mixed_batch(images, perturbation_grid, clean_ratio, rng):
    if not perturbation_grid:
        return images.copy()

    family_to_severities = {}
    for perturbation in perturbation_grid:
        family_to_severities.setdefault(perturbation['type'], []).append(perturbation['severity'])
    families = sorted(family_to_severities.keys())

    batch_size = images.shape[0]
    num_clean = int(round(batch_size * clean_ratio))
    indices = rng.permutation(batch_size)
    clean_indices = indices[:num_clean]
    perturbed_indices = indices[num_clean:]

    mixed = images.copy()
    if perturbed_indices.size > 0:
        sampled_families = rng.integers(0, len(families), size=perturbed_indices.shape[0])
        for family_idx, family_name in enumerate(families):
            local_mask = sampled_families == family_idx
            if not np.any(local_mask):
                continue
            severities = family_to_severities[family_name]
            local_indices = perturbed_indices[local_mask]
            sampled_severity_ids = rng.integers(0, len(severities), size=local_indices.shape[0])
            for severity_id, severity in enumerate(severities):
                severity_mask = sampled_severity_ids == severity_id
                if not np.any(severity_mask):
                    continue
                severity_indices = local_indices[severity_mask]
                mixed[severity_indices] = apply_perturbation(
                    images[severity_indices],
                    {'type': family_name, 'severity': severity},
                    rng,
                )
    if clean_indices.size > 0:
        mixed[clean_indices] = images[clean_indices]
    return mixed


def iterate_minibatches(images, labels, batch_size, rng, drop_last=False):
    indices = rng.permutation(images.shape[0])
    for start in range(0, images.shape[0], batch_size):
        batch_idx = indices[start:start + batch_size]
        if drop_last and batch_idx.shape[0] < batch_size:
            continue
        yield images[batch_idx], labels[batch_idx]


def save_history_csv(path, history):
    save_csv(path, history, fieldnames=[
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
    ])


def _clean_selector(metrics, best_state):
    score = metrics['clean_val_acc']
    is_better = best_state is None or score > best_state['score']
    return is_better, score, 'clean_val_acc'


def _target_selector(metrics, best_state):
    score = metrics['target_val_acc']
    is_better = best_state is None or score > best_state['score']
    return is_better, score, 'target_val_acc'


def _mixed_score_selector(metrics, best_state):
    score = metrics['target_val_acc']
    is_better = best_state is None or score > best_state['score']
    return is_better, score, 'mixed_val_score'


def _rgft_selector(metrics, best_state, baseline_clean_val_acc, clean_drop_limit):
    clean_drop = baseline_clean_val_acc - metrics['clean_val_acc']
    satisfies = clean_drop <= clean_drop_limit
    if not satisfies:
        return False, None, f'clean_drop_exceeds_limit:{clean_drop:.6f}'

    score = metrics['target_val_acc']
    if best_state is None:
        return True, score, 'target_val_acc_with_clean_constraint'

    if score > best_state['score']:
        return True, score, 'target_val_acc_with_clean_constraint'
    if np.isclose(score, best_state['score']):
        if clean_drop < best_state['clean_drop']:
            return True, score, 'smaller_clean_drop'
        if np.isclose(clean_drop, best_state['clean_drop']) and metrics['epoch'] < best_state['epoch']:
            return True, score, 'earlier_epoch_tie_break'
    return False, score, 'not_better_than_current_best'


def train_model(
    model,
    model_kind,
    optimizer,
    train_images,
    train_labels,
    valid_images,
    valid_labels,
    batch_size,
    max_epochs,
    checkpoint_path,
    selection_mode='clean',
    target_perturbation=None,
    mixed_train_grid=None,
    selection_score_perturbations=None,
    clean_mix_ratio=0.5,
    baseline_clean_val_acc=None,
    clean_drop_limit=0.01,
    seed=309,
    eval_seed=2026,
    epoch_callback=None,
    initial_history=None,
    initial_best_state=None,
    start_epoch=0,
    drop_last=False,
):
    rng = np.random.default_rng(seed)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    history = [] if initial_history is None else list(initial_history)
    best_state = initial_best_state
    fallback_state = None
    fallback_checkpoint_path = checkpoint_path + '.fallback'

    for epoch in range(start_epoch + 1, start_epoch + max_epochs + 1):
        epoch_start = time.time()
        batch_losses = []
        batch_accs = []

        for batch_images, batch_labels in iterate_minibatches(train_images, train_labels, batch_size, rng, drop_last=drop_last):
            train_batch = batch_images
            if mixed_train_grid is not None:
                train_batch = make_family_mixed_batch(batch_images, mixed_train_grid, clean_mix_ratio, rng)
            elif target_perturbation is not None:
                train_batch = make_mixed_batch(batch_images, target_perturbation, clean_mix_ratio, rng)

            logits = model(format_inputs(train_batch, model_kind))
            loss = loss_fn(logits, batch_labels)
            loss_fn.backward()
            optimizer.step()

            batch_losses.append(float(loss))
            batch_accs.append(float(nn.metric.accuracy(logits, batch_labels)))

        clean_val_metrics = evaluate_model(model, model_kind, valid_images, valid_labels, loss_fn)
        target_val_acc = ''
        if selection_score_perturbations is not None:
            selected_accs = [clean_val_metrics['acc']]
            for perturbation in selection_score_perturbations:
                fixed_rng = np.random.default_rng(eval_seed)
                perturbed_valid = apply_perturbation(valid_images, perturbation, fixed_rng)
                selected_metrics = evaluate_model(model, model_kind, perturbed_valid, valid_labels, loss_fn)
                selected_accs.append(selected_metrics['acc'])
            target_val_acc = float(np.mean(selected_accs))
        elif target_perturbation is not None:
            fixed_rng = np.random.default_rng(eval_seed)
            perturbed_valid = apply_perturbation(valid_images, target_perturbation, fixed_rng)
            target_val_metrics = evaluate_model(model, model_kind, perturbed_valid, valid_labels, loss_fn)
            target_val_acc = target_val_metrics['acc']

        epoch_metrics = {
            'epoch': epoch,
            'train_loss': float(np.mean(batch_losses)),
            'train_acc': float(np.mean(batch_accs)),
            'clean_val_loss': clean_val_metrics['loss'],
            'clean_val_acc': clean_val_metrics['acc'],
            'target_val_acc': target_val_acc,
            'epoch_time_sec': float(time.time() - epoch_start),
            'learning_rate': float(optimizer.init_lr),
            'selected_as_best': False,
            'selection_reason': '',
        }

        if selection_mode == 'clean':
            is_better, score, reason = _clean_selector(epoch_metrics, best_state)
        elif selection_mode == 'target':
            is_better, score, reason = _target_selector(epoch_metrics, best_state)
        elif selection_mode == 'mixed_score':
            is_better, score, reason = _mixed_score_selector(epoch_metrics, best_state)
        elif selection_mode == 'rgft':
            is_better, score, reason = _rgft_selector(epoch_metrics, best_state, baseline_clean_val_acc, clean_drop_limit)
        else:
            raise ValueError(f'Unknown selection mode: {selection_mode}')

        if selection_mode == 'rgft':
            clean_drop = baseline_clean_val_acc - epoch_metrics['clean_val_acc']
            fallback_better = False
            if fallback_state is None:
                fallback_better = True
            elif epoch_metrics['target_val_acc'] > fallback_state['score']:
                fallback_better = True
            elif np.isclose(epoch_metrics['target_val_acc'], fallback_state['score']):
                if clean_drop < fallback_state['clean_drop']:
                    fallback_better = True
                elif np.isclose(clean_drop, fallback_state['clean_drop']) and epoch < fallback_state['epoch']:
                    fallback_better = True
            if fallback_better:
                ensure_dir(os.path.dirname(fallback_checkpoint_path) or '.')
                model.save_model(fallback_checkpoint_path)
                fallback_state = {
                    'epoch': epoch,
                    'score': epoch_metrics['target_val_acc'],
                    'clean_drop': clean_drop,
                    'metrics': epoch_metrics.copy(),
                    'reason': 'fallback_target_then_clean_drop',
                }

        if is_better:
            ensure_dir(os.path.dirname(checkpoint_path) or '.')
            model.save_model(checkpoint_path)
            best_state = {
                'epoch': epoch,
                'score': score,
                'clean_drop': (baseline_clean_val_acc - epoch_metrics['clean_val_acc']) if baseline_clean_val_acc is not None else 0.0,
                'metrics': epoch_metrics.copy(),
                'reason': reason,
            }
            epoch_metrics['selected_as_best'] = True
            epoch_metrics['selection_reason'] = reason
        else:
            epoch_metrics['selection_reason'] = reason

        history.append(epoch_metrics)
        if epoch_callback is not None:
            epoch_callback(history, epoch_metrics, best_state)

    if selection_mode == 'rgft' and best_state is None and fallback_state is not None:
        shutil.copyfile(fallback_checkpoint_path, checkpoint_path)
        fallback_state['metrics']['selection_reason'] = 'fallback_no_epoch_satisfied_clean_constraint'
        best_state = fallback_state

    return {
        'history': history,
        'best_state': best_state,
        'loss_fn': loss_fn,
    }


def evaluate_under_perturbation(model, model_kind, images, labels, perturbation, seed=309):
    rng = np.random.default_rng(seed)
    perturbed_images = apply_perturbation(images, perturbation, rng)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    metrics = evaluate_model(model, model_kind, perturbed_images, labels, loss_fn)
    return metrics, perturbed_images


def robustness_diagnosis(model, model_kind, images, labels, perturbation_grid, clean_seed=2026):
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    clean_metrics = evaluate_model(model, model_kind, images, labels, loss_fn)
    rows = []
    per_class_rows = []

    for perturbation in perturbation_grid:
        metrics, _ = evaluate_under_perturbation(model, model_kind, images, labels, perturbation, seed=clean_seed)
        accuracy_drop = clean_metrics['acc'] - metrics['acc']
        rows.append({
            'perturbation_type': perturbation['type'],
            'severity': perturbation['severity'],
            'clean_accuracy': clean_metrics['acc'],
            'perturbed_accuracy': metrics['acc'],
            'accuracy_drop': accuracy_drop,
        })

        clean_per_class = per_class_accuracy_from_logits(clean_metrics['logits'], labels)
        perturbed_per_class = per_class_accuracy_from_logits(metrics['logits'], labels)
        for clean_row, perturbed_row in zip(clean_per_class, perturbed_per_class):
            per_class_rows.append({
                'perturbation_type': perturbation['type'],
                'severity': perturbation['severity'],
                'class_id': clean_row['class_id'],
                'clean_accuracy': clean_row['accuracy'],
                'perturbed_accuracy': perturbed_row['accuracy'],
                'accuracy_drop': clean_row['accuracy'] - perturbed_row['accuracy'],
            })

    return rows, per_class_rows


def select_target_perturbation(diagnosis_rows):
    best = max(diagnosis_rows, key=lambda row: row['accuracy_drop'])
    severity = best['severity']
    if best['perturbation_type'] in {'translation', 'rotation'}:
        severity = int(severity)
    return {
        'type': best['perturbation_type'],
        'severity': severity,
    }


def same_type_neighbor(target):
    grids = {
        'noise': [0.05, 0.10, 0.20],
        'translation': [1, 2, 3],
        'rotation': [5, 10, 15],
    }
    values = grids[target['type']]
    current = target['severity']
    idx = values.index(current)
    if idx > 0:
        neighbor = values[idx - 1]
    else:
        neighbor = values[min(idx + 1, len(values) - 1)]
    return {'type': target['type'], 'severity': neighbor}


def non_target_representatives(target):
    fixed = {
        'noise': {'type': 'noise', 'severity': 0.10},
        'translation': {'type': 'translation', 'severity': 2},
        'rotation': {'type': 'rotation', 'severity': 10},
    }
    return [fixed[key] for key in ['noise', 'translation', 'rotation'] if key != target['type']]


def dense_target_strengths(target):
    if target['type'] == 'noise':
        return [0.00, 0.05, 0.10, 0.15, 0.20]
    if target['type'] == 'translation':
        return [0, 1, 2, 3]
    if target['type'] == 'rotation':
        return [0, 5, 10, 15]
    raise ValueError(target['type'])
