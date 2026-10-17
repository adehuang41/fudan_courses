import argparse
import csv
import os


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_parser():
    parser = argparse.ArgumentParser(description='Generate report markdown skeletons from experiment outputs.')
    parser.add_argument('--output-dir', required=True)
    return parser


def write_part_b(output_dir):
    part_b_dir = os.path.join(output_dir, 'part_b')
    model_rows = read_csv_rows(os.path.join(part_b_dir, 'model_summary.csv'))
    lines = [
        '# Part B Analysis',
        '',
        '## Experimental Setup',
        '',
        'This part compares `MLP-clean` and `CNN-clean` under the same train/validation split, the same preprocessing, the same batch size, and the same epoch budget. Learning rates may differ only when needed for training stability, and any such adjustment should be reported explicitly in the final paper.',
        '',
        '## Key Results',
        '',
    ]
    if model_rows:
        for row in model_rows:
            lines.append(
                f"- {row['model_name']}: params={row['num_trainable_parameters']}, "
                f"train_acc={row['final_train_accuracy']}, val_acc={row['final_validation_accuracy']}, "
                f"test_acc={row['final_test_accuracy']}"
            )
    else:
        lines.append('- Fill in the final train/validation/test results after running the full experiment.')
    lines.extend([
        '',
        '## Interpretation',
        '',
        '1. Explain whether CNN outperforms MLP on clean MNIST under comparable training settings.',
        '2. Explain that flattening in MLP removes explicit 2D neighborhood structure, while CNN keeps local receptive fields.',
        '3. Explain why weight sharing can improve parameter efficiency for repeated local stroke patterns.',
        '4. Explain why convolution is translation-equivariant and why pooling/spatial aggregation may improve tolerance to small local translations.',
        '5. Explain that deeper convolutional layers can combine low-level strokes into higher-level digit parts.',
        '6. Keep the discussion faithful to the observed results; do not overclaim if the gain is small.',
        '',
        '## Limitations',
        '',
        '- A standard CNN is not automatically rotation-invariant.',
        '- Observed gains depend on model capacity, training budget, learning rate, and implementation quality.',
        '- MNIST is relatively simple, so a reasonable MLP can already perform strongly.',
        '',
        '## Figures To Reference',
        '',
        '- `mlp_learning_curve.png`',
        '- `cnn_learning_curve.png`',
        '- `mlp_vs_cnn_learning_curve.png`',
        '- `mlp_first_layer_weights.png`',
        '- `cnn_first_layer_kernels.png`',
    ])
    with open(os.path.join(part_b_dir, 'part_b_analysis.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def write_part_c(output_dir):
    part_c_dir = os.path.join(output_dir, 'part_c')
    target_rows = read_csv_rows(os.path.join(part_c_dir, 'selected_target.csv'))
    overall_rows = read_csv_rows(os.path.join(part_c_dir, 'test_overall_results.csv'))
    economy_rows = read_csv_rows(os.path.join(part_c_dir, 'economy_comparison.csv'))

    target_text = 'TBD'
    if target_rows:
        target_text = f"{target_rows[0]['target_type']} @ {target_rows[0]['target_severity']}"

    lines = [
        '# Part C Analysis',
        '',
        '## Research Question',
        '',
        'This part studies `Robustness-Guided Data Augmentation`.',
        'The workflow is: train `CNN-clean`, diagnose robustness on the validation set, select a single target perturbation by overall validation accuracy drop, then compare `CNN-AugScratch-target` and `CNN-RGFT-target`.',
        '',
        f'Chosen target perturbation: `{target_text}`.',
        '',
        '## Main Questions',
        '',
        '1. Which perturbation is the main weakness of `CNN-clean`?',
        '2. Can target augmentation improve robustness to that weakness without unacceptable clean accuracy loss?',
        '3. Does RGFT achieve a favorable robustness-cost trade-off compared with AugScratch?',
        '',
        '## Final Test Summary',
        '',
    ]
    if overall_rows:
        grouped = {}
        for row in overall_rows:
            grouped.setdefault(row['model_name'], []).append(row)
        for model_name, rows in grouped.items():
            row_map = {row['setting']: row for row in rows}
            clean = row_map.get('clean', {})
            target = row_map.get('target', {})
            neighbor = row_map.get('same_type_neighbor', {})
            lines.append(
                f"- {model_name}: clean={clean.get('accuracy', 'NA')}, "
                f"target={target.get('accuracy', 'NA')}, target_drop={target.get('accuracy_drop', 'NA')}, "
                f"neighbor={neighbor.get('accuracy', 'NA')}"
            )
    else:
        lines.append('- Fill in the final test summary after the full experiment finishes.')

    lines.extend([
        '',
        '## Cost Comparison',
        '',
    ])
    if economy_rows:
        for row in economy_rows:
            lines.append(
                f"- {row['model_name']}: source={row['source_checkpoint']}, "
                f"extra_epochs={row['extra_epochs']}, time_sec={row['time_sec']}"
            )
    else:
        lines.append('- Fill in cost comparison after the full experiment finishes.')

    lines.extend([
        '',
        '## Interpretation',
        '',
        '- Explain why robustness diagnosis and target augmentation are studied together.',
        '- Explain whether `CNN-AugScratch-target` improves target robustness.',
        '- Explain whether `CNN-RGFT-target` gives a useful robustness gain at lower additional cost.',
        '- Explain whether the improvement remains local to the target perturbation or transfers to neighboring/non-target settings.',
        '- Explain whether clean accuracy is preserved or sacrificed, and quantify the trade-off honestly.',
        '',
        '## Target-Focused Error Analysis',
        '',
        '- Use the target-perturbation confusion matrices to identify which confused pairs are reduced most.',
        '- Use `repaired_examples.png` and `new_errors.png` to discuss what kinds of mistakes were fixed and what new errors appeared.',
        '- Do not claim general robustness if the gains are mostly target-specific.',
        '',
        '## Figures To Reference',
        '',
        '- `validation_robustness_overall.csv`',
        '- `robustness_curves.png`',
        '- `target_only_dense_curve.png`',
        '- target confusion matrices',
        '- `confusion_shift.csv`',
        '- `repaired_examples.png`',
        '- `new_errors.png`',
    ])
    with open(os.path.join(part_c_dir, 'part_c_analysis.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    args = build_parser().parse_args()
    ensure_dir(os.path.join(args.output_dir, 'part_b'))
    ensure_dir(os.path.join(args.output_dir, 'part_c'))
    write_part_b(args.output_dir)
    write_part_c(args.output_dir)


if __name__ == '__main__':
    main()
