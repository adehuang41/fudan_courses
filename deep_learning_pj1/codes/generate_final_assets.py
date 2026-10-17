import argparse
import csv
import math
import os

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from experiment_utils import (
    confusion_matrix_from_logits,
    dense_target_strengths,
    ensure_dir,
    evaluate_model,
    evaluate_under_perturbation,
    load_mnist,
    load_model_from_checkpoint,
    matrix_to_rows,
    non_target_representatives,
    per_class_accuracy_from_logits,
    same_type_neighbor,
    save_csv,
)
import mynn as nn


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset", "MNIST")
TEST_IMAGES_PATH = os.path.join(DATASET_DIR, "t10k-images-idx3-ubyte.gz")
TEST_LABELS_PATH = os.path.join(DATASET_DIR, "t10k-labels-idx1-ubyte.gz")


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def total_epoch_time(history_path):
    rows = read_csv_rows(history_path)
    total = 0.0
    for row in rows:
        value = row.get("epoch_time_sec", "")
        if value != "":
            total += float(value)
    return total


def load_selected_target(path):
    rows = read_csv_rows(path)
    row = rows[0]
    severity = int(float(row["target_severity"])) if row["target_type"] in {"translation", "rotation"} else float(row["target_severity"])
    return {"type": row["target_type"], "severity": severity}


def plot_grid(images, save_path, title):
    if plt is None:
        return
    num_items = len(images)
    cols = int(math.ceil(math.sqrt(num_items)))
    rows = int(math.ceil(num_items / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    axes = axes.reshape(-1) if hasattr(axes, "reshape") else [axes]
    for ax, image in zip(axes, images):
        ax.imshow(image, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(images):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    ensure_dir(os.path.dirname(save_path) or ".")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_confusion_heatmap(matrix, save_path, title):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_robustness_curves(rows, save_path):
    if plt is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, ptype in zip(axes, ["noise", "translation", "rotation"]):
        for model_name in ["CNN-clean", "CNN-AugScratch-target", "CNN-AugScratch-mixed", "CNN-RGFT-target"]:
            sub = [row for row in rows if row["perturbation_type"] == ptype and row["model_name"] == model_name]
            sub = sorted(sub, key=lambda r: r["severity"])
            ax.plot([r["severity"] for r in sub], [r["accuracy"] for r in sub], marker="o", label=model_name)
        ax.set_title(ptype)
        ax.set_xlabel("Severity")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_dense_curve(rows, target, save_path):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for model_name in ["CNN-clean", "CNN-AugScratch-target", "CNN-AugScratch-mixed", "CNN-RGFT-target"]:
        sub = [row for row in rows if row["model_name"] == model_name]
        sub = sorted(sub, key=lambda r: r["severity"])
        ax.plot([r["severity"] for r in sub], [r["accuracy"] for r in sub], marker="o", label=model_name)
    ax.set_title(f"Dense Target Curve: {target['type']}")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_example_grid(example_indices, clean_images, perturbed_images, preds_a, preds_b, labels, title, save_path, max_items=10):
    if plt is None or len(example_indices) == 0:
        return
    rng = np.random.default_rng(2026)
    chosen = rng.choice(example_indices, size=min(max_items, len(example_indices)), replace=False)
    chosen = sorted(int(idx) for idx in chosen)
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.8))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.84, bottom=0.08, wspace=0.16, hspace=0.42)
    fig.suptitle(title, fontsize=15)
    for ax, sample_idx in zip(axes.flat, chosen):
        combo = np.concatenate([clean_images[sample_idx, 0], perturbed_images[sample_idx, 0]], axis=1)
        ax.imshow(combo, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"y={int(labels[sample_idx])}, {int(preds_a[sample_idx])}->{int(preds_b[sample_idx])}", fontsize=10, pad=4)
        ax.text(0.5, -0.10, "clean | pert.", transform=ax.transAxes, ha="center", va="top", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_edgecolor("#666666")
    for ax in axes.flat[len(chosen):]:
        ax.axis("off")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def generate_part_b_assets():
    mlp_ckpt = "full_mlp_clean_sgd_v1/checkpoints/mlp_clean_best_model.pickle"
    cnn_ckpt = "full_cnn_clean_sgd_v1/checkpoints/cnn_clean_best_model.pickle"
    part_b_dir = "project_outputs/part_b"
    ensure_dir(part_b_dir)

    mlp_model = load_model_from_checkpoint("mlp", mlp_ckpt)
    cnn_model = load_model_from_checkpoint("cnn", cnn_ckpt)

    mlp_weights = mlp_model.layers[0].params["W"].T
    mlp_images = [mlp_weights[idx].reshape(28, 28) for idx in range(min(16, mlp_weights.shape[0]))]
    plot_grid(mlp_images, os.path.join(part_b_dir, "mlp_first_layer_weights.png"), "MLP First-Layer Weights")

    kernels = cnn_model.layers[0].params["W"]
    kernel_images = [kernels[idx, 0] for idx in range(min(16, kernels.shape[0]))]
    plot_grid(kernel_images, os.path.join(part_b_dir, "cnn_first_layer_kernels.png"), "CNN First-Layer Kernels")


def generate_part_c_main_assets():
    part_c_dir = "project_outputs/part_c"
    ensure_dir(part_c_dir)
    target = load_selected_target(os.path.join(part_c_dir, "selected_target.csv"))

    test_images, test_labels = load_mnist(TEST_IMAGES_PATH, TEST_LABELS_PATH, flatten=False, normalize=True)
    models = {
        "MLP-clean": ("mlp", load_model_from_checkpoint("mlp", "full_mlp_clean_sgd_v1/checkpoints/mlp_clean_best_model.pickle")),
        "CNN-clean": ("cnn", load_model_from_checkpoint("cnn", "full_cnn_clean_sgd_v1/checkpoints/cnn_clean_best_model.pickle")),
        "CNN-AugScratch-target": ("cnn", load_model_from_checkpoint("cnn", "full_cnn_augscratch_target_v1/checkpoints/cnn_augscratch_best_model.pickle")),
        "CNN-AugScratch-mixed": ("cnn", load_model_from_checkpoint("cnn", "full_cnn_augscratch_mixed_v1/checkpoints/cnn_augscratch_mixed_best_model.pickle")),
        "CNN-RGFT-target": ("cnn", load_model_from_checkpoint("cnn", "full_cnn_rgft_target_v1/checkpoints/cnn_rgft_best_model.pickle")),
    }
    loss_fns = {name: nn.op.MultiCrossEntropyLoss(model=model, max_classes=10) for name, (_, model) in models.items()}

    eval_settings = [("clean", None), ("target", target), ("same_type_neighbor", same_type_neighbor(target))]
    eval_settings.extend([(f"non_target_{idx+1}", p) for idx, p in enumerate(non_target_representatives(target))])

    overall_rows = []
    per_class_rows = []
    curve_rows = []
    dense_rows = []
    target_metrics = {}
    target_images = {}

    for model_name, (model_kind, model) in models.items():
        clean_metrics = evaluate_model(model, model_kind, test_images, test_labels, loss_fns[model_name])
        clean_acc = clean_metrics["acc"]
        for setting_name, perturbation in eval_settings:
            if perturbation is None:
                metrics = clean_metrics
            else:
                metrics, _ = evaluate_under_perturbation(model, model_kind, test_images, test_labels, perturbation, seed=2026)
            overall_rows.append(
                {
                    "model_name": model_name,
                    "setting": setting_name,
                    "perturbation_type": "clean" if perturbation is None else perturbation["type"],
                    "severity": 0 if perturbation is None else perturbation["severity"],
                    "accuracy": metrics["acc"],
                    "accuracy_drop": clean_acc - metrics["acc"],
                }
            )
            for row in per_class_accuracy_from_logits(metrics["logits"], test_labels):
                per_class_rows.append(
                    {
                        "model_name": model_name,
                        "setting": setting_name,
                        "perturbation_type": "clean" if perturbation is None else perturbation["type"],
                        "severity": 0 if perturbation is None else perturbation["severity"],
                        **row,
                    }
                )

        if model_name != "MLP-clean":
            for perturbation in (
                [{"type": "noise", "severity": sigma} for sigma in [0.05, 0.10, 0.20]]
                + [{"type": "translation", "severity": shift} for shift in [1, 2, 3]]
                + [{"type": "rotation", "severity": angle} for angle in [5, 10, 15]]
            ):
                metrics, _ = evaluate_under_perturbation(model, model_kind, test_images, test_labels, perturbation, seed=2026)
                curve_rows.append(
                    {
                        "model_name": model_name,
                        "perturbation_type": perturbation["type"],
                        "severity": perturbation["severity"],
                        "accuracy": metrics["acc"],
                    }
                )
            for severity in dense_target_strengths(target):
                perturbation = {"type": target["type"], "severity": severity}
                metrics, _ = evaluate_under_perturbation(model, model_kind, test_images, test_labels, perturbation, seed=2026)
                dense_rows.append({"model_name": model_name, "severity": severity, "accuracy": metrics["acc"]})

            target_metric, perturbed = evaluate_under_perturbation(model, model_kind, test_images, test_labels, target, seed=2026)
            target_metrics[model_name] = target_metric
            target_images[model_name] = perturbed
            matrix = confusion_matrix_from_logits(target_metric["logits"], test_labels)
            csv_name = f"{model_name.lower().replace('-', '_')}_target_confusion_matrix.csv"
            png_name = f"{model_name.lower().replace('-', '_')}_target_confusion_matrix.png"
            save_csv(os.path.join(part_c_dir, csv_name), matrix_to_rows(matrix))
            plot_confusion_heatmap(matrix, os.path.join(part_c_dir, png_name), f"{model_name} Target Confusion")

    save_csv(os.path.join(part_c_dir, "test_overall_results.csv"), overall_rows)
    save_csv(os.path.join(part_c_dir, "test_per_class_results.csv"), per_class_rows)
    save_csv(os.path.join(part_c_dir, "accuracy_drop_results.csv"), overall_rows)
    save_csv(os.path.join(part_c_dir, "robustness_curves.csv"), curve_rows)
    plot_robustness_curves(curve_rows, os.path.join(part_c_dir, "robustness_curves.png"))
    save_csv(os.path.join(part_c_dir, "target_only_dense_curve.csv"), dense_rows)
    plot_dense_curve(dense_rows, target, os.path.join(part_c_dir, "target_only_dense_curve.png"))

    clean_preds = np.argmax(target_metrics["CNN-clean"]["logits"], axis=1)
    rgft_preds = np.argmax(target_metrics["CNN-RGFT-target"]["logits"], axis=1)

    clean_matrix = confusion_matrix_from_logits(target_metrics["CNN-clean"]["logits"], test_labels)
    rgft_matrix = confusion_matrix_from_logits(target_metrics["CNN-RGFT-target"]["logits"], test_labels)
    confusion_shift_rows = []
    for i in range(clean_matrix.shape[0]):
        for j in range(clean_matrix.shape[1]):
            if i == j:
                continue
            clean_count = int(clean_matrix[i, j])
            rgft_count = int(rgft_matrix[i, j])
            confusion_shift_rows.append(
                {
                    "true_class": i,
                    "pred_class": j,
                    "clean_count": clean_count,
                    "rgft_count": rgft_count,
                    "absolute_reduction": clean_count - rgft_count,
                    "relative_reduction": ((clean_count - rgft_count) / clean_count) if clean_count > 0 else 0.0,
                }
            )
    confusion_shift_rows.sort(key=lambda row: row["absolute_reduction"], reverse=True)
    save_csv(os.path.join(part_c_dir, "confusion_shift.csv"), confusion_shift_rows)

    repaired_indices = np.where((clean_preds != test_labels) & (rgft_preds == test_labels))[0]
    new_error_indices = np.where((clean_preds == test_labels) & (rgft_preds != test_labels))[0]
    save_csv(os.path.join(part_c_dir, "repaired_examples.csv"), [{"index": int(idx)} for idx in repaired_indices])
    save_csv(os.path.join(part_c_dir, "new_errors.csv"), [{"index": int(idx)} for idx in new_error_indices])
    plot_example_grid(repaired_indices, test_images, target_images["CNN-clean"], clean_preds, rgft_preds, test_labels, "Repaired Examples", os.path.join(part_c_dir, "repaired_examples.png"))
    plot_example_grid(new_error_indices, test_images, target_images["CNN-clean"], clean_preds, rgft_preds, test_labels, "New Errors", os.path.join(part_c_dir, "new_errors.png"))

    save_csv(
        os.path.join(part_c_dir, "economy_comparison.csv"),
        [
            {
                "model_name": "CNN-AugScratch-target",
                "source_checkpoint": "random_init",
                "epochs": 30,
                "extra_epochs": 30,
                "time_sec": total_epoch_time("full_cnn_augscratch_target_v1/cnn_augscratch/history.csv"),
            },
            {
                "model_name": "CNN-AugScratch-mixed",
                "source_checkpoint": "random_init",
                "epochs": 30,
                "extra_epochs": 30,
                "time_sec": total_epoch_time("full_cnn_augscratch_mixed_v1/cnn_augscratch_mixed/history.csv"),
            },
            {
                "model_name": "CNN-RGFT-target",
                "source_checkpoint": "CNN-clean",
                "epochs": 10,
                "extra_epochs": 10,
                "time_sec": total_epoch_time("full_cnn_rgft_target_v1/cnn_rgft/history.csv"),
            },
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Generate missing final report assets from existing checkpoints.")
    parser.add_argument("--output-root", default="project_outputs")
    _ = parser.parse_args()
    generate_part_b_assets()
    generate_part_c_main_assets()


if __name__ == "__main__":
    main()
