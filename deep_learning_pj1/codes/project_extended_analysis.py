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
    confusion_matrix_from_logits,
    ensure_dir,
    evaluate_model,
    evaluate_under_perturbation,
    load_mnist,
    load_model_from_checkpoint,
    matrix_to_rows,
    save_csv,
)
import mynn as nn


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset", "MNIST")
TEST_IMAGES_PATH = os.path.join(DATASET_DIR, "t10k-images-idx3-ubyte.gz")
TEST_LABELS_PATH = os.path.join(DATASET_DIR, "t10k-labels-idx1-ubyte.gz")


def load_history(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "train_acc": float(row["train_acc"]),
                    "clean_val_loss": float(row["clean_val_loss"]),
                    "clean_val_acc": float(row["clean_val_acc"]),
                    "target_val_acc": None if row["target_val_acc"] == "" else float(row["target_val_acc"]),
                    "selected_as_best": row["selected_as_best"] == "True",
                    "selection_reason": row["selection_reason"],
                }
            )
    return rows


def load_selected_target(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    target_type = row.get("type", row.get("target_type"))
    severity_raw = row.get("severity", row.get("target_severity"))
    severity = int(float(severity_raw)) if target_type in {"translation", "rotation"} else float(severity_raw)
    return {"type": target_type, "severity": severity}


def get_trainable_summary(freeze_policy):
    model = build_model("cnn", freeze_policy=freeze_policy)
    return model.trainable_parameter_summary()


def sample_updates(subset_size, epochs):
    return int(subset_size * epochs)


def build_parser():
    parser = argparse.ArgumentParser(description="Aggregate extended Part C results.")
    parser.add_argument("--output-dir", default="project_outputs/part_c_extended")
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--target-path", default="project_outputs/part_c/selected_target.csv")
    return parser


def model_specs(base_dir):
    return [
        {
            "model_name": "CNN-clean",
            "checkpoint": "full_cnn_clean_sgd_v1/checkpoints/cnn_clean_best_model.pickle",
            "freeze_policy": "all",
            "history": "full_cnn_clean_sgd_v1/cnn_clean/history.csv",
            "subset_size": 50000,
            "group": "baseline",
        },
        {
            "model_name": "CNN-AugScratch-target",
            "checkpoint": "full_cnn_augscratch_target_v1/checkpoints/cnn_augscratch_best_model.pickle",
            "freeze_policy": "all",
            "history": "full_cnn_augscratch_target_v1/cnn_augscratch/history.csv",
            "subset_size": 50000,
            "group": "main",
        },
        {
            "model_name": "CNN-RGFT-full",
            "checkpoint": "full_cnn_rgft_target_v1/checkpoints/cnn_rgft_best_model.pickle",
            "freeze_policy": "all",
            "history": "full_cnn_rgft_target_v1/cnn_rgft/history.csv",
            "subset_size": 50000,
            "group": "main",
        },
        {
            "model_name": "CNN-RGFT-random10-all",
            "checkpoint": os.path.join(base_dir, "random10_all/checkpoints/cnn_rgft_best_model.pickle"),
            "freeze_policy": "all",
            "history": os.path.join(base_dir, "random10_all/cnn_rgft/history.csv"),
            "subset_size": 5000,
            "group": "main",
        },
        {
            "model_name": "CNN-RGFT-hard10-all",
            "checkpoint": os.path.join(base_dir, "hard10_all/checkpoints/cnn_rgft_best_model.pickle"),
            "freeze_policy": "all",
            "history": os.path.join(base_dir, "hard10_all/cnn_rgft/history.csv"),
            "subset_size": 5000,
            "group": "main",
        },
        {
            "model_name": "CNN-RGFT-hard10-head",
            "checkpoint": os.path.join(base_dir, "hard10_head/checkpoints/cnn_rgft_best_model.pickle"),
            "freeze_policy": "head_only",
            "history": os.path.join(base_dir, "hard10_head/cnn_rgft/history.csv"),
            "subset_size": 5000,
            "group": "ablation",
        },
        {
            "model_name": "CNN-RGFT-hard10-conv2head",
            "checkpoint": os.path.join(base_dir, "hard10_conv2head/checkpoints/cnn_rgft_best_model.pickle"),
            "freeze_policy": "conv2_head",
            "history": os.path.join(base_dir, "hard10_conv2head/cnn_rgft/history.csv"),
            "subset_size": 5000,
            "group": "ablation",
        },
    ]


def main_eval_perturbations(target):
    return [
        ("clean", None),
        ("target", target),
        ("same_type_neighbor", {"type": target["type"], "severity": 2 if target["type"] == "translation" else target["severity"]}),
        ("non_target_noise", {"type": "noise", "severity": 0.10}),
        ("non_target_rotation", {"type": "rotation", "severity": 10}),
    ]


def dense_strengths(target):
    if target["type"] == "translation":
        return [0, 1, 2, 3]
    if target["type"] == "noise":
        return [0.0, 0.05, 0.10, 0.15, 0.20]
    if target["type"] == "rotation":
        return [0, 5, 10, 15]
    raise ValueError(target["type"])


def plot_group_bars(rows, save_path, title, metric_key, model_order):
    if plt is None:
        return
    values = []
    for model_name in model_order:
        row = next(item for item in rows if item["model_name"] == model_name)
        values.append(row[metric_key])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(model_order)), values)
    ax.set_xticks(np.arange(len(model_order)))
    ax.set_xticklabels(model_order, rotation=25, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_efficiency(rows, save_path):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    for row in rows:
        ax.scatter(row["sample_updates"], row["target_test_acc"], s=60)
        ax.text(row["sample_updates"], row["target_test_acc"], row["model_name"], fontsize=8)
    ax.set_xlabel("Sample updates")
    ax.set_ylabel("Target test acc")
    ax.set_title("Efficiency vs Target Robustness")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_dense_curve(rows, save_path):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["CNN-clean", "CNN-RGFT-random10-all", "CNN-RGFT-hard10-all", "CNN-RGFT-hard10-head", "CNN-RGFT-hard10-conv2head"]
    for name in names:
        sub = [row for row in rows if row["model_name"] == name]
        ax.plot([row["severity"] for row in sub], [row["accuracy"] for row in sub], marker="o", label=name)
    ax.set_xlabel("Target severity")
    ax.set_ylabel("Accuracy")
    ax.set_title("Target-only Dense Curve (Extended)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    ensure_dir(args.output_dir)
    target = load_selected_target(args.target_path)

    test_images, test_labels = load_mnist(TEST_IMAGES_PATH, TEST_LABELS_PATH, flatten=False, normalize=True)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=None, max_classes=10)

    specs = model_specs(args.output_dir)
    loaded = {}
    for spec in specs:
        loaded[spec["model_name"]] = {
            "spec": spec,
            "history": load_history(spec["history"]),
            "model": load_model_from_checkpoint("cnn", spec["checkpoint"], freeze_policy=spec["freeze_policy"]),
            "trainable_summary": get_trainable_summary(spec["freeze_policy"]),
        }

    clean_baseline_metrics = evaluate_model(loaded["CNN-clean"]["model"], "cnn", test_images, test_labels, loss_fn)

    overall_rows = []
    summary_rows = []
    repair_rows = []
    dense_rows = []

    main_models = ["CNN-clean", "CNN-AugScratch-target", "CNN-RGFT-full", "CNN-RGFT-random10-all", "CNN-RGFT-hard10-all"]
    ablation_models = ["CNN-RGFT-hard10-head", "CNN-RGFT-hard10-conv2head", "CNN-RGFT-hard10-all"]

    target_logits = {}
    for model_name, payload in loaded.items():
        clean_metrics = evaluate_model(payload["model"], "cnn", test_images, test_labels, loss_fn)
        target_metrics, _ = evaluate_under_perturbation(payload["model"], "cnn", test_images, test_labels, target, seed=args.eval_seed)
        target_logits[model_name] = target_metrics["logits"]

        history = payload["history"]
        epochs = history[-1]["epoch"]
        summary = payload["trainable_summary"]
        target_drop = clean_metrics["acc"] - target_metrics["acc"]
        summary_rows.append(
            {
                "model_name": model_name,
                "freeze_policy": payload["spec"]["freeze_policy"],
                "epochs": epochs,
                "subset_size": payload["spec"]["subset_size"],
                "sample_updates": sample_updates(payload["spec"]["subset_size"], epochs),
                "trainable_param_count": summary["trainable_param_count"],
                "total_param_count": summary["total_param_count"],
                "trainable_param_ratio": summary["trainable_param_ratio"],
                "clean_test_acc": clean_metrics["acc"],
                "target_test_acc": target_metrics["acc"],
                "target_test_drop": target_drop,
                "group": payload["spec"]["group"],
            }
        )

        for setting_name, perturbation in main_eval_perturbations(target):
            if perturbation is None:
                metrics = clean_metrics
            else:
                metrics, _ = evaluate_under_perturbation(payload["model"], "cnn", test_images, test_labels, perturbation, seed=args.eval_seed)
            overall_rows.append(
                {
                    "model_name": model_name,
                    "setting": setting_name,
                    "perturbation_type": "clean" if perturbation is None else perturbation["type"],
                    "severity": 0 if perturbation is None else perturbation["severity"],
                    "accuracy": metrics["acc"],
                    "accuracy_drop": clean_metrics["acc"] - metrics["acc"],
                    "group": payload["spec"]["group"],
                }
            )

        if model_name in {"CNN-clean", "CNN-RGFT-random10-all", "CNN-RGFT-hard10-all", "CNN-RGFT-hard10-head", "CNN-RGFT-hard10-conv2head"}:
            for severity in dense_strengths(target):
                perturbation = {"type": target["type"], "severity": severity}
                metrics, _ = evaluate_under_perturbation(payload["model"], "cnn", test_images, test_labels, perturbation, seed=args.eval_seed)
                dense_rows.append({"model_name": model_name, "severity": severity, "accuracy": metrics["acc"]})

    clean_preds = np.argmax(target_logits["CNN-clean"], axis=1)
    for model_name in ["CNN-RGFT-full", "CNN-RGFT-random10-all", "CNN-RGFT-hard10-all", "CNN-RGFT-hard10-head", "CNN-RGFT-hard10-conv2head", "CNN-AugScratch-target"]:
        preds = np.argmax(target_logits[model_name], axis=1)
        repaired = int(np.sum((clean_preds != test_labels) & (preds == test_labels)))
        new_errors = int(np.sum((clean_preds == test_labels) & (preds != test_labels)))
        repair_rows.append(
            {
                "model_name": model_name,
                "repaired_count": repaired,
                "new_error_count": new_errors,
                "net_repair": repaired - new_errors,
            }
        )

    save_csv(os.path.join(args.output_dir, "extended_test_overall_results.csv"), overall_rows)
    save_csv(os.path.join(args.output_dir, "extended_summary.csv"), summary_rows)
    save_csv(os.path.join(args.output_dir, "extended_repair_summary.csv"), repair_rows)
    save_csv(os.path.join(args.output_dir, "extended_target_dense_curve.csv"), dense_rows)

    main_table_rows = []
    for model_name in main_models:
        summary = next(row for row in summary_rows if row["model_name"] == model_name)
        repairs = next((row for row in repair_rows if row["model_name"] == model_name), {"repaired_count": "", "new_error_count": "", "net_repair": ""})
        neighbor = next(row for row in overall_rows if row["model_name"] == model_name and row["setting"] == "same_type_neighbor")
        non_target_noise = next(row for row in overall_rows if row["model_name"] == model_name and row["setting"] == "non_target_noise")
        non_target_rotation = next(row for row in overall_rows if row["model_name"] == model_name and row["setting"] == "non_target_rotation")
        main_table_rows.append(
            {
                "model_name": model_name,
                "clean_test_acc": summary["clean_test_acc"],
                "target_test_acc": summary["target_test_acc"],
                "target_test_drop": summary["target_test_drop"],
                "same_type_neighbor_acc": neighbor["accuracy"],
                "non_target_noise_acc": non_target_noise["accuracy"],
                "non_target_rotation_acc": non_target_rotation["accuracy"],
                "sample_updates": summary["sample_updates"],
                "repaired_count": repairs["repaired_count"],
                "new_error_count": repairs["new_error_count"],
                "net_repair": repairs["net_repair"],
            }
        )

    ablation_table_rows = []
    for model_name in ablation_models:
        summary = next(row for row in summary_rows if row["model_name"] == model_name)
        repairs = next(row for row in repair_rows if row["model_name"] == model_name)
        ablation_table_rows.append(
            {
                "model_name": model_name,
                "freeze_policy": next(spec["freeze_policy"] for spec in specs if spec["model_name"] == model_name),
                "clean_test_acc": summary["clean_test_acc"],
                "target_test_acc": summary["target_test_acc"],
                "target_test_drop": summary["target_test_drop"],
                "trainable_param_count": summary["trainable_param_count"],
                "trainable_param_ratio": summary["trainable_param_ratio"],
                "repaired_count": repairs["repaired_count"],
                "new_error_count": repairs["new_error_count"],
                "net_repair": repairs["net_repair"],
                "sample_updates": summary["sample_updates"],
            }
        )

    save_csv(os.path.join(args.output_dir, "extended_main_table.csv"), main_table_rows)
    save_csv(os.path.join(args.output_dir, "extended_ablation_table.csv"), ablation_table_rows)

    clean_matrix = confusion_matrix_from_logits(target_logits["CNN-clean"], test_labels)
    hard_matrix = confusion_matrix_from_logits(target_logits["CNN-RGFT-hard10-all"], test_labels)
    confusion_shift_rows = []
    for i in range(clean_matrix.shape[0]):
        for j in range(clean_matrix.shape[1]):
            if i == j:
                continue
            clean_count = int(clean_matrix[i, j])
            hard_count = int(hard_matrix[i, j])
            confusion_shift_rows.append(
                {
                    "true_class": i,
                    "pred_class": j,
                    "clean_count": clean_count,
                    "hard10_all_count": hard_count,
                    "absolute_reduction": clean_count - hard_count,
                    "relative_reduction": ((clean_count - hard_count) / clean_count) if clean_count > 0 else 0.0,
                }
            )
    confusion_shift_rows.sort(key=lambda row: row["absolute_reduction"], reverse=True)
    save_csv(os.path.join(args.output_dir, "extended_confusion_shift_hard10_all.csv"), confusion_shift_rows)
    save_csv(os.path.join(args.output_dir, "extended_cnn_clean_target_confusion_matrix.csv"), matrix_to_rows(clean_matrix))
    save_csv(os.path.join(args.output_dir, "extended_hard10_all_target_confusion_matrix.csv"), matrix_to_rows(hard_matrix))

    plot_group_bars(main_table_rows, os.path.join(args.output_dir, "extended_main_target_acc.png"), "Extended Main Comparison: Target Accuracy", "target_test_acc", main_models)
    plot_group_bars(ablation_table_rows, os.path.join(args.output_dir, "extended_ablation_target_acc.png"), "Freezing Ablation: Target Accuracy", "target_test_acc", ablation_models)
    plot_group_bars(repair_rows, os.path.join(args.output_dir, "extended_net_repair.png"), "Net Repair by Model", "net_repair", [row["model_name"] for row in repair_rows])
    plot_efficiency(summary_rows, os.path.join(args.output_dir, "extended_efficiency_vs_target.png"))
    plot_dense_curve(dense_rows, os.path.join(args.output_dir, "extended_target_dense_curve.png"))


if __name__ == "__main__":
    main()
