import os
import time

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from experiment_utils import (
    build_model,
    build_optimizer,
    dense_target_strengths,
    ensure_dir,
    evaluate_model,
    evaluate_under_perturbation,
    format_inputs,
    load_mnist,
    load_model_from_checkpoint,
    non_target_representatives,
    same_type_neighbor,
    save_csv,
    save_history_csv,
    train_model,
    train_val_split,
)
import mynn as nn


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset", "MNIST")
TRAIN_IMAGES_PATH = os.path.join(DATASET_DIR, "train-images-idx3-ubyte.gz")
TRAIN_LABELS_PATH = os.path.join(DATASET_DIR, "train-labels-idx1-ubyte.gz")
TEST_IMAGES_PATH = os.path.join(DATASET_DIR, "t10k-images-idx3-ubyte.gz")
TEST_LABELS_PATH = os.path.join(DATASET_DIR, "t10k-labels-idx1-ubyte.gz")

BASELINE_CHECKPOINT = os.path.join(
    SCRIPT_DIR,
    "..",
    "full_cnn_clean_sgd_v1",
    "checkpoints",
    "cnn_clean_best_model.pickle",
)
SPLIT_PATH = os.path.join(SCRIPT_DIR, "..", "full_cnn_clean_sgd_v1", "splits", "train_val_idx.npy")

OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "..", "full_cnn_mixedft_v1")
CHECKPOINT_PATH = os.path.join(OUTPUT_ROOT, "checkpoints", "cnn_mixedft_best_model.pickle")
FALLBACK_CHECKPOINT_PATH = CHECKPOINT_PATH + ".fallback"
HISTORY_PATH = os.path.join(OUTPUT_ROOT, "cnn_mixedft", "history.csv")
LEARNING_CURVE_PATH = os.path.join(OUTPUT_ROOT, "cnn_mixedft", "cnn_mixedft_learning_curve.png")
SUMMARY_PATH = os.path.join(OUTPUT_ROOT, "cnn_mixedft", "mixed_eval_summary.csv")
VAL_SCORE_PATH = os.path.join(OUTPUT_ROOT, "cnn_mixedft", "mixed_validation_score.csv")
LIVE_PROGRESS_PATH = os.path.join(OUTPUT_ROOT, "live_progress.csv")

SEED = 309
EVAL_SEED = 2026
VAL_SIZE = 10000
BATCH_SIZE = 64
MAX_EPOCHS = 10
LEARNING_RATE = 0.005
CLEAN_MIX_RATIO = 0.5
CLEAN_DROP_LIMIT = 0.01
TARGET = {"type": "translation", "severity": 3}

DIAGNOSIS_GRID = (
    [{"type": "noise", "severity": sigma} for sigma in [0.05, 0.10, 0.20]]
    + [{"type": "translation", "severity": shift} for shift in [1, 2, 3]]
    + [{"type": "rotation", "severity": angle} for angle in [5, 10, 15]]
)
SELECTION_GRID = [
    {"type": "noise", "severity": 0.10},
    {"type": "translation", "severity": 2},
    {"type": "rotation", "severity": 10},
]


def maybe_plot_learning(history, save_path):
    if plt is None:
        return
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train loss")
    axes[0].plot(epochs, [row["clean_val_loss"] for row in history], marker="s", label="Clean val loss")
    axes[0].set_title("CNN-MixedFT Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [row["train_acc"] for row in history], marker="o", label="Train acc")
    axes[1].plot(epochs, [row["clean_val_acc"] for row in history], marker="s", label="Clean val acc")
    axes[1].plot(epochs, [row["target_val_acc"] for row in history], marker="^", label="Mixed val score")
    axes[1].set_title("CNN-MixedFT Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def append_live_progress(history):
    rows = []
    for row in history:
        rows.append(
            {
                "model_name": "cnn_mixedft",
                "epoch": row["epoch"],
                "train_loss": row["train_loss"],
                "train_acc": row["train_acc"],
                "clean_val_loss": row["clean_val_loss"],
                "clean_val_acc": row["clean_val_acc"],
                "target_val_acc": row["target_val_acc"],
                "epoch_time_sec": row["epoch_time_sec"],
                "learning_rate": row["learning_rate"],
                "selected_as_best": row["selected_as_best"],
                "selection_reason": row["selection_reason"],
            }
        )
    save_csv(LIVE_PROGRESS_PATH, rows)


def main():
    np.random.seed(SEED)
    ensure_dir(os.path.join(OUTPUT_ROOT, "checkpoints"))
    ensure_dir(os.path.dirname(HISTORY_PATH))

    train_images, train_labels = load_mnist(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, flatten=False, normalize=True)
    train_images, train_labels, valid_images, valid_labels, _ = train_val_split(
        train_images,
        train_labels,
        val_size=VAL_SIZE,
        seed=SEED,
        split_path=SPLIT_PATH,
    )
    test_images, test_labels = load_mnist(TEST_IMAGES_PATH, TEST_LABELS_PATH, flatten=False, normalize=True)

    baseline_model = load_model_from_checkpoint("cnn", BASELINE_CHECKPOINT)
    baseline_loss_fn = nn.op.MultiCrossEntropyLoss(model=baseline_model, max_classes=10)
    baseline_clean_val_acc = evaluate_model(
        baseline_model,
        "cnn",
        valid_images,
        valid_labels,
        baseline_loss_fn,
    )["acc"]

    model = load_model_from_checkpoint("cnn", BASELINE_CHECKPOINT)
    optimizer = build_optimizer("sgd", model, LEARNING_RATE)

    start = time.time()
    result = train_model(
        model=model,
        model_kind="cnn",
        optimizer=optimizer,
        train_images=train_images,
        train_labels=train_labels,
        valid_images=valid_images,
        valid_labels=valid_labels,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        checkpoint_path=CHECKPOINT_PATH,
        selection_mode="rgft",
        mixed_train_grid=DIAGNOSIS_GRID,
        selection_score_perturbations=SELECTION_GRID,
        clean_mix_ratio=CLEAN_MIX_RATIO,
        baseline_clean_val_acc=baseline_clean_val_acc,
        clean_drop_limit=CLEAN_DROP_LIMIT,
        seed=SEED,
        eval_seed=EVAL_SEED,
        drop_last=False,
    )
    total_time = time.time() - start

    save_history_csv(HISTORY_PATH, result["history"])
    append_live_progress(result["history"])
    maybe_plot_learning(result["history"], LEARNING_CURVE_PATH)

    best_model = load_model_from_checkpoint("cnn", CHECKPOINT_PATH)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=best_model, max_classes=10)
    clean_test_metrics = evaluate_model(best_model, "cnn", test_images, test_labels, loss_fn)

    summary_rows = []
    eval_settings = [("clean", None), ("target", TARGET), ("same_type_neighbor", same_type_neighbor(TARGET))]
    eval_settings.extend([(f"non_target_{idx+1}", p) for idx, p in enumerate(non_target_representatives(TARGET))])
    for setting_name, perturbation in eval_settings:
        if perturbation is None:
            metrics = clean_test_metrics
        else:
            metrics, _ = evaluate_under_perturbation(best_model, "cnn", test_images, test_labels, perturbation, seed=EVAL_SEED)
        summary_rows.append(
            {
                "model_name": "CNN-MixedFT",
                "setting": setting_name,
                "perturbation_type": "clean" if perturbation is None else perturbation["type"],
                "severity": 0 if perturbation is None else perturbation["severity"],
                "accuracy": metrics["acc"],
                "accuracy_drop": clean_test_metrics["acc"] - metrics["acc"],
            }
        )

    robustness_rows = []
    robustness_accs = []
    for perturbation in DIAGNOSIS_GRID:
        metrics, _ = evaluate_under_perturbation(best_model, "cnn", test_images, test_labels, perturbation, seed=EVAL_SEED)
        robustness_rows.append(
            {
                "model_name": "CNN-RGFT-mixed",
                "perturbation_type": perturbation["type"],
                "severity": perturbation["severity"],
                "accuracy": metrics["acc"],
            }
        )
        robustness_accs.append(metrics["acc"])

    dense_rows = []
    for severity in dense_target_strengths(TARGET):
        metrics, _ = evaluate_under_perturbation(
            best_model,
            "cnn",
            test_images,
            test_labels,
            {"type": TARGET["type"], "severity": severity},
            seed=EVAL_SEED,
        )
        dense_rows.append(
            {
                "model_name": "CNN-RGFT-mixed",
                "severity": severity,
                "accuracy": metrics["acc"],
            }
        )

    selected_accs = [evaluate_model(best_model, "cnn", valid_images, valid_labels, loss_fn)["acc"]]
    for perturbation in SELECTION_GRID:
        metrics, _ = evaluate_under_perturbation(best_model, "cnn", valid_images, valid_labels, perturbation, seed=EVAL_SEED)
        selected_accs.append(metrics["acc"])

    save_csv(SUMMARY_PATH, summary_rows)
    save_csv(os.path.join(OUTPUT_ROOT, "cnn_mixedft", "mixed_robustness_curve.csv"), robustness_rows)
    save_csv(os.path.join(OUTPUT_ROOT, "cnn_mixedft", "mixed_target_dense_curve.csv"), dense_rows)
    save_csv(
        VAL_SCORE_PATH,
        [
            {
                "model_name": "CNN-MixedFT",
                "baseline_clean_val_acc": baseline_clean_val_acc,
                "clean_drop_limit": CLEAN_DROP_LIMIT,
                "clean_val_acc": selected_accs[0],
                "noise_0.10_val_acc": selected_accs[1],
                "translation_2_val_acc": selected_accs[2],
                "rotation_10_val_acc": selected_accs[3],
                "mixed_val_score": float(np.mean(selected_accs)),
                "test_robustness_average": float(np.mean(robustness_accs)),
                "total_time_sec": total_time,
                "checkpoint_path": CHECKPOINT_PATH,
                "fallback_checkpoint_path": FALLBACK_CHECKPOINT_PATH,
            }
        ],
    )

    best_epoch = result["best_state"]["epoch"] if result["best_state"] is not None else ""
    best_reason = result["best_state"]["reason"] if result["best_state"] is not None else ""
    print("training_complete")
    print(f"checkpoint={CHECKPOINT_PATH}")
    print(f"baseline_clean_val_acc={baseline_clean_val_acc:.6f}")
    print(f"best_epoch={best_epoch}")
    print(f"best_reason={best_reason}")
    print(f"mixed_val_score={float(np.mean(selected_accs)):.6f}")
    print(f"test_clean={clean_test_metrics['acc']:.6f}")
    print(f"test_target={summary_rows[1]['accuracy']:.6f}")
    print(f"test_robustness_average={float(np.mean(robustness_accs)):.6f}")


if __name__ == "__main__":
    main()
