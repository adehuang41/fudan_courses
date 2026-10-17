import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from src.train import evaluate_model
from src.model import MLP

def collect_predictions(model, data_loader):
    model.eval()
    all_preds = []
    for X_batch, _ in data_loader:
        logits = model.forward(X_batch)
        all_preds.extend(np.argmax(logits, axis=1))
    return np.array(all_preds)

def load_trained_model(model_path, config):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"weights not found: {model_path}")
    model = MLP(
        input_dim=784,
        hidden_dims=config['hidden_dims'],
        num_classes=10,
        activation=config.get('activation', 'relu'),
        dropout_rate=config.get('dropout_rate', 0.0)
    )
    model.load_model(model_path)
    return model

def evaluate_and_plot_confusion_matrix(model, test_loader, criterion, y_test_labels,
                                       save_dir='./logs', return_predictions=False,
                                       class_names=None):
    test_loss, test_acc = evaluate_model(model, test_loader, criterion)
    print(f"test  loss={test_loss:.4f}  acc={test_acc:.4f}")

    y_pred_labels = collect_predictions(model, test_loader)

    cm = confusion_matrix(y_test_labels, y_pred_labels)
    tick_labels = class_names if class_names is not None else range(10)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title('Confusion Matrix on Test Set', fontsize=16)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.ylabel('True Class', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()

    if return_predictions:
        return test_acc, y_pred_labels, cm
    return test_acc

def visualize_first_layer_weights(model, num_neurons_to_show=16, save_dir='./logs'):
    first_linear = next(l for l in model.layers if hasattr(l, 'params') and 'W' in l.params)
    W1 = first_linear.params['W']

    grid_size = int(np.ceil(np.sqrt(num_neurons_to_show)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.suptitle('Visualization of 1st Hidden Layer Neurons', fontsize=18, y=1.02)

    for i, ax in enumerate(axes.flat):
        if i < W1.shape[1] and i < num_neurons_to_show:
            w = W1[:, i].reshape(28, 28)
            w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)
            ax.imshow(w_norm, cmap='viridis')
            ax.set_title(f'Neuron #{i+1}', fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'weight_visualization.png'), dpi=150, bbox_inches='tight')
    plt.show()

def plot_misclassified_samples(X_test, y_true_labels, y_pred_labels, class_names=None,
                               num_samples=9, save_dir='./logs'):
    misclassified_idx = np.where(y_true_labels != y_pred_labels)[0]

    if len(misclassified_idx) == 0:
        print("no misclassified samples.")
        return misclassified_idx

    num_to_show = min(num_samples, len(misclassified_idx))
    show_idx = np.random.choice(misclassified_idx, num_to_show, replace=False)
    grid_size = int(np.ceil(np.sqrt(num_to_show)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = np.array(axes).reshape(-1)
    fig.suptitle('Misclassified Test Samples', fontsize=18, y=1.02)

    for i, ax in enumerate(axes):
        if i < num_to_show:
            idx = show_idx[i]
            ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
            true_label = class_names[y_true_labels[idx]] if class_names is not None else y_true_labels[idx]
            pred_label = class_names[y_pred_labels[idx]] if class_names is not None else y_pred_labels[idx]
            ax.set_title(f'T: {true_label}\nP: {pred_label}', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'error_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    return show_idx

def summarize_top_confusions(y_true_labels, y_pred_labels, class_names=None, top_k=5):
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    np.fill_diagonal(cm, 0)
    flat_idx = np.argsort(cm.ravel())[::-1]
    summary = []
    for idx in flat_idx:
        count = int(cm.ravel()[idx])
        if count == 0:
            break
        true_idx, pred_idx = np.unravel_index(idx, cm.shape)
        summary.append({
            'true_idx':  true_idx,
            'pred_idx':  pred_idx,
            'true_name': class_names[true_idx] if class_names is not None else true_idx,
            'pred_name': class_names[pred_idx] if class_names is not None else pred_idx,
            'count':     count
        })
        if len(summary) >= top_k:
            break
    return summary
