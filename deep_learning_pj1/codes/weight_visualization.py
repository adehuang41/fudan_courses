import argparse
import math
import os

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from experiment_utils import ensure_dir, load_model_from_checkpoint


def build_parser():
    parser = argparse.ArgumentParser(description='Visualize learned MLP weights or CNN kernels.')
    parser.add_argument('--model-kind', required=True, choices=['mlp', 'cnn'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-items', type=int, default=16)
    return parser


def plot_grid(images, save_path, title):
    if plt is None:
        return

    num_items = len(images)
    num_cols = int(math.ceil(math.sqrt(num_items)))
    num_rows = int(math.ceil(num_items / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2.2 * num_cols, 2.2 * num_rows))
    axes = axes.reshape(-1) if hasattr(axes, 'reshape') else [axes]

    for ax, image in zip(axes, images):
        ax.imshow(image, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(images):]:
        ax.axis('off')

    fig.suptitle(title)
    fig.tight_layout()
    ensure_dir(os.path.dirname(save_path) or '.')
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def visualize_mlp(checkpoint, output, max_items):
    model = load_model_from_checkpoint('mlp', checkpoint)
    weights = model.layers[0].params['W'].T
    images = [weights[idx].reshape(28, 28) for idx in range(min(max_items, weights.shape[0]))]
    plot_grid(images, output, 'MLP First-Layer Weights')


def visualize_cnn(checkpoint, output, max_items):
    model = load_model_from_checkpoint('cnn', checkpoint)
    kernels = model.layers[0].params['W']
    images = [kernels[idx, 0] for idx in range(min(max_items, kernels.shape[0]))]
    plot_grid(images, output, 'CNN First-Layer Kernels')


def main():
    args = build_parser().parse_args()
    if args.model_kind == 'mlp':
        visualize_mlp(args.checkpoint, args.output, args.max_items)
    else:
        visualize_cnn(args.checkpoint, args.output, args.max_items)


if __name__ == '__main__':
    main()
