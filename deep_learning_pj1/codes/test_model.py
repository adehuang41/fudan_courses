import argparse
import os

from experiment_utils import evaluate_under_perturbation, format_inputs, load_mnist, load_model_from_checkpoint
import mynn as nn


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset', 'MNIST')
TEST_IMAGES_PATH = os.path.join(DATASET_DIR, 't10k-images-idx3-ubyte.gz')
TEST_LABELS_PATH = os.path.join(DATASET_DIR, 't10k-labels-idx1-ubyte.gz')


def build_parser():
    parser = argparse.ArgumentParser(description='Evaluate a saved MNIST model.')
    parser.add_argument('--model-kind', required=True, choices=['mlp', 'cnn'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--perturbation-type', choices=['noise', 'translation', 'rotation'], default=None)
    parser.add_argument('--perturbation-severity', type=float, default=None)
    parser.add_argument('--seed', type=int, default=2026)
    return parser


def main():
    args = build_parser().parse_args()
    model = load_model_from_checkpoint(args.model_kind, args.checkpoint)
    test_images, test_labels = load_mnist(TEST_IMAGES_PATH, TEST_LABELS_PATH, flatten=False, normalize=True)

    if args.perturbation_type is None:
        logits = model(format_inputs(test_images, args.model_kind))
        print(nn.metric.accuracy(logits, test_labels))
        return

    perturbation = {
        'type': args.perturbation_type,
        'severity': int(args.perturbation_severity) if args.perturbation_type in {'translation', 'rotation'} else float(args.perturbation_severity),
    }
    metrics, _ = evaluate_under_perturbation(model, args.model_kind, test_images, test_labels, perturbation, seed=args.seed)
    print(metrics['acc'])


if __name__ == '__main__':
    main()
