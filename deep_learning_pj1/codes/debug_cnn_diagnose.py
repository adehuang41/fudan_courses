import argparse
import os

import numpy as np

import mynn as nn


def rel_error(a, b):
    denom = np.maximum(1e-8, np.abs(a) + np.abs(b))
    return float(np.max(np.abs(a - b) / denom))


def numeric_grad_param(loss_fn, param, index, eps=1e-5):
    original = param[index]
    param[index] = original + eps
    loss_pos = loss_fn()
    param[index] = original - eps
    loss_neg = loss_fn()
    param[index] = original
    return (loss_pos - loss_neg) / (2 * eps)


def numeric_grad_input(loss_fn, x, index, eps=1e-5):
    original = x[index]
    x[index] = original + eps
    loss_pos = loss_fn()
    x[index] = original - eps
    loss_neg = loss_fn()
    x[index] = original
    return (loss_pos - loss_neg) / (2 * eps)


def check_conv_gradients(seed=0):
    rng = np.random.default_rng(seed)
    layer = nn.op.conv2D(1, 2, kernel_size=3, stride=1, padding=1)
    x = rng.normal(size=(2, 1, 5, 5)).astype(np.float64)
    upstream = rng.normal(size=(2, 2, 5, 5)).astype(np.float64)

    out = layer.forward(x)
    grad_input = layer.backward(upstream)

    def scalar_loss():
        return float(np.sum(layer.forward(x) * upstream))

    checks = {
        "conv_W": ((0, 0, 1, 1), layer.grads["W"], layer.W),
        "conv_b": ((0,), layer.grads["b"], layer.b),
    }

    results = {}
    for name, (idx, analytic, param) in checks.items():
        num = numeric_grad_param(scalar_loss, param, idx)
        results[name] = {
            "analytic": float(analytic[idx]),
            "numeric": float(num),
            "rel_error": rel_error(np.array([analytic[idx]]), np.array([num])),
        }

    input_idx = (0, 0, 2, 3)
    num_in = numeric_grad_input(scalar_loss, x, input_idx)
    results["conv_input"] = {
        "analytic": float(grad_input[input_idx]),
        "numeric": float(num_in),
        "rel_error": rel_error(np.array([grad_input[input_idx]]), np.array([num_in])),
    }
    return results


def check_pool_gradients(seed=0):
    rng = np.random.default_rng(seed)
    layer = nn.op.MaxPool2D(kernel_size=2, stride=2)
    x = rng.normal(size=(2, 3, 4, 4)).astype(np.float64)
    upstream = rng.normal(size=(2, 3, 2, 2)).astype(np.float64)
    layer.forward(x)
    grad_input = layer.backward(upstream)

    def scalar_loss():
        return float(np.sum(layer.forward(x) * upstream))

    input_idx = (0, 1, 2, 1)
    num_in = numeric_grad_input(scalar_loss, x, input_idx)
    return {
        "pool_input": {
            "analytic": float(grad_input[input_idx]),
            "numeric": float(num_in),
            "rel_error": rel_error(np.array([grad_input[input_idx]]), np.array([num_in])),
        }
    }


def check_loss_scaling(seed=0):
    rng = np.random.default_rng(seed)
    model = nn.models.Model_MLP([4, 3], "ReLU")
    x = rng.normal(size=(5, 4))
    y = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    logits = model(x)
    loss = nn.op.MultiCrossEntropyLoss(model=model, max_classes=3)
    loss.forward(logits, y)
    loss.backward()

    probs = nn.op.softmax(logits)
    expected = probs.copy()
    expected[np.arange(x.shape[0]), y] -= 1.0
    expected /= x.shape[0]
    return {
        "loss_grad_rel_error": rel_error(loss.grads, expected),
        "analytic_norm": float(np.linalg.norm(loss.grads)),
        "expected_norm": float(np.linalg.norm(expected)),
    }


def one_batch_descent(seed=0, lr=0.005, steps=30):
    rng = np.random.default_rng(seed)
    model = nn.models.Model_CNN()
    optimizer = nn.optimizer.SGD(init_lr=lr, model=model)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)

    x = rng.normal(size=(8, 1, 28, 28)).astype(np.float32)
    y = np.arange(8, dtype=np.int64) % 10
    losses = []
    accs = []
    for _ in range(steps):
        logits = model(x)
        loss = loss_fn(logits, y)
        losses.append(float(loss))
        accs.append(float(nn.metric.accuracy(logits, y)))
        loss_fn.backward()
        optimizer.step()
    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "initial_acc": accs[0],
        "final_acc": accs[-1],
        "losses": losses,
        "accs": accs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["conv", "pool", "loss", "one-batch", "all"], default="all")
    args = parser.parse_args()

    if args.task in {"conv", "all"}:
        print("CONV", check_conv_gradients())
    if args.task in {"pool", "all"}:
        print("POOL", check_pool_gradients())
    if args.task in {"loss", "all"}:
        print("LOSS", check_loss_scaling())
    if args.task in {"one-batch", "all"}:
        print("ONE_BATCH", one_batch_descent())


if __name__ == "__main__":
    main()
