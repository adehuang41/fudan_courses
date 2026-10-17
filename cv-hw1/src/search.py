import itertools
import os
import time
from src.data_utils import BatchIterator
from src.model import MLP
from src.loss import CrossEntropyLoss
from src.optimizers import SGD, ExponentialScheduler
from src.train import train_model

def run_grid_search(X_train, y_train_oh, X_val, y_val_oh, param_grid, max_epochs=10, patience=2):
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./models', exist_ok=True)

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"{len(combinations)} trials to run ...")

    search_results = []

    for i, config in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] {config}")

        train_loader = BatchIterator(X_train, y_train_oh, batch_size=config['batch_size'], shuffle=True)
        val_loader   = BatchIterator(X_val,   y_val_oh,   batch_size=config['batch_size'], shuffle=False)

        model     = MLP(input_dim=784, hidden_dims=config['hidden_dims'], num_classes=10,
                        activation=config['activation'], dropout_rate=config.get('dropout_rate', 0.0))
        criterion = CrossEntropyLoss()
        optimizer = SGD(model=model, lr=config['learning_rate'], momentum=0.9,
                        weight_decay=config['weight_decay'])
        scheduler = ExponentialScheduler(optimizer, decay_rate=0.9, decay_steps=5)

        save_path = f'./models/grid_search_trial_{i+1}.pkl'

        history = train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, criterion=criterion, epochs=max_epochs,
            scheduler=scheduler, save_path=save_path,
            early_stopping_patience=patience,
            min_delta=0.001,
            verbose=False
        )

        search_results.append({
            'config':     config,
            'val_acc':    max(history['val_acc']),
            'model_path': save_path
        })

    sorted_results = sorted(search_results, key=lambda x: x['val_acc'], reverse=True)
    report_path = f"./logs/report_{time.strftime('%m%d_%H%M')}.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("grid search results\n\n")
        for rank, res in enumerate(sorted_results):
            f.write(f"Top {rank+1} | Acc: {res['val_acc']:.4f} | 配置: {res['config']}\n")

    return sorted_results, report_path
