import numpy as np
import os
import time

def compute_accuracy(logits, y_true_oh):
    preds  = np.argmax(logits, axis=1)
    labels = np.argmax(y_true_oh, axis=1)
    return np.mean(preds == labels)

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss, total_acc, total_samples = 0.0, 0.0, 0

    for X_batch, y_batch in data_loader:
        logits = model.forward(X_batch)
        loss   = criterion.forward(logits, y_batch)
        batch_size = X_batch.shape[0]
        total_loss += loss * batch_size
        total_acc  += compute_accuracy(logits, y_batch) * batch_size
        total_samples += batch_size

    return total_loss / total_samples, total_acc / total_samples

def train_model(model, train_loader, val_loader, optimizer, criterion,
                epochs=30, scheduler=None, save_path='./models/best_model.pkl',
                early_stopping_patience=0, min_delta=0.0, verbose=True):

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = -np.inf
    patience_counter = 0

    if verbose:
        print(f"training {epochs} epochs ...")

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()
        train_loss_total, train_acc_total, train_samples = 0.0, 0.0, 0

        for X_batch, y_batch in train_loader:
            logits  = model.forward(X_batch)
            loss    = criterion.forward(logits, y_batch)
            dlogits = criterion.backward()
            model.backward(dlogits)
            optimizer.step()

            batch_size = X_batch.shape[0]
            train_loss_total += loss * batch_size
            train_acc_total  += compute_accuracy(logits, y_batch) * batch_size
            train_samples    += batch_size

        if scheduler is not None:
            scheduler.step()

        train_loss = train_loss_total / train_samples
        train_acc  = train_acc_total  / train_samples
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if verbose:
            epoch_time = time.time() - epoch_start
            print(f"epoch {epoch+1:02d}/{epochs} [{epoch_time:.1f}s] "
                  f"lr={optimizer.lr:.5f}  "
                  f"train loss={train_loss:.4f} acc={train_acc:.4f}  "
                  f"val loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc - best_val_acc > min_delta:
            patience_counter = 0
            best_val_acc = val_acc
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            model.save_model(save_path)
            if verbose:
                print(f"  -> saved (val acc {val_acc:.4f})")
        else:
            patience_counter += 1

        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            if verbose:
                print(f"early stopping at epoch {epoch+1}")
            break

    total_time = time.time() - start_time
    if verbose:
        print(f"done. {total_time:.1f}s  best val acc: {best_val_acc:.4f}")

    model.load_model(save_path)
    return history
