import os
import gzip
import urllib.request
import numpy as np

def download_and_load_fashion_mnist(data_dir='./data'):
    """从官方镜像下载并解析 Fashion-MNIST 二进制文件"""
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }

    os.makedirs(data_dir, exist_ok=True)
    data = {}

    for key, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"正在下载 {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)

        # 解析 gzip 压缩的 IDX 格式文件
        with gzip.open(filepath, 'rb') as f:
            if 'labels' in key:
                data[key] = np.frombuffer(f.read(), np.uint8, offset=8)
            else:
                data[key] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)

    return data['train_images'], data['train_labels'], data['test_images'], data['test_labels']

def one_hot_encode(y, num_classes=10):
    """将整数标签转换为独热编码"""
    return np.eye(num_classes)[y]

def get_preprocessed_data(val_split=0.2, seed=42):
    """数据预处理主流水线"""
    X_train_full, y_train_full, X_test, y_test = download_and_load_fashion_mnist()

    # 1. 归一化到 [0, 1] 区间
    X_train_full = X_train_full.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # 2. 独热编码
    y_train_full_oh = one_hot_encode(y_train_full)
    y_test_oh = one_hot_encode(y_test)

    # 3. 划分训练集和验证集
    num_val = int(len(X_train_full) * val_split)
    num_train = len(X_train_full) - num_val

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X_train_full))
    train_idx, val_idx = indices[:num_train], indices[num_train:]

    X_train = X_train_full[train_idx]
    y_train = y_train_full_oh[train_idx]
    y_train_labels = y_train_full[train_idx] 

    X_val = X_train_full[val_idx]
    y_val = y_train_full_oh[val_idx]
    y_val_labels = y_train_full[val_idx]

    return (X_train, y_train, y_train_labels), \
           (X_val, y_val, y_val_labels), \
           (X_test, y_test_oh, y_test)

class BatchIterator:
    """数据批处理迭代器，用于训练循环中高效获取 Mini-batch"""
    
    def __init__(self, x, y, batch_size=128, shuffle=True, seed=42):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
        self.num_samples = x.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / batch_size))
        self.reset()
    
    def reset(self):
        """每个 Epoch 开始前调用，重置索引并按需生成打乱顺序。
        使用索引数组而非直接复制数据，避免每次 Epoch 重复分配大块内存。"""
        self.current_batch = 0
        if self.shuffle:
            self.current_indices = self.rng.permutation(self.num_samples)
        else:
            self.current_indices = np.arange(self.num_samples)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            self.reset()
            raise StopIteration

        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)

        idx = self.current_indices[start_idx:end_idx]
        batch_x = self.x[idx]
        batch_y = self.y[idx]

        self.current_batch += 1
        return batch_x, batch_y