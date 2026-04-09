"""
Утилиты для загрузки стандартных датасетов
"""

import numpy as np
import urllib.request
import gzip
import os
from typing import Tuple
from urllib.error import HTTPError, URLError
from .dataset import ArrayDataset, Dataset


def load_mnist(data_path: str | None = None) -> Tuple[Dataset, Dataset]:
    """
    Загружает датасет MNIST

    Returns:
        train_dataset, test_dataset
    """
    if data_path is None:
        data_path = os.getenv("DATA_DIR", "./datasets")
    os.makedirs(data_path, exist_ok=True)

    # URLs для MNIST (основной источник + зеркала)
    url_candidates = {
        'train_images': [
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
        ],
        'train_labels': [
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
        ],
        'test_images': [
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
        ],
        'test_labels': [
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
            'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
        ],
    }

    def download_file(urls, filename):
        filepath = os.path.join(data_path, filename)
        if os.path.exists(filepath):
            return filepath

        last_error = None
        for url in urls:
            try:
                print(f"Downloading {filename} from {url}...")
                request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    with open(filepath, 'wb') as file_obj:
                        file_obj.write(response.read())
                return filepath
            except (URLError, HTTPError, TimeoutError, OSError) as error:
                last_error = error

        if last_error is not None:
            raise RuntimeError(f"Failed to download {filename}: {last_error}") from last_error
        return filepath

    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 784).astype(np.float32) / 255.0

    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data.astype(np.int64)

    try:
        # Загрузка данных
        train_images = load_images(download_file(url_candidates['train_images'], 'train-images-idx3-ubyte.gz'))
        train_labels = load_labels(download_file(url_candidates['train_labels'], 'train-labels-idx1-ubyte.gz'))
        test_images = load_images(download_file(url_candidates['test_images'], 't10k-images-idx3-ubyte.gz'))
        test_labels = load_labels(download_file(url_candidates['test_labels'], 't10k-labels-idx1-ubyte.gz'))

        train_dataset = ArrayDataset(train_images, train_labels)
        test_dataset = ArrayDataset(test_images, test_labels)
        return train_dataset, test_dataset
    except Exception:
        # Offline fallback: встроенный датасет sklearn digits (8x8 -> 64 признака)
        from sklearn.datasets import load_digits

        digits = load_digits()
        X = digits.data.astype(np.float32) / 16.0
        y = digits.target.astype(np.int64)

        split_idx = int(0.8 * len(X))
        train_dataset = ArrayDataset(X[:split_idx], y[:split_idx])
        test_dataset = ArrayDataset(X[split_idx:], y[split_idx:])
        return train_dataset, test_dataset


def load_iris() -> Dataset:
    """
    Загружает датасет Iris
    """
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.int64)

    # Нормализация
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    return ArrayDataset(X, y)


def load_california_housing() -> Dataset:
    """
    Загружает датасет California Housing для регрессии
    """
    from sklearn.datasets import fetch_california_housing

    housing = fetch_california_housing()
    X = housing.data.astype(np.float32)
    y = housing.target.astype(np.float32).reshape(-1, 1)

    # Нормализация
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)

    return ArrayDataset(X, y)


def create_train_val_split(
        dataset: Dataset,
        val_ratio: float = 0.2,
        shuffle: bool = True
) -> Tuple[Dataset, Dataset]:
    """
    Разделяет датасет на тренировочный и валидационный
    """
    splits = dataset.split([1 - val_ratio, val_ratio], shuffle=shuffle)
    return splits[0], splits[1]