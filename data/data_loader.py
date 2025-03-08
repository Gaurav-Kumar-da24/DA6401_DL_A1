# data/data_loader.py
import numpy as np
from tensorflow.keras.datasets import fashion_mnist, mnist

def load_data(dataset_name):
    """
    Load the specified dataset
    Args:
        dataset_name (str): Name of the dataset to load ('mnist' or 'fashion_mnist')
    
    Returns:
        tuple: (X_train, y_train), (X_test, y_test)
    """
    if dataset_name == 'fashion_mnist':
        return fashion_mnist.load_data()
    elif dataset_name == 'mnist':
        return mnist.load_data()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def preprocess_data(X_train, y_train, X_test, y_test, val_split=0.1):
    """
    Preprocess the data: normalize, reshape, one-hot encode, and split train/val
    
    Args:
        X_train (ndarray): Training images
        y_train (ndarray): Training labels
        X_test (ndarray): Test images
        y_test (ndarray): Test labels
        val_split (float): Proportion of training data to use for validation
    
    Returns:
        tuple: X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Reshape and normalize images
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
    
    # One-hot encode labels
    num_classes = 10  # Both MNIST and Fashion-MNIST have 10 classes
    y_train_onehot = np.zeros((y_train.shape[0], num_classes))
    y_train_onehot[np.arange(y_train.shape[0]), y_train] = 1
    
    y_test_onehot = np.zeros((y_test.shape[0], num_classes))
    y_test_onehot[np.arange(y_test.shape[0]), y_test] = 1
    
    # Split training data into train and validation sets
    val_size = int(X_train.shape[0] * val_split)
    X_val = X_train[:val_size]
    y_val = y_train_onehot[:val_size]
    X_train = X_train[val_size:]
    y_train = y_train_onehot[val_size:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test_onehot
