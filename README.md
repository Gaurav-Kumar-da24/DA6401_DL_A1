# Neural Network Implementation for Fashion MNIST Classification

## GitHub and Report
- Github: https://github.com/Gaurav-Kumar-da24/da6401_assignment1
- Report: https://wandb.ai/da24m006-iit-madras/DA6401_DL_A1/reports/DA6401-Assignment-1--VmlldzoxMTgzNDA1Mw?accessToken=le64p2nq4lhkuj25zo87r7vla25hy6wmosi3i69xbncqgtfm97sc071yn52g5lqi

This repository contains a custom implementation of a feedforward neural network for image classification on the Fashion MNIST dataset. The implementation includes various optimizers, activation functions, loss functions, and hyperparameter tuning capabilities using Weights & Biases (wandb).

## Project Structure

```
.
│   q1_sample_image.py
│   q4_sweep_cross_entropy.py
│   q8_sweep_mse_loss.py
│   README.md
│   train.py
│
├───data
│       data_loader.py
│       __init__.py
│
├───models
│       neural_network.py
│       __init__.py
│
├───optimizers
│       adam.py
│       momentum.py
│       nadam.py
│       nesterov.py
│       rmsprop.py
│       sgd.py
│       __init__.py
│
└───utils
        activations.py
        losses.py
        metrics.py
        plotting.py
        weight_init.py
        __init__.py
```

## Main Components

### Scripts

- **train.py**: Main training script for running neural network experiments with configurable parameters.
- **q1_sample_image.py**: Script to visualize sample images from the Fashion MNIST dataset.
- **q4_sweep_cross_entropy.py**: Hyperparameter sweep using Cross-Entropy loss function.
- **q8_sweep_mse_loss.py**: Hyperparameter sweep using Mean Squared Error loss function.

### Modules

#### Data
- **data_loader.py**: Functions to load and preprocess MNIST and Fashion MNIST datasets.

#### Models
- **neural_network.py**: Implementation of the feedforward neural network architecture.

#### Optimizers
- **sgd.py**: Stochastic Gradient Descent optimizer.
- **momentum.py**: Momentum optimizer.
- **nesterov.py**: Nesterov Accelerated Gradient optimizer.
- **rmsprop.py**: RMSProp optimizer.
- **adam.py**: Adam optimizer.
- **nadam.py**: Nesterov-accelerated Adam optimizer.

#### Utils
- **activations.py**: Activation functions (sigmoid, tanh, ReLU, identity).
- **losses.py**: Loss functions (mean squared error, cross-entropy).
- **metrics.py**: Performance metrics (accuracy).
- **plotting.py**: Visualization utilities.
- **weight_init.py**: Weight initialization methods (random, Xavier).

## Usage

### Basic Training

To train a neural network with default parameters:

```bash
python train.py
```

### Custom Training

To customize training parameters:

```bash
python train.py -d fashion_mnist -e 10 -b 64 -l cross_entropy -o adam -lr 0.001 -nhl 3 -sz 64 -a tanh -w_i Xavier
```

### Parameter Explanation

- `-d, --dataset`: Dataset to use (mnist or fashion_mnist)
- `-e, --epochs`: Number of training epochs
- `-b, --batch_size`: Batch size for training
- `-l, --loss`: Loss function (mean_squared_error or cross_entropy)
- `-o, --optimizer`: Optimizer type (sgd, momentum, nag, rmsprop, adam, nadam)
- `-lr, --learning_rate`: Learning rate
- `-nhl, --num_layers`: Number of hidden layers
- `-sz, --hidden_size`: Number of neurons in each hidden layer
- `-a, --activation`: Activation function (identity, sigmoid, tanh, ReLU)
- `-w_i, --weight_init`: Weight initialization method (random or Xavier)

### Hyperparameter Tuning

To run a hyperparameter sweep with cross-entropy loss:

```bash
python q4_sweep_cross_entropy.py -c 5
```

To run a hyperparameter sweep with MSE loss:

```bash
python q8_sweep_mse_loss.py -c 5
```

Where `-c` specifies the number of runs to execute in the sweep.

### Visualizing Sample Images

To visualize sample images from the Fashion MNIST dataset:

```bash
python q1_sample_image.py
```

## Dependencies

- NumPy
- TensorFlow (for dataset loading)
- Weights & Biases (for experiment tracking)

## Weights & Biases Integration

This project uses Weights & Biases for experiment tracking and hyperparameter optimization. Results, including loss curves, accuracy metrics, and confusion matrices, are automatically logged to the wandb dashboard.
