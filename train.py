# train.py 
import argparse
import os
import numpy as np
import wandb
from data.data_loader import load_data, preprocess_data
from models.neural_network import FeedForwardNN
from utils.losses import get_loss_function
from utils.activations import get_activation_function
from utils.metrics import compute_accuracy
from optimizers import get_optimizer

def parse_args():
    parser = argparse.ArgumentParser(description="Run a neural network training experiment with Weights & Biases.")

    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401_DL_A1",
                        help="Project name used to track experiments in Weights & Biases dashboard.")
    
    parser.add_argument("-we", "--wandb_entity", type=str, default="da24m006-iit-madras",
                        help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")

    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist",
                        help="Dataset to use for training. Choices: ['mnist', 'fashion_mnist'].")

    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Number of epochs to train the neural network.")

    parser.add_argument("-b", "--batch_size", type=int, default=64,
                        help="Batch size used to train the neural network.")

    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy",
                        help="Loss function to use. Choices: ['mean_squared_error', 'cross_entropy'].")

    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="adam",
                        help="Optimizer to use. Choices: ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'].")

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                        help="Learning rate used to optimize model parameters.")

    parser.add_argument("-m", "--momentum", type=float, default=0.5,
                        help="Momentum used by 'momentum' and 'nag' optimizers.")

    parser.add_argument("-beta", "--beta", type=float, default=0.5,
                        help="Beta used by 'rmsprop' optimizer.")

    parser.add_argument("-beta1", "--beta1", type=float, default=0.5,
                        help="Beta1 used by 'adam' and 'nadam' optimizers.")

    parser.add_argument("-beta2", "--beta2", type=float, default=0.5,
                        help="Beta2 used by 'adam' and 'nadam' optimizers.")

    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001,
                        help="Epsilon used by optimizers.")

    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0,
                        help="Weight decay used by optimizers.")

    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="Xavier",
                        help="Weight initialization method. Choices: ['random', 'Xavier'].")

    parser.add_argument("-nhl", "--num_layers", type=int, default=3,
                        help="Number of hidden layers used in the feedforward neural network.")

    parser.add_argument("-sz", "--hidden_size", type=int, default=64,
                        help="Number of hidden neurons in each feedforward layer.")

    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="tanh",
                        help="Activation function to use. Choices: ['identity', 'sigmoid', 'tanh', 'ReLU'].")

    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initialize wandb
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    
    # Load and preprocess data
    (X_train, y_train), (X_test, y_test) = load_data(args.dataset)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test, val_split=0.1)
    
    # loss and  activation function to be used 
    loss_fn, loss_derivative = get_loss_function(args.loss)
    activation_fn, activation_derivative = get_activation_function(args.activation)
    
    # Define input and output dimensions
    input_dim = X_train.shape[1]
    output_dim = 10  # For MNIST and Fashion-MNIST
    
    # Create neural network
    hidden_sizes = [args.hidden_size] * args.num_layers
    model = FeedForwardNN(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        output_dim=output_dim,
        activation_fn=activation_fn,
        activation_derivative=activation_derivative,
        weight_init=args.weight_init
    )
    
    # Get optimizer
    optimizer = get_optimizer(
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        beta=args.beta,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    num_samples = X_train.shape[0]
    num_batches = int(np.ceil(num_samples / args.batch_size))
    
    for epoch in range(args.epochs):
        # Shuffle data
        indices = np.random.permutation(num_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        epoch_loss = 0
        
        for batch in range(num_batches):
            start_idx = batch * args.batch_size
            end_idx = min((batch + 1) * args.batch_size, num_samples)
            
            # Get batch data
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # Forward pass
            y_pred = model.forward(X_batch)
            
            # Compute loss
            batch_loss = loss_fn(y_pred, y_batch)
            epoch_loss += batch_loss
            
            # Compute gradients
            d_loss = loss_derivative(y_pred, y_batch)
            
            # Backward pass
            model.backward(d_loss)
            
            # Update weights
            optimizer.update(model)
        
        # Compute metrics on validation set
        y_pred_val = model.forward(X_val)
        val_loss = loss_fn(y_pred_val, y_val)
        val_accuracy = compute_accuracy(y_pred_val, y_val)
        
        # Compute metrics on training set
        y_pred_train = model.forward(X_train)
        train_loss = loss_fn(y_pred_train, y_train)
        train_accuracy = compute_accuracy(y_pred_train, y_train)
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'loss': train_loss,
            'accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })
        
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    # Evaluate on test set
    y_pred_test = model.forward(X_test)
    test_accuracy = compute_accuracy(y_pred_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    wandb.log({'test_accuracy': test_accuracy})
    
    # Generate and log confusion matrix
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    y_pred_test_ids = np.argmax(y_pred_test, axis=1)
    y_test_ids = np.argmax(y_test, axis=1)
    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(y_true=y_test_ids, preds=y_pred_test_ids,class_names=class_names)})

    wandb.finish()

if __name__ == "__main__":
    main()