import argparse
import wandb

# sweep configuration for hyperparameter search
def get_sweep_config():
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
        'parameters': {
            'epochs': {'values': [5, 10]},
            'num_layers': {'values': [3, 4, 5]},
            'hidden_size': {'values': [32, 64, 128]},
            'weight_decay': {'values': [0, 0.0005, 0.5]},
            'learning_rate': {'values': [1e-3, 1e-4]},
            'optimizer': {'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
            'batch_size': {'values': [16, 32, 64]},
            'weight_init': {'values': ['random', 'Xavier']},
            'activation': {'values': ['sigmoid', 'tanh', 'ReLU']}
        }
    }
    
    return sweep_config

def parse_args():
    parser = argparse.ArgumentParser(description='Run wandb sweep for hyperparameter tuning')
    parser.add_argument('-wp', '--wandb_project', type=str, default='DA6401_DL_A1', help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', type=str, default='da24m006-iit-madras', help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    parser.add_argument('-c', '--count', type=int, default=5, help='Number of runs to execute')
    return parser.parse_args()

def train_function():
    """Function that runs a single training job with hyperparameters from sweep."""
    run = wandb.init()
    
    # Set the name of the run to have Meaningful names for each sweep (e.g. hl_3_bs_16_ac_tanh to indicate that there were 3 hidden layers, batch size was 16 and activation function was ReLU).
    run.name = f"hl_{wandb.config.num_layers}_bs_{wandb.config.batch_size}_ac_{wandb.config.activation}_{wandb.config.optimizer}_loss_CE"
    
    config = wandb.config

    # Import dependencies here to keep them within the function scope
    import numpy as np
    from data.data_loader import load_data, preprocess_data
    from models.neural_network import FeedForwardNN
    from utils.losses import get_loss_function
    from utils.activations import get_activation_function
    from utils.metrics import compute_accuracy
    from optimizers import get_optimizer
    
    # Set default dataset if not specified in sweep config
    dataset = getattr(config, 'dataset', 'fashion_mnist')
    
    # Load and preprocess data
    (X_train, y_train), (X_test, y_test) = load_data(dataset)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test, val_split=0.1)
    
    # Get the loss function 
    loss_type = getattr(config, 'loss', 'cross_entropy')
    loss_fn, loss_derivative = get_loss_function(loss_type)

    activation_fn, activation_derivative = get_activation_function(config.activation)
    
    input_dim = X_train.shape[1]
    output_dim = 10  
    
    # Create neural network
    hidden_sizes = [config.hidden_size] * config.num_layers
    model = FeedForwardNN(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        output_dim=output_dim,
        activation_fn=activation_fn,
        activation_derivative=activation_derivative,
        weight_init=config.weight_init
    )
    
    # Get optimizer
    optimizer = get_optimizer(
        optimizer_name=config.optimizer,
        learning_rate=config.learning_rate,
        momentum=getattr(config, 'momentum', 0.5),
        beta=getattr(config, 'beta', 0.5),
        beta1=getattr(config, 'beta1', 0.5),
        beta2=getattr(config, 'beta2', 0.5),
        epsilon=getattr(config, 'epsilon', 0.000001),
        weight_decay=getattr(config, 'weight_decay', 0.0)
    )
    
    # Training loop
    num_samples = X_train.shape[0]
    num_batches = int(np.ceil(num_samples / config.batch_size))
    
    for epoch in range(config.epochs):
        indices = np.random.permutation(num_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        epoch_loss = 0
        
        for batch in range(num_batches):
            start_idx = batch * config.batch_size
            end_idx = min((batch + 1) * config.batch_size, num_samples)
            
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
        
        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    # Evaluate on test set
    y_pred_test = model.forward(X_test)
    test_accuracy = compute_accuracy(y_pred_test, y_test)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    wandb.log({'test_accuracy': test_accuracy})
        
    # Confusion Matrix plot using wand and loging
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # Convert one-hot encoded vectors to class indices
    y_pred_test_ids = np.argmax(y_pred_test, axis=1)
    y_test_ids = np.argmax(y_test, axis=1)
    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(y_true=y_test_ids, preds=y_pred_test_ids,class_names=class_names)})



def main():
    args = parse_args()
    sweep_config = get_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)
    wandb.agent(sweep_id, function=train_function, count=args.count)

if __name__ == "__main__":
    main()