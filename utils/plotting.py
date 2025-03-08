# utils/plotting.py
import matplotlib.pyplot as plt
import numpy as np
import wandb

def plot_sample_images(X_train, y_train):
    """
    Plot and log one sample image for each class
    
    Args:
        X_train (ndarray): Training images
        y_train (ndarray): Training labels
    """
    # Create a figure with a 2x5 grid
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    # Fashion MNIST class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Plot one image for each class
    for i in range(10):
        # Find the first image of this class
        idx = np.where(y_train == i)[0][0]
        
        # Display the image
        axes[i].imshow(X_train[idx], cmap='gray')
        axes[i].set_title(class_names[i])
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Log to wandb
    wandb.log({"examples": wandb.Image(fig)})
    
    plt.close()
