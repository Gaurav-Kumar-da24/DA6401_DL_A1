#utils/plotting.py for plotting_sample_images 
import numpy as np
import matplotlib.pyplot as plt
import wandb

def plot_sample_images(X_train, y_train):
    # Create a figure with a 2x5 grid
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Plot one image for each class
    for i in range(10):
        # Find the first image of this class
        idx = np.where(y_train == i)[0][0]
        image = X_train[idx].reshape(28, 28) if X_train.shape[1] == 784 else X_train[idx]
        # Display the image
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(class_names[i])
        axes[i].axis('off')
    
    plt.tight_layout()
    sample_images_path = 'sample_images.png'
    plt.savefig(sample_images_path)
    plt.close()
    
    # Log the image to wandb
    wandb.log({"sample_images": wandb.Image(sample_images_path)})