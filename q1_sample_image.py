import argparse
import wandb
from data.data_loader import load_data
from utils.plotting import plot_sample_images # code for plotting sample images

def parse_args():
    parser = argparse.ArgumentParser(description="Run a neural network training experiment with Weights & Biases.")

    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401_DL_A1",
                        help="Project name used to track experiments in Weights & Biases dashboard.")
    
    parser.add_argument("-we", "--wandb_entity", type=str, default="da24m006-iit-madras",
                        help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")

    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist",
                        help="Dataset to use for training. Choices: ['mnist', 'fashion_mnist'].")

    return parser.parse_args()


def main():
    args = parse_args()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    (X_train, y_train), (X_test, y_test) = load_data(args.dataset)
    # Question 1: Plot sample images for each class 
    if args.dataset == 'fashion_mnist':
        plot_sample_images(X_train, y_train) 
    wandb.finish()

if __name__ == "__main__":
    main()
