import argparse
from pathlib import Path

import torch
from torchvision import transforms

from DatasetLoader import *
from gan_modules import Generator, Discriminator
from GAN import PGGAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():   
    """
    CLI argument parser 
    Args: CLI defined arguments
    """

    parser = argparse.ArgumentParser(
        "Train PGGAN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Arguments that require inputs
    parser.add_argument("train_path", action="store", type=Path,
                        help="Define path for training images")
    parser.add_argument("output", action="store", type=Path,
                        help="Path to store model and checkpoints")
                    
    # Arguments that can be modified
    parser.add_argument("--depth", action="store", type=int, default=7, required=False,
                        help="Used to determine final resoluton of output image." 
                        "Starts at depth 2 -> (4x4)")
    parser.add_argument("--num_channels", action="store", type=int, default=128, required=False,
                        help="number of channels in image data. 3 -> RBG, 1 -> Grayscale")
    parser.add_argument("--latent_size", action="store", type=int, default=128, required=False,
                        help="latent size of discriminator and generator")
    parser.add_argument("--num_classes", action="store", type=int, default=7, required=False,
                        help="Defining number of classes present in the given dataset")
                
    # Training parameters
    parser.add_argument("--epochs", action="store", type=int, required=False, nargs='+',
                        default=[10 for _ in range(6)])
    parser.add_argument("--g_lr", action="store", type=float, required=False, default=0.003,
                        help="learning rate used by the generator")
    parser.add_argument("--fade_in", action="store", type=int, required=False, nargs='+',
                        default=[50 for _ in range(6)],
                        help="percentage for fade in for each defined layer")
    parser.add_argument("--d_lr", action="store", type=float, required=False, default=0.003,
                        help="learning rate used by the discriminator")
    parser.add_argument("--num_workers", action="store", type=int, required=False, default=4,
                        help="Define number of workers for training")
    parser.add_argument("--start_depth", action="store", type=int, required=False, default=1,
                        help="Determines what resolution depth the images will start at")
    parser.add_argument("--checkpoint_factor", action="store", type=int, required=False, default=5,
                        help="number of epochs after which a model snapshot is saved per training stage")

    parsed_args = parser.parse_args()
    return parsed_args

def trainPGGAN(args):
    """
    Function to train PGGAN using parsed arguments
    Args:
       args: config defined from CLI 
    Returns:
        None
    """
    print(f"Defined arguments: {args}")

    generator = Generator(
        depth=args.depth,
        in_channels=args.num_channels,
        latent_size=args.latent_size
    )

    discriminator = Discriminator(
        #depth=args.depth,
        #in_channels=args.num_channels,
        latent_size=args.latent_size
    )

    pggan = PGGAN(
        generator,
        discriminator,
        device=device
    )

    loader = image_loader(args.train_path)

    pggan.train(
        loader=loader,
        epochs=args.epochs,
        gen_lr=args.g_lr,
        dis_lr=args.d_lr,
        start_depth=args.start_depth,
        save_dir=args.output,
        fade_in=args.fade_in,
        checkpoint_factor=args.checkpoint_factor
    )

def main():
    trainPGGAN(parse_args())

if __name__ == "__main__":
    main()