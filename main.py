"""
Eviformer: An uncertainty fault diagnosis framework guided by evidential deep learning.

Main entry point for training and evaluation.

Usage:
    # Train with evidential digamma loss
    python main.py --train --uncertainty --digamma --epochs 100

    # Train with standard cross-entropy loss
    python main.py --train --epochs 100

    # Train with evidential MSE loss
    python main.py --train --uncertainty --mse --epochs 100

    # Train with evidential log loss
    python main.py --train --uncertainty --log --epochs 100
"""

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from utils import get_device
from models.losses import edl_mse_loss, edl_digamma_loss, edl_log_loss
from models.lenet import LeNet
from models.vit import VisionTransformer1D
from models.mcswint import MCSwinT
from datasets.cwru import load_cwru_dataset
from train import train_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eviformer: Uncertainty fault diagnosis with evidential deep learning"
    )

    # Mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Train the model.")
    mode_group.add_argument("--test", action="store_true", help="Test the model.")

    # Model selection
    parser.add_argument(
        "--model", type=str, default="mcswint",
        choices=["lenet", "vit", "mcswint"],
        help="Model architecture to use (default: mcswint)."
    )

    # Training hyperparameters
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--weight_decay", default=0.005, type=float, help="Weight decay.")
    parser.add_argument("--num_classes", default=10, type=int, help="Number of fault classes.")
    parser.add_argument("--dropout", action="store_true", help="Enable dropout (LeNet only).")

    # Uncertainty / EDL
    parser.add_argument("--uncertainty", action="store_true", help="Use evidential deep learning.")
    uncertainty_type = parser.add_mutually_exclusive_group()
    uncertainty_type.add_argument("--mse", action="store_true", help="EDL MSE loss.")
    uncertainty_type.add_argument("--digamma", action="store_true", help="EDL Digamma loss.")
    uncertainty_type.add_argument("--log", action="store_true", help="EDL Log loss.")

    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="Path to CWRU dataset root.")
    parser.add_argument("--normalize", type=str, default="0-1", choices=["0-1", "-1-1", "mean-std"],
                        help="Normalization type.")

    # Output
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save model checkpoints.")

    return parser.parse_args()


def build_model(args):
    """Build model based on command-line arguments."""
    if args.model == "lenet":
        model = LeNet(num_classes=args.num_classes, dropout=args.dropout)
    elif args.model == "vit":
        model = VisionTransformer1D(
            data_size=1024, in_c=1, num_classes=args.num_classes,
            patch_size=32, overlap=30, depth=3, num_heads=4,
            h_args=[256, 128, 64, 32]
        )
    elif args.model == "mcswint":
        model = MCSwinT(in_c=1, num_classes=args.num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    return model


def get_criterion(args, parser):
    """Get loss function based on arguments."""
    if args.uncertainty:
        if args.digamma:
            return edl_digamma_loss
        elif args.log:
            return edl_log_loss
        elif args.mse:
            return edl_mse_loss
        else:
            parser.error("--uncertainty requires --mse, --log, or --digamma.")
    else:
        return nn.CrossEntropyLoss()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    # Load data
    dataloaders = load_cwru_dataset(
        data_dir=args.data_dir,
        normalize_type=args.normalize,
        batch_size=args.batch_size,
    )

    if args.train:
        # Build model
        model = build_model(args).to(device)
        print(f"Model: {args.model} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Loss and optimizer
        criterion = get_criterion(args, argparse.ArgumentParser())
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

        # Train
        model, metrics = train_model(
            model, dataloaders, args.num_classes, criterion, optimizer,
            scheduler=scheduler, num_epochs=args.epochs,
            device=device, uncertainty=args.uncertainty,
        )

        # Save checkpoint
        state = {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        }

        loss_name = "digamma" if args.digamma else ("log" if args.log else ("mse" if args.mse else "ce"))
        save_path = os.path.join(args.save_dir, f"model_{args.model}_{loss_name}.pt")
        torch.save(state, save_path)
        print(f"Model saved to: {save_path}")

    elif args.test:
        # Load checkpoint
        loss_name = "digamma" if args.digamma else ("log" if args.log else ("mse" if args.mse else "ce"))
        ckpt_path = os.path.join(args.save_dir, f"model_{args.model}_{loss_name}.pt")

        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}")
            return

        model = build_model(args).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"Loaded checkpoint from: {ckpt_path}")
        print("Model ready for inference.")


if __name__ == "__main__":
    main()
