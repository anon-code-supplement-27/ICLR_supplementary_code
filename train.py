"""
Attention: Minimal Working Example
Paper: A Reassessment of Global Weight-Space Sampling via Deterministic Flat Minimavia Deterministic Flat Minima

Description:
Training loop to fine-tune a deterministic base model using SAM or ASAM.
"""

import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sam import SAM

def main():
    parser = argparse.ArgumentParser(description="Deterministic Flat-Minima Training")
    parser.add_argument('--model', type=str, choices=['resnet50', 'vit_b_16'], required=True)
    parser.add_argument('--data_dir', type=str, default='./data/imagenet')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--rho', type=float, default=0.5, help="Wasserstein radius")
    parser.add_argument('--adaptive', action='store_true', help="Use ASAM instead of SAM")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=f"{args.data_dir}/train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # Load Model
    if args.model == 'resnet50':
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    else:
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
    
    model = model.to(device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    
    # Base optimizer (SGD for ResNet, AdamW for ViT)
    if args.model == 'resnet50':
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.lr, momentum=0.9)
    else:
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=args.epochs)

    print(f"Starting {args.epochs}-epoch fine-tuning for {args.model} using {'ASAM' if args.adaptive else 'SAM'} (rho={args.rho})...")

    # Training Loop
    for epoch in range(args.epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Pass 1: Extract geometry and step to adversarial boundary
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # Pass 2: Calculate robust loss and apply actual update
            criterion(model(inputs), targets).backward()
            optimizer.second_step(zero_grad=True)

        scheduler.step()
        print(f"Epoch [{epoch+1}/{args.epochs}] completed.")

    torch.save(model.state_dict(), f"deterministic_{args.model}_rho{args.rho}.pth")
    print("Training complete. Checkpoint saved.")

if __name__ == "__main__":
    main()
