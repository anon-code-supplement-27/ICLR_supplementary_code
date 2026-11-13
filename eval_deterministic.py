"""
Attention: Minimal Working Example
Paper: A Reassessment of Global Weight-Space Sampling via Deterministic Flat Minima

Description:
Standard, no-overhead deterministic evaluation (M=0). 
Evaluates in-distribution (ID) and out-of-distribution (OOD) performance.
"""

import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

def compute_metrics(logits, targets):
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    acc = (preds == targets).float().mean().item()
    entropy = -torch.sum(probs * torch.log2(probs + 1e-12), dim=1).mean().item() # Shannon Entropy (in bits)
    return acc, entropy

def main():
    parser = argparse.ArgumentParser(description="Evaluate Deterministic Baseline")
    parser.add_argument('--model', type=str, choices=['resnet50', 'vit_b_16'], required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data/imagenet/val')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = torchvision.datasets.ImageFolder(root=args.data_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    if args.model == 'resnet50':
        model = torchvision.models.resnet50()
    else:
        model = torchvision.models.vit_b_16()
        
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    model.eval()

    all_logits, all_targets = [], []

    print(f"Evaluating {args.model} on {args.data_dir}...")
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            all_logits.append(logits)
            all_targets.append(targets)

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    acc, entropy = compute_metrics(all_logits, all_targets)
    print(f"Results -> Accuracy: {acc:.4f} | Mean Entropy: {entropy:.4f}")

if __name__ == "__main__":
    main()
