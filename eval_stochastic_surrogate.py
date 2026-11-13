"""
Attention: Minimal Working Example
Paper: A Reassessment of Global Weight-Space Sampling via Deterministic Flat Minima

Description:
Implementation of the Synthetic W-DRO Boundary Ablation (Algorithm 1). 
Isolates the 'Dimensionality Dilution' and 'Functional Disruption' phenomena 
by injecting isotropic Gaussian noise strictly bounded by the Wasserstein perturbation 
radius (rho) and scaling it via the multiplier (k).
"""

import argparse
import math
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

@torch.no_grad()
def inject_bounded_noise(model, rho, multiplier=1.0, is_asam=False):
    """
    Injects isotropic Gaussian noise bounded by the Wasserstein perturbation radius.
    Saves original weights to allow restoration after the forward pass.
    """
    global_trace = 0.0
    original_weights = {}

    # Step 1: Calculate Global Trace
    for name, p in model.named_parameters():
        t_operator = p.abs() if is_asam else torch.ones_like(p)
        global_trace += torch.sum(t_operator ** 2).item()
        
    beta = (rho ** 2) / global_trace
    effective_beta = beta * (multiplier ** 2)

    # Step 2: Inject Noise
    for name, p in model.named_parameters():
        original_weights[name] = p.data.clone()
        t_operator = p.abs() if is_asam else torch.ones_like(p)
        noise = torch.randn_like(p) * math.sqrt(effective_beta) * t_operator
        p.add_(noise)
        
    return original_weights

@torch.no_grad()
def restore_weights(model, original_weights):
    for name, p in model.named_parameters():
        p.data.copy_(original_weights[name])

def main():
    parser = argparse.ArgumentParser(description="Diagnostic Stochastic Surrogate (Scale Sweep)")
    parser.add_argument('--model', type=str, choices=['resnet50', 'vit_b_16'], required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data/imagenet/val')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--rho', type=float, required=True, help="Base Wasserstein radius used during training")
    parser.add_argument('--multiplier', type=float, default=10.0, help="Noise multiplier (k) for dimensionality dilution tests")
    parser.add_argument('--samples', type=int, default=10, help="Number of stochastic forward passes (N)")
    parser.add_argument('--adaptive', action='store_true', help="Set if base model was trained with ASAM")
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

    print(f"Running Stochastic Surrogate | Multiplier: {args.multiplier}x | Samples: {args.samples}")

    total_acc, total_entropy, count = 0.0, 0.0, 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_probs = torch.zeros((inputs.size(0), 1000), device=device)

            # N stochastic forward passes per batch
            for _ in range(args.samples):
                orig_weights = inject_bounded_noise(model, args.rho, args.multiplier, args.adaptive)
                
                logits = model(inputs)
                batch_probs += F.softmax(logits, dim=1)
                
                restore_weights(model, orig_weights)

            batch_probs /= args.samples
            
            preds = torch.argmax(batch_probs, dim=1)
            total_acc += (preds == targets).sum().item()
            total_entropy += -torch.sum(batch_probs * torch.log2(batch_probs + 1e-12), dim=1).sum().item() # Shannon Entropy (in bits)
            count += inputs.size(0)

    print(f"Results (Scale {args.multiplier}x) -> Accuracy: {total_acc/count:.4f} | Mean Entropy: {total_entropy/count:.4f}")

if __name__ == "__main__":
    main()
