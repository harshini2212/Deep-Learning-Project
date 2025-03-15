import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import pickle
from tqdm import tqdm
import argparse

# Import our model architecture
from resnet_model import resnet18_compact, resnet34_compact, resnet18, resnet34, resnet50

print("Script starting...")
import sys
sys.stdout.flush()  # Force output to flush

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data augmentation and normalization
def get_transforms(train=True):
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)
    
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
            transforms.RandomErasing(p=0.2),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])

# Load CIFAR-10 datasets
def load_cifar10(batch_size=128):
    transform_train = get_transforms(train=True)
    transform_test = get_transforms(train=False)
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader, train_dataset.classes

# LabelSmoothingLoss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

# Mixup augmentation
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Training function
def train(model, train_loader, optimizer, criterion, epoch, mixup_alpha=0.2):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply mixup
        if mixup_alpha > 0:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        
        # For mixup, we don't have a clear prediction
        if mixup_alpha > 0:
            total += targets.size(0)
            # Approximate accuracy for progress bar
            correct += (lam * predicted.eq(targets_a).sum().float() + 
                       (1 - lam) * predicted.eq(targets_b).sum().float())
        else:
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': train_loss/(batch_idx+1), 
            'acc': 100.*correct/total
        })
    
    return train_loss/len(train_loader), 100.*correct/total

# Evaluation function
def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss/len(test_loader), 100.*correct/total

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data
    train_loader, test_loader, classes = load_cifar10(args.batch_size)
    print(f"Classes: {classes}")
    
    # Create model
    if args.model == 'resnet18_compact':
        model = resnet18_compact(drop_rate=args.dropout)
    elif args.model == 'resnet34_compact':
        model = resnet34_compact(drop_rate=args.dropout)
    elif args.model == 'resnet18':
        model = resnet18(initial_channels=args.initial_channels, drop_rate=args.dropout)
    elif args.model == 'resnet34':
        model = resnet34(initial_channels=args.initial_channels, drop_rate=args.dropout)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    
    # Print total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Define loss function and optimizer
    criterion = LabelSmoothingLoss(classes=10, smoothing=args.label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                         momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_acc = 0
    start_time = time.time()
    
    # Lists to store metrics
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(args.epochs):
        # Mixup strength decreases over epochs
        mixup_alpha = max(0, args.mixup * (1 - epoch / args.epochs))
        
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch, mixup_alpha)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        
        scheduler.step()
        
        # Save metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # Save best model
        if test_acc > best_acc:
            print(f'Saving best model: {test_acc:.2f}%')
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
        
        print(f'Epoch: {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    training_time = time.time() - start_time
    print(f'Training completed in {training_time/60:.2f} minutes')
    print(f'Best test accuracy: {best_acc:.2f}%')
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'test_loss': test_losses,
        'train_acc': train_accs,
        'test_acc': test_accs,
    }
    
    np.save(os.path.join(args.output_dir, 'history.npy'), history)
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
    _, final_acc = evaluate(model, test_loader, criterion)
    print(f'Final test accuracy: {final_acc:.2f}%')

if __name__ == '__main__':
    print("Parsing arguments...")
    sys.stdout.flush()
    
    try:
        parser = argparse.ArgumentParser(description='Train a modified ResNet on CIFAR-10')
        parser.add_argument('--model', type=str, default='resnet18_compact', 
                            choices=['resnet18_compact', 'resnet34_compact', 'resnet18', 'resnet34'],
                            help='Model architecture')
        parser.add_argument('--initial_channels', type=int, default=48, 
                            help='Initial channels for non-compact models')
        parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
        parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
        parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
        parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
        parser.add_argument('--mixup', type=float, default=0.2, help='Mixup alpha')
        parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
        parser.add_argument('--seed', type=int, default=42, help='Random seed')
        parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
        args = parser.parse_args()
        print(f"Arguments: {args}")
        sys.stdout.flush()
        
        main(args)

    except Exception as e:
        print(f"ERROR OCCURRED: {e}")
        import traceback
        traceback.print_exc()