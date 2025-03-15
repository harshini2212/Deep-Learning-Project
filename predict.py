import os
import torch
import torchvision.transforms as transforms
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
import argparse

# Import our model
from resnet_model import resnet18_compact, resnet34_compact, resnet18, resnet34

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transforms for testing
def get_test_transform():
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)
    
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

# Custom dataset for test images
class CIFARTestDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, idx

# Load the unlabeled test dataset
def load_test_data(test_path, batch_size=128):
    transform_test = get_test_transform()
    
    # Load test data from pickle file
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    
    # Extract test images
    test_images = test_data[b'data']
    
    test_dataset = CIFARTestDataset(test_images, transform=transform_test)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return test_loader

# Predict using a single model
def predict_single(model_path, model_type, test_loader, initial_channels=48):
    # Initialize model
    if model_type == 'resnet18_compact':
        model = resnet18_compact(drop_rate=0.0)  # No dropout for inference
    elif model_type == 'resnet34_compact':
        model = resnet34_compact(drop_rate=0.0)
    elif model_type == 'resnet18':
        model = resnet18(initial_channels=initial_channels, drop_rate=0.0)
    elif model_type == 'resnet34':
        model = resnet34(initial_channels=initial_channels, drop_rate=0.0)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Load the best model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Make predictions
    predictions = []
    indices = []
    
    with torch.no_grad():
        for inputs, idx in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            predictions.extend(preds.cpu().numpy())
            indices.extend(idx.numpy())
    
    # Sort by index to maintain original order
    predictions = [p for _, p in sorted(zip(indices, predictions))]
    
    return predictions

# Predict using an ensemble of models
def predict_ensemble(model_paths, model_types, test_loader, initial_channels=48):
    all_probs = []
    indices = []
    
    # Get predictions from each model
    for i, (model_path, model_type) in enumerate(zip(model_paths, model_types)):
        # Initialize model
        if model_type == 'resnet18_compact':
            model = resnet18_compact(drop_rate=0.0)
        elif model_type == 'resnet34_compact':
            model = resnet34_compact(drop_rate=0.0)
        elif model_type == 'resnet18':
            model = resnet18(initial_channels=initial_channels, drop_rate=0.0)
        elif model_type == 'resnet34':
            model = resnet34(initial_channels=initial_channels, drop_rate=0.0)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model.to(device)
        
        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        model_probs = []
        model_indices = []
        
        with torch.no_grad():
            for inputs, idx in tqdm(test_loader, desc=f"Evaluating model {i+1}"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                model_probs.append(probs.cpu())
                model_indices.append(idx)
        
        # Concatenate all batch predictions
        all_probs.append(torch.cat(model_probs))
        if not indices:
            indices = torch.cat(model_indices).numpy()
    
    # Average predictions from all models
    ensemble_probs = sum(all_probs) / len(model_paths)
    
    # Get the class with highest probability
    _, predictions = ensemble_probs.max(1)
    predictions = predictions.numpy()
    
    # Sort by index to maintain original order
    predictions = [p for _, p in sorted(zip(indices, predictions))]
    
    return predictions

# Save predictions to a CSV file
def save_predictions(predictions, output_path):
    submission = pd.DataFrame({
        'ID': range(len(predictions)),
        'Labels': predictions
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Created submission file with {len(predictions)} predictions")

def main(args):
    # Load test data
    test_loader = load_test_data(args.test_path, args.batch_size)
    
    if args.ensemble:
        # Use ensemble prediction
        model_paths = args.model_paths.split(',')
        model_types = args.model_types.split(',')
        
        if len(model_paths) != len(model_types):
            raise ValueError("Number of model paths must match number of model types")
        
        predictions = predict_ensemble(model_paths, model_types, test_loader, args.initial_channels)
    else:
        # Use single model prediction
        predictions = predict_single(args.model_path, args.model_type, test_loader, args.initial_channels)
    
    # Save predictions
    save_predictions(predictions, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate predictions for CIFAR-10 test set')
    parser.add_argument('--test_path', type=str, required=True, 
                        help='Path to the test data pickle file')
    parser.add_argument('--output_path', type=str, default='submission.csv', 
                        help='Path to save the submission file')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size for prediction')
    
    # Single model arguments
    parser.add_argument('--model_path', type=str, default='./results/best_model.pth', 
                        help='Path to the trained model')
    parser.add_argument('--model_type', type=str, default='resnet18_compact', 
                        choices=['resnet18_compact', 'resnet34_compact', 'resnet18', 'resnet34'],
                        help='Type of model architecture')
    
    # Ensemble arguments
    parser.add_argument('--ensemble', action='store_true', 
                        help='Use ensemble of models for prediction')
    parser.add_argument('--model_paths', type=str, 
                        help='Comma-separated list of paths to trained models')
    parser.add_argument('--model_types', type=str, 
                        help='Comma-separated list of model types corresponding to model_paths')
    
    parser.add_argument('--initial_channels', type=int, default=48, 
                        help='Initial channels for non-compact models')
    
    args = parser.parse_args()
    main(args)