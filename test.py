import torch
from torch.utils.data import DataLoader

from models import ParticleNet, ParticleNetLite
from data_loader import get_tt_dataloaders, TopTaggingDataset, get_qg_dataloader, ChunkedQGDataset
from data_prep import read_h5_dataset

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

def evaluate_model(model, test_loader, device):
    """
    Evaluates the trained ParticleNet model on a test dataset.

    Args:
        model: model object to be evaluated
        test_loader: test set loaded via dataloader object
        device: pytorch device
    """
    # putting model in eval mode
    model.eval()
    
    all_labels = []
    all_probs = []
    all_preds = []
    
    # Disable gradient tracking for speed and memory efficiency
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            
            # forward pass - get raw logits
            logits = model(features)
            # get probabilities using Softmax
            probabilities = torch.softmax(logits, dim=1)
            # Get the predicted class (0 or 1) by finding the max probability
            _, predicted = torch.max(probabilities, 1)
            
            # Store the results for metric calculation
            # We specifically want the probability of the *signal* class (index 1) for ROC AUC
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities[:, 1].cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
    # Convert lists to numpy arrays
    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)
    y_preds = np.array(all_preds)
    
    # Standard Accuracy Calculation
    accuracy = accuracy_score(y_true, y_preds)
    
    # ROC AUC Calculation
    auc = roc_auc_score(y_true, y_probs)
    
    # Background Rejection at specific Signal Efficiencies
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    # Helper function to find background rejection at a target signal efficiency
    def get_bg_rejection(target_tpr):
        # Find the index where the true positive rate (tpr) is closest to our target
        idx = np.argmin(np.abs(tpr - target_tpr))
        false_positive_rate = fpr[idx]
        
        # Background rejection is 1 / false_positive_rate (1 / epsilon_b)
        # Add a tiny epsilon to prevent division by zero if the model is perfect
        bg_rejection = 1.0 / (false_positive_rate + 1e-9)
        return bg_rejection
    
    # Calculate for 50% and 30% signal efficiency as specified in the paper
    bg_rej_50 = get_bg_rejection(0.50)
    bg_rej_30 = get_bg_rejection(0.30)
    
    # Print the final report
    print("=== ParticleNet Final Test Results ===")
    print(f"Accuracy:                    {accuracy:.4f}")
    print(f"ROC AUC:                     {auc:.4f}")
    print(f"Background Rejection @ 50%:  {bg_rej_50:.1f}")
    print(f"Background Rejection @ 30%:  {bg_rej_30:.1f}")
    print("======================================")
    
    return accuracy, auc, bg_rej_50, bg_rej_30


if __name__ == '__main__':

    task = "quark_gluon_classification" # other is "quark_gluon_classification"
    model_variant = "lite" # other is "full"

    best_model_weight_path = 'particlenet_best_model.pth'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating with device: {device}")
    print(f"Task: {task}, Model Variant: {model_variant}")
    
    # create the model object
    if task == "top-tagging":
        # 1. Full and Lite models for Top-Tagging Task 
        if model_variant == "full":
            model = ParticleNet(num_features=7, num_classes=2, k=16).to(device)
        else:
            model = ParticleNetLite(num_features=7, num_classes=2, k=7).to(device)
        
        # load the test data
        test_jets, test_labels = read_h5_dataset(file_path='../datasets/top_tagging_dataset/test.h5', key='table')
        test_dataset = TopTaggingDataset(test_jets, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=384, shuffle=False, num_workers=4)
    
    else:
        # 1. Full and Lite models for Quark-Gluon Classification Task 
        if model_variant == "full":
            model = ParticleNet(num_features=13, num_classes=2, k=16).to(device)
        else:
            model = ParticleNetLite(num_features=13, num_classes=2, k=7).to(device)
        
        # load the test data
        folder_path = '../task2Datasets/'
        test_loader = get_qg_dataloader(folder_path=folder_path, begin_file_idx=18, end_file_idx=20, batch_size=384, shuffle=False)
    
    # load the trained model weights
    model.load_state_dict(torch.load(best_model_weight_path, map_location=device))
    print(f"Successfully loaded weights from {best_model_weight_path}")
    
    # run evaluation
    evaluate_model(model, test_loader, device)
    