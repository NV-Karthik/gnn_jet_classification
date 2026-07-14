import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

from models import ParticleNet, ParticleNetLite
from data_loader import get_tt_dataloaders, get_qg_dataloader
from data_prep import read_h5_dataset


def train_model(
        model, train_loader, val_loader, 
        lossfn, optimizer, scheduler, num_epochs, 
        device, target_batch_size=384, micro_batch_size=32
    ):

    best_val_acc = 0.0

    accumulation_steps = target_batch_size // micro_batch_size
    print(f"Using Gradient Accumulation: {accumulation_steps} steps of size {micro_batch_size} to simulate batch size {target_batch_size}")
    
    if (device == torch.device('cuda')):
        scaler = GradScaler(device='cuda')
    else:
        scaler = GradScaler(device='cpu')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        # Training mode
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Zero gradients from at start of every epoch
        optimizer.zero_grad()

        # Iterate over batches of data
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)

            with autocast(device_type='cuda'): # mixed precision
                outputs = model(features) # forward pass
                loss = lossfn(outputs, labels)
                loss = loss / accumulation_steps # loss normalization - for gradient acc
            
            scaler.scale(loss).backward() # backward pass with gradient acc

            # update weights only after accumulation steps / end-of-dataset
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # update network weights
                scaler.step(optimizer)
                scaler.update()
                
                optimizer.zero_grad() # clear grads for next accumulation cycle
                
                # step scheduler
                scheduler.step()

            # track metrics for plotting
            running_loss += (loss.item() * accumulation_steps) * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Evaluation mode - for validation stats
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)

                with autocast(device_type='cuda'): # mixed precision
                    outputs = model(features)
                    loss = lossfn(outputs, labels)

                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}")

        # Save the model with best validation-set-performance
        if epoch_val_acc > best_val_acc:
            print(f"Validation accuracy increased ({best_val_acc:.4f} --> {epoch_val_acc:.4f}). Saving this as best model...")
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'particlenet_best_model.pth')

        print()

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.4f}")
    return model

# --- Execution Block ---
if __name__ == '__main__':

    task = "quark_gluon_classification" # other is "quark_gluon_classification"
    model_variant = "lite" # other is "full"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    print(f"Task: {task}, Model Variant: {model_variant}")

    if task == "top-tagging":
        # data loading
        train_jets, train_labels = read_h5_dataset(file_path='../datasets/top_tagging_dataset/train.h5', key='table')
        val_jets, val_labels = read_h5_dataset(file_path='../datasets/top_tagging_dataset/val.h5', key='table')
        train_loader, val_loader = get_tt_dataloaders(train_jets, train_labels, val_jets, val_labels, batch_size=384)
    
        # model definition
        if model_variant == "full":
            model = ParticleNet(num_features=7, num_classes=2, k=16).to(device)
        else:
            model = ParticleNetLite(num_features=7, num_classes=2, k=7).to(device)
    else:
        # data loading
        folder_path = '../task2Datasets/'
        train_loader = get_qg_dataloader(folder_path=folder_path, begin_file_idx=0, end_file_idx=10, batch_size=384, shuffle=True)
        val_loader = get_qg_dataloader(folder_path=folder_path, begin_file_idx=16, end_file_idx=18, batch_size=384, shuffle=False)
        
        # model definition
        if model_variant == "full":
            model = ParticleNet(num_features=13, num_classes=2, k=16).to(device)
        else:
            model = ParticleNetLite(num_features=13, num_classes=2, k=7).to(device)
    
    # Hyperparameters
    lossfn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0001)
    
    total_epochs = 20 # 8 (up) + 8 (down) + 4 (cooldown) = 20
    steps_per_epoch = len(train_loader)
    batch_size = 384
    micro_batch_size = 32 # for gradient accumulation. tweak according to your GPU VRAM
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=3e-3, 
        epochs=total_epochs, 
        steps_per_epoch=steps_per_epoch,
        pct_start=8/20, # increasing LR - epoch ratio
        anneal_strategy='linear', # The paper specifies linear transitions
        div_factor=10.0, # max_lr / div_factor = initial_lr (3e-3 / 10 = 3e-4)
        final_div_factor=600.0 # initial_lr / final_div_factor = cooldown_lr (3e-4 / 600 = 5e-7)
    )
    
    # start training
    trained_model = train_model(
        model, train_loader, val_loader, 
        lossfn, optimizer, scheduler, total_epochs, 
        device, batch_size, micro_batch_size)

