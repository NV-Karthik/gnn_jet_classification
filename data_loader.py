import os

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from data_prep import load_qg_npz

class TopTaggingDataset(Dataset):
    def __init__(self, raw_jets, labels, max_particles=100):
        """
        Args:
            raw_jets: A list (or array) of jets. Each jet is an (N, 4) numpy array of particle 4-momenta (px, py, pz, E).
            labels: A list or array of binary labels (0 for background, 1 for signal).
            max_particles: Maximum number of particles to keep per jet.
        """
        self.raw_jets = raw_jets
        self.labels = labels
        self.max_particles = max_particles

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        jet = self.raw_jets[idx]
        label = self.labels[idx]
        
        px, py, pz, energy = jet[:, 0], jet[:, 1], jet[:, 2], jet[:, 3]

        # p_T = sqrt(px^2 + py^2)
        pt = np.sqrt(px**2 + py**2)
        
        # p = sqrt(px^2 + py^2 + pz^2)
        p = np.sqrt(px**2 + py**2 + pz**2)
        
        # eta = 0.5 * ln((p + pz) / (p - pz))
        eta = 0.5 * np.log((p + pz + 1e-9) / (p - pz + 1e-9))
        
        # phi = atan2(py, px)
        phi = np.arctan2(py, px)

        # Sort by descending p_T and truncate
        sort_indices = np.argsort(pt)[::-1][:self.max_particles]
        pt = pt[sort_indices]
        energy = energy[sort_indices]
        eta = eta[sort_indices]
        phi = phi[sort_indices]
        px = px[sort_indices]
        py = py[sort_indices]

        # get jet-level features
        jet_pt = np.sum(pt)
        jet_energy = np.sum(energy)
        
        jet_px = np.sum(px)
        jet_py = np.sum(py)
        jet_pz = np.sum(pz[sort_indices])
        jet_p = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2)
        jet_eta = 0.5 * np.log((jet_p + jet_pz + 1e-9) / (jet_p - jet_pz + 1e-9))
        jet_phi = np.arctan2(jet_py, jet_px)

        # final feature extraction
        delta_eta = eta - jet_eta
        
        delta_phi = phi - jet_phi
        delta_phi[delta_phi > np.pi] -= 2 * np.pi
        delta_phi[delta_phi < -np.pi] += 2 * np.pi
        
        log_pt = np.log(pt + 1e-9)
        log_e = np.log(energy + 1e-9)
        log_pt_rel = np.log(pt / (jet_pt + 1e-9) + 1e-9)
        log_e_rel = np.log(energy / (jet_energy + 1e-9) + 1e-9)
        delta_r = np.sqrt(delta_eta**2 + delta_phi**2)


        features = np.stack([
            delta_eta, delta_phi, log_pt, log_e, log_pt_rel, log_e_rel, delta_r
        ], axis=0)

        # padding
        num_particles = features.shape[1]
        if num_particles < self.max_particles:
            pad_width = self.max_particles - num_particles
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')

        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return features_tensor, label_tensor


def get_tt_dataloaders(train_jets, train_labels, val_jets, val_labels, batch_size=384):
    train_dataset = TopTaggingDataset(train_jets, train_labels)
    val_dataset = TopTaggingDataset(val_jets, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


class ChunkedQGDataset(Dataset):
    def __init__(self, file_paths, max_particles=100):
        """
        Dynamically loads multiple .npz files to prevent RAM overflow.
        """
        self.file_paths = file_paths
        self.max_particles = max_particles
        
        self.all_features = []
        self.all_labels = []
        
        for fp in file_paths:
            feats, lbls = load_qg_npz(fp, max_particles)
            self.all_features.append(feats)
            self.all_labels.append(lbls)
            
        self.all_features = torch.cat(self.all_features, dim=0)
        self.all_labels = torch.cat(self.all_labels, dim=0)
        print(f"Dataset fully loaded. Total Jets: {self.all_labels.shape[0]}")

    def __len__(self):
        return self.all_labels.shape[0]

    def __getitem__(self, idx):
        return self.all_features[idx], self.all_labels[idx]

def get_qg_dataloader(folder_path, begin_file_idx, end_file_idx, batch_size=384, shuffle=True):
    """
    Dynamically loads a specific range of .npz files for the QG dataset.
    Files are assumed to be named QG_jets_withbc_0.npz to QG_jets_withbc_19.npz.
    
    Args:
        start_idx: Integer, the starting file index (inclusive).
        end_idx: Integer, the ending file index (exclusive).
        batch_size: Integer, number of jets per batch.
        shuffle: Boolean, whether to shuffle the dataset (True for training).
        
    Returns:
        A PyTorch DataLoader containing the specified data chunk.
    """
    
    print(f"Creating DataLoader for files {begin_file_idx} to {end_file_idx - 1}")
    
    file_paths = [os.path.join(folder_path, f'QG_jets_withbc_{i}.npz') 
                  for i in range(begin_file_idx, end_file_idx)]
    
    dataset = ChunkedQGDataset(file_paths)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    
    return loader

