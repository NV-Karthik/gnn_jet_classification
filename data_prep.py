import torch
import pandas as pd
import numpy as np
import gc

def read_h5_dataset(file_path, key, num_jets=None):
    """
    Reads the raw Top Tagging HDF5 file and converts it into the 3D NumPy arrays 
    required by the ParticleNet DataLoader.
    
    Args:
        file_path: Path to the raw .h5 file.
        key: key identifier within the .h5 file
        jet_cutoff: reduce the number of jets
        
    Returns:
        jets_3d: NumPy array of shape (N_jets, 200, 4) ordered as (px, py, pz, E).
        labels: NumPy array of shape (N_jets,) containing binary labels.
    """
    
    df = pd.read_hdf(file_path, key=key, stop=num_jets)
    if num_jets is not None:
        df = df.iloc[:num_jets]
    
    labels = df['is_signal_new'].values.copy() # Extract labels - 1 for Top Quark (Signal), 0 for QCD (Background)
    
    jets = df.iloc[:, :800].values # ignore truthPX, ttv, etc.
    
    del df
    gc.collect()
    
    # 4 == num_features (E, px, py, pz), 200 == num_particles in a jet, -1 give num_jets
    jets = jets.reshape(-1, 200, 4)
    
    # reorder the 4-momentum features from (E, px, py, pz) to (px, py, pz, E)
    jets = jets[:, :, [1, 2, 3, 0]]
    
    print(f"Data successfully processed. Final shape: {jets.shape}")
    return jets, labels


def load_qg_npz(file_path, max_particles=100):
    """
    Reads a QG .npz file, computes the 13 ParticleNet features, 
    and returns memory-efficient PyTorch tensors.
    """
    print(f"Loading {file_path}...")
    data = np.load(file_path)
    X_raw = data['X']  # shape (100000, M, 4) -> (pt, y, phi, pdgid)
    labels = data['y'] # shape (100000,) -> 0 for Gluon, 1 for Quark
    

    pt = X_raw[..., 0]
    y_part = X_raw[..., 1]
    phi_part = X_raw[..., 2]
    pdgid = X_raw[..., 3]
    
    energy = pt * np.cosh(y_part) # E = pT * cosh(y))
    px = pt * np.cos(phi_part)
    py = pt * np.sin(phi_part)
    pz = pt * np.sinh(y_part)
    
    jet_px = np.sum(px, axis=1, keepdims=True)
    jet_py = np.sum(py, axis=1, keepdims=True)
    jet_pz = np.sum(pz, axis=1, keepdims=True)
    jet_energy = np.sum(energy, axis=1, keepdims=True)
    
    jet_pt = np.sqrt(jet_px**2 + jet_py**2) + 1e-9
    jet_y = 0.5 * np.log((jet_energy + jet_pz + 1e-9) / (jet_energy - jet_pz + 1e-9))
    jet_phi = np.arctan2(jet_py, jet_px)
    
    # 3. Sort by descending pT truncate
    sort_idx = np.argsort(pt, axis=1)[:, ::-1][:, :max_particles]
    
    pt = np.take_along_axis(pt, sort_idx, axis=1)
    y_part = np.take_along_axis(y_part, sort_idx, axis=1)
    phi_part = np.take_along_axis(phi_part, sort_idx, axis=1)
    energy = np.take_along_axis(energy, sort_idx, axis=1)
    pdgid = np.take_along_axis(pdgid, sort_idx, axis=1)
    
    # Calculate final features
    delta_y = y_part - jet_y
    delta_phi = phi_part - jet_phi
    
    delta_phi[delta_phi > np.pi] -= 2 * np.pi
    delta_phi[delta_phi < -np.pi] += 2 * np.pi
    
    log_pt = np.log(pt + 1e-9)
    log_e = np.log(energy + 1e-9)
    log_pt_rel = np.log(pt / jet_pt + 1e-9)
    log_e_rel = np.log(energy / jet_energy + 1e-9)
    delta_r = np.sqrt(delta_y**2 + delta_phi**2)
    
    abs_pdgid = np.abs(pdgid)
    is_electron = (abs_pdgid == 11).astype(np.float32)
    is_muon = (abs_pdgid == 13).astype(np.float32)
    is_photon = (abs_pdgid == 22).astype(np.float32)
    is_neutral_hadron = np.isin(abs_pdgid, [111, 130, 310, 2112, 3122]).astype(np.float32)
    is_charged_hadron = np.isin(abs_pdgid, [211, 321, 2212, 3222, 3112, 3312, 3334]).astype(np.float32)
    
    charge = np.zeros_like(pdgid, dtype=np.float32)
    charge[is_electron == 1] = -np.sign(pdgid[is_electron == 1])
    charge[is_muon == 1] = -np.sign(pdgid[is_muon == 1])
    charge[is_charged_hadron == 1] = np.sign(pdgid[is_charged_hadron == 1])
    
    features = np.stack([
        delta_y, delta_phi, log_pt, log_e, log_pt_rel, log_e_rel, delta_r,
        charge, is_electron, is_muon, is_charged_hadron, is_neutral_hadron, is_photon
    ], axis=1)
    
    # free up RAM 
    del X_raw, data, pt, y_part, phi_part, energy, pdgid
    gc.collect()
    
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

