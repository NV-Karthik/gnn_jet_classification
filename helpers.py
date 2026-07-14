# Contains the K-NearestNeighbours and Graphimplementation
# To be used by the functions under models.py

import torch


def knn(x, k):
    """
    Calculates the k-nearest neighbors for a batch of point clouds.
    
    Args:
        x: Tensor of shape (B, C, N) representing coordinates.
        k: Integer, number of neighbors to find.
        
    Returns:
        idx: Tensor of shape (B, N, k) containing the indices of the neighbors.
    """
    
    # For -2 * (x_i . x_j)
    # x.transpose{(B, C, N)} == (B, N, C) 
    # (B, N, C) x (B, C, N) == (B, N, N)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    
    # For ||x_i||^2. Final Shape = (B, 1, N) due to keepdim=True
    x_sq = torch.sum(x**2, dim=1, keepdim=True)
    
    # x_sq.transpose(2, 1) is (B, N, 1)     --> Represents ||x_i||^2
    # inner is (B, N, N)    --> Represents -2(x_i \cdot x_j)
    # x_sq is (B, 1, N)     --> Represents ||x_j||^2
    pairwise_distance = x_sq.transpose(2, 1) + inner + x_sq
    
    idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]
    
    return idx


def get_graph_feature(x, k, idx):
    """
    Constructs the edge features for the DGCNN.
    
    Args:
        x: Tensor of shape (B, C, N) representing particle features.
        k: Integer, number of neighbors.
        idx: Tensor of shape (B, N, k) containing neighbor indices.
        
    Returns:
        edge_features: Tensor of shape (B, 2*C, N, k) ready for the MLP.
    """
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    
    # We transpose x to (B, N, C) to easily index particles.
    x = x.transpose(2, 1).contiguous()
    
    # generate batch index tensor - get all batches
    batch_idx = torch.arange(batch_size, device=x.device).view(batch_size, 1, 1).expand(batch_size, num_points, k)
    
    # for every batch, point, and neighbor, get the C features. [Shape- (B, N, k, C)]
    neighbors_features = x[batch_idx, idx, :]
    
    # copy central particle features - k times [Shape - (B, N, C) to (B, N, k, C)]
    central_features = x.view(batch_size, num_points, 1, num_dims).expand(batch_size, num_points, k, num_dims)
    
    # edge features == concatenate[central_features, (neighbors_features - central_features)]
    # Shape: (B, N, k, 2*C)
    edge_features = torch.cat((central_features, neighbors_features - central_features), dim=-1)
    
    # Switch dimensions for final output [Shape- (B, N, k, 2*C) to (B, 2*C, N, k)]
    return edge_features.permute(0, 3, 1, 2).contiguous()

