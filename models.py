import torch
import torch.nn as nn

from helpers import knn, get_graph_feature

class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels_list, k):
        """
        Args:
            in_channels: Number of input features per particle.
            out_channels_list: A list of 3 integers defining the MLP layer sizes (e.g., [64, 64, 64]).
            k: Number of nearest neighbors.
        """
        super(EdgeConvBlock, self).__init__()
        self.k = k
        
        # We use Sequential to stack our MLP layers cleanly.
        # The first layer takes 2 * in_channels because of the concatenated edge features.
        layers = []
        current_in = 2 * in_channels
        
        for out_channels in out_channels_list:
            layers.append(nn.Conv2d(current_in, out_channels, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            current_in = out_channels
            
        self.mlp = nn.Sequential(*layers)
        
        # The paper implements a ResNet-style shortcut connection 
        # If the input dimension doesn't match the final output dimension, 
        # we need a 1x1 conv to project it to the right size so we can add them.
        final_out = out_channels_list[-1]
        if in_channels != final_out:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, final_out, kernel_size=1, bias=False),
                nn.BatchNorm1d(final_out)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, coords=None):
        """
        Args:
            x: Features tensor of shape (B, C, N).
            coords: Coordinates tensor of shape (B, C', N). If None, uses x as coords.
        """
        # 1. Dynamic Graph: Use features as coordinates if none are explicitly provided
        if coords is None:
            coords = x
            
        # 2. Find the k-nearest neighbors in the coordinate space
        idx = knn(coords, k=self.k)
        
        # 3. Construct the edge features
        # Shape goes from (B, C, N) -> (B, 2*C, N, k)
        edge_features = get_graph_feature(x, self.k, idx)
        
        # 4. Apply the MLP (1x1 Convolutions)
        # Shape goes from (B, 2*C, N, k) -> (B, out_channels, N, k)
        out = self.mlp(edge_features)
        
        # 5. Aggregation Operation
        # The paper specifies using the 'mean' aggregation over the neighbors.
        # We average over the last dimension (the k neighbors).
        # Shape goes from (B, out_channels, N, k) -> (B, out_channels, N)
        out = out.mean(dim=-1)
        
        # 6. Apply Shortcut Connection 
        # Add the original input (passed through the shortcut projection) to the output
        out = out + self.shortcut(x)
        
        return out


class ParticleNet(nn.Module):
    def __init__(self, num_features, num_classes=2, k=16):
        """
        Args:
            num_features: Number of input variables per particle (e.g., 7 or 13).
            num_classes: Number of output classes (e.g., 2 for signal vs background).
            k: Number of nearest neighbors to use in EdgeConv.
        """
        super(ParticleNet, self).__init__()
        self.k = k
        
        # EdgeConv Blocks
        # Block 1: Input is num_features, output is 64
        self.conv1 = EdgeConvBlock(in_channels=num_features, out_channels_list=[64, 64, 64], k=self.k)
        # Block 2: Input is 64, output is 128
        self.conv2 = EdgeConvBlock(in_channels=64, out_channels_list=[128, 128, 128], k=self.k)
        # Block 3: Input is 128, output is 256
        self.conv3 = EdgeConvBlock(in_channels=128, out_channels_list=[256, 256, 256], k=self.k)
        
        # Fully Connected Classifier layer
        self.fc_classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, N) containing particle features.
               We assume the first two channels (x[:, 0:2, :]) are delta_eta and delta_phi.
        """
        # Extract the spatial coordinates (delta_eta, delta_phi) for the first block [Shape: (B, 2, N)]
        coords = x[:, 0:2, :].contiguous()
        
        # Pass through Edge Conv Blocks
        out1 = self.conv1(x, coords=coords) # also passing spatial coordinates [Shape (B, C, N) -> (B, 64, N)]
        out2 = self.conv2(out1) # coords are omitted (meaning coord=out1) [Shape (B, 64, N) -> (B, 128, N)]
        out3 = self.conv3(out2) # [Shape (B, 128, N) -> (B, 256, N)]

        # Mean Pooling
        pooled = out3.mean(dim=-1) # [Shape from (B, 256, N) -> (B, 256)] 
        
        # Final Classification [Shape from (B, 256) -> (B, num_classes)]
        logits = self.fc_classifier(pooled)
        
        return logits


class ParticleNetLite(nn.Module):
    def __init__(self, num_features, num_classes=2, k=7):
        """
        A lightweight version of ParticleNet with only 26k parameters.
        """
        super(ParticleNetLite, self).__init__()
        self.k = k
        
        self.conv1 = EdgeConvBlock(in_channels=num_features, out_channels_list=[32, 32, 32], k=self.k)
        self.conv2 = EdgeConvBlock(in_channels=32, out_channels_list=[64, 64, 64], k=self.k)
        
        self.fc_classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        coords = x[:, 0:2, :].contiguous()
        
        out1 = self.conv1(x, coords=coords)
        out2 = self.conv2(out1)
        pooled = out2.mean(dim=-1)
        
        logits = self.fc_classifier(pooled)
        
        return logits

