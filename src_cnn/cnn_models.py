# src/cnn_models.py
"""
Contains the PyTorch model architectures for the airflow prediction task.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Attention(nn.Module):
    """
    A simple but effective attention mechanism.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.energy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        
        # Calculate energy scores for each time step
        energy = self.energy(lstm_output).squeeze(-1) # -> (batch, seq_len)
        
        # Get attention weights by applying softmax
        attention_weights = F.softmax(energy, dim=1) # -> (batch, seq_len)
        
        # Reshape weights for batch matrix multiplication
        # -> (batch, 1, seq_len)
        attention_weights = attention_weights.unsqueeze(1) 
        
        # Calculate the context vector (weighted sum of lstm_output)
        # (batch, 1, seq_len) @ (batch, seq_len, hidden_size) -> (batch, 1, hidden_size)
        context_vector = torch.bmm(attention_weights, lstm_output)
        
        # Remove the middle dimension
        return context_vector.squeeze(1) # -> (batch, hidden_size)

class UltimateHybridRegressor(nn.Module):
    """
    A model that uses a pre-trained CNN to extract features from each frame
    of a sequence, and an LSTM to learn temporal patterns from those features.
    """
    def __init__(self,num_context_features,num_dynamic_features, 
                 lstm_hidden_size=256, lstm_layers=3, pretrained=True):
        super().__init__()
        
        # --- CNN Feature Extractor ---
        # Load a pre-trained ResNet-18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        
        # Adapt the first conv layer for our 1-channel images(for thermal data)
        # original_conv1 = resnet.conv1
        # resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # if pretrained:
        #     resnet.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)

        #(for 2-channel optical flow):
        # Adapt the first conv layer for our 2-channel images
        original_conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            in_channels=2, # <-- The key change
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        if pretrained:
            # We can't just sum the weights like before. A common strategy is to
            # average the R and G channels and discard the B channel, as these
            # often contain the most motion-relevant information in ImageNet.
            # Or, for simplicity, we can just average all three.
            with torch.no_grad():
                original_weights = original_conv1.weight.data
                # Average across the 3 RGB channels to get a (64, 1, 7, 7) tensor
                avg_weights = original_weights.mean(dim=1, keepdim=True)
                # Repeat this average weight for our two input channels
                resnet.conv1.weight.data = avg_weights.repeat(1, 2, 1, 1)

        # We take all layers of ResNet except for the final classification layer (fc)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze the CNN layers. We will only use it as a fixed feature extractor.
        for param in self.cnn.parameters():
            param.requires_grad = False

        for param in self.cnn[7].parameters():
            param.requires_grad = True
            
        # --- LSTM for Temporal Features ---
        # The input size for the LSTM is the number of features output by ResNet-18 (512)
        self.lstm = nn.LSTM(
            input_size=512, 
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers, 
            batch_first=True,  # This makes handling batch dimensions easier
            dropout=0.4 if lstm_layers > 1 else 0
        )

        # Add the attention layer
        self.attention = Attention(lstm_hidden_size)

        self.context_mlp = nn.Sequential(
            nn.Linear(num_context_features, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8)
        )

         # --- NEW: MLP for Dynamic Handcrafted Features ---
        self.dynamic_mlp = nn.Sequential(
            nn.Linear(num_dynamic_features, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8)
        )
        
        # --- Final Combined Head ---
        # LSTM output (256) + context (8) + dynamic (8)
        combined_feature_size = lstm_hidden_size + 8 + 8

        self.head = nn.Sequential(
            nn.Linear(combined_feature_size, 64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, 1)
        )

    def forward(self, image_sequence, context_data, dynamic_data):
        # image_sequence has shape (Batch, Sequence, Channels, Height, Width)
        batch_size, seq_len, C, H, W = image_sequence.shape
        
        # Reshape to process all frames in the batch through the CNN at once
        # -> (Batch * Sequence, Channels, Height, Width)
        image_sequence = image_sequence.view(batch_size * seq_len, C, H, W)
        
        # Get features for each frame from the CNN
        cnn_out = self.cnn(image_sequence) # Shape: (B*S, 512, 1, 1)
        
        # Reshape back to a sequence for the LSTM
        # -> (Batch, Sequence, Features)
        cnn_out = cnn_out.view(batch_size, seq_len, -1)
        
        # Pass the sequence of features to the LSTM
        lstm_out, _ = self.lstm(cnn_out)
        
        # We only need the output from the very last time step of the sequence
        #final_lstm_out = lstm_out[:, -1, :] # Shape: (Batch, lstm_hidden_size)

        #Use the attention mechanism to get a weighted summary of the entire sequence
        attention_out = self.attention(lstm_out) # Shape: (Batch, lstm_hidden_size)

        context_features = self.context_mlp(context_data)
        dynamic_features = self.dynamic_mlp(dynamic_data)
        
        # Combine the attention-weighted temporal features with the tabular features
        combined = torch.cat([attention_out, context_features, dynamic_features], dim=1)
        
        prediction = self.head(combined)
        return prediction.squeeze(-1)
    
