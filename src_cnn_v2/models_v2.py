# src_cnn_v2/models_v2.py
"""
Contains the PyTorch model architectures for the V2 (bottom-up) pipeline.
"""
import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    """A simple 3-layer CNN to encode a single cropped frame."""
    def __init__(self, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            # Input: (Batch, 1, 32, 32)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> (Batch, 32, 16, 16)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> (Batch, 64, 8, 8)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # -> (Batch, 128, 1, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.features(x)
        # Flatten the output
        x = x.view(x.size(0), -1) # -> (Batch, 128)
        x = self.dropout(x)
        return x

class SimpleCropRegressor(nn.Module):
    """
    A simplified CNN-LSTM model for cropped video sequences.
    - Uses a shallow CNN encoder.
    - Uses an LSTM with mean pooling for temporal aggregation.
    - Fuses video features with an embedded delta_T feature.
    """
    def __init__(self, lstm_hidden_size=128, lstm_layers=2, dropout=0.4):
        super().__init__()

        # --- CNN Feature Extractor ---
        self.cnn = CNNEncoder(dropout=dropout)
        cnn_output_size = 128

        # --- LSTM for Temporal Features ---
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # --- MLP for Delta_T Feature ---
        # A small network to embed the single delta_T value
        self.delta_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )
        delta_output_size = 32
        
        # --- Final Prediction Head ---
        combined_feature_size = lstm_hidden_size + delta_output_size
        self.head = nn.Sequential(
            nn.Linear(combined_feature_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, image_sequence, delta_t):
        # image_sequence shape: (Batch, Time, Channels, Height, Width)
        B, T, C, H, W = image_sequence.shape
        
        # 1. Process each frame with the CNN
        # Flatten batch and time dimensions to process all frames at once
        c_in = image_sequence.view(B * T, C, H, W)
        c_out = self.cnn(c_in)
        
        # Reshape back to a sequence for the LSTM
        r_in = c_out.view(B, T, -1) # -> (Batch, Time, 128)
        
        # 2. Process sequence with LSTM
        lstm_out, _ = self.lstm(r_in)
        
        # 3. Aggregate temporal features using mean pooling
        # lstm_out shape: (Batch, Time, lstm_hidden_size)
        video_features = lstm_out.mean(dim=1) # -> (Batch, lstm_hidden_size)
        
        # 4. Process delta_T feature
        # Add a dimension for the linear layer
        delta_t_reshaped = delta_t.unsqueeze(1) # -> (Batch, 1)
        delta_features = self.delta_mlp(delta_t_reshaped) # -> (Batch, 32)

        # 5. Combine video and delta_T features
        combined = torch.cat([video_features, delta_features], dim=1)
        
        # 6. Make final prediction
        prediction = self.head(combined)
        
        return prediction.squeeze(-1)