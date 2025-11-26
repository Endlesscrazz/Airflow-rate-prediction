# src_cnn_v2/models_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """A simple 3-layer CNN to encode a single cropped frame."""
    def __init__(self, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Input: (Batch, 1, 15, 15)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> (32, 7, 7)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> (64, 3, 3)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # -> (128, 1, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # -> (Batch, 128)
        x = self.dropout(x)
        return x

class SimpleCropRegressor(nn.Module):
    """
    Matches Colleague's Architecture:
    - 3-Layer CNN
    - 3-Layer LSTM (Hidden 256)
    - Delta T Projection (Hidden 256) <--- CRITICAL FIX
    - Concat Fusion (512 inputs to final FC)
    """
    def __init__(self, lstm_hidden_size=256, lstm_layers=3, dropout=0.3):
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

        # --- Delta_T Feature Projection ---
        # Colleague projects scalar delta_t directly to hidden_dim (256)
        self.delta_fc = nn.Linear(1, lstm_hidden_size)
        
        # --- Final Prediction Head ---
        # Colleague fuses LSTM(256) + Delta(256) = 512
        combined_feature_size = lstm_hidden_size * 2
        
        self.final_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(combined_feature_size, 1)

    def forward(self, image_sequence, delta_t):
        # image_sequence shape: (Batch, Time, 1, H, W)
        B, T, C, H, W = image_sequence.shape
        
        # 1. CNN
        c_in = image_sequence.view(B * T, C, H, W)
        c_out = self.cnn(c_in)
        
        # 2. LSTM
        r_in = c_out.view(B, T, -1) # -> (Batch, Time, 128)
        lstm_out, _ = self.lstm(r_in)
        
        # 3. Mean Pooling (Colleague uses mean pooling)
        video_features = lstm_out.mean(dim=1) # -> (Batch, 256)
        
        # 4. Process delta_T feature
        # Colleague logic: self.delta_fc(delta_t.unsqueeze(1))
        delta_t_reshaped = delta_t.unsqueeze(1) # -> (Batch, 1)
        delta_features = self.delta_fc(delta_t_reshaped) # -> (Batch, 256)

        # 5. Fusion (Concat)
        combined = torch.cat([video_features, delta_features], dim=1) # -> (Batch, 512)
        
        # 6. Final Prediction
        combined = self.final_dropout(combined)
        prediction = self.head(combined)
        
        return prediction.squeeze(-1)


# MAX-POOLING LSTM
# class CNNEncoder(nn.Module):
#     """A simple 3-layer CNN to encode a single cropped frame."""
#     def __init__(self, dropout=0.4):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((1, 1))
#         )
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.dropout(x)
#         return x

# class SimpleCropRegressor(nn.Module):
#     """
#     CNN-LSTM with Max Pooling for cropped video sequences.
#     """
#     def __init__(self, lstm_hidden_size=128, lstm_layers=2, dropout=0.4):
#         super().__init__()

#         self.cnn = CNNEncoder(dropout=dropout)
#         cnn_output_size = 128

#         self.lstm = nn.LSTM(
#             input_size=cnn_output_size,
#             hidden_size=lstm_hidden_size,
#             num_layers=lstm_layers,
#             batch_first=True,
#             dropout=dropout if lstm_layers > 1 else 0
#         )
        
#         # MLP for Delta_T Feature
#         self.delta_mlp = nn.Sequential(
#             nn.Linear(1, 16),
#             nn.ReLU(),
#             nn.Linear(16, 32)
#         )
#         delta_output_size = 32
        
#         # Final Prediction Head
#         combined_feature_size = lstm_hidden_size + delta_output_size
#         self.head = nn.Sequential(
#             nn.Linear(combined_feature_size, 64),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, 1)
#         )

#     def forward(self, image_sequence, delta_t):
#         # image_sequence: (Batch, Time, 1, H, W)
#         B, T, C, H, W = image_sequence.shape
        
#         # 1. CNN
#         c_in = image_sequence.view(B * T, C, H, W)
#         c_out = self.cnn(c_in)
        
#         # 2. LSTM
#         r_in = c_out.view(B, T, -1)
#         lstm_out, _ = self.lstm(r_in) # (Batch, Time, Hidden)
        
#         # 3. TEMPORAL MAX POOLING
#         # Take the maximum activation across the time dimension
#         video_features, _ = torch.max(lstm_out, dim=1) # (Batch, Hidden)
        
#         # 4. Delta T
#         delta_t_reshaped = delta_t.unsqueeze(1)
#         delta_features = self.delta_mlp(delta_t_reshaped)

#         # 5. Fusion
#         combined = torch.cat([video_features, delta_features], dim=1)
#         prediction = self.head(combined)
        
#         return prediction.squeeze(-1)