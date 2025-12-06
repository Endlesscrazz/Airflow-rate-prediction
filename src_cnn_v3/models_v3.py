# src_cnn_v3/models_v3.py
import torch
import torch.nn as nn

class HybridCropRegressor(nn.Module):
    """
    Dual-stream architecture: CNN-LSTM + MLP.
    """
    def __init__(self, num_tabular_features, lstm_hidden=256, lstm_layers=3, dropout=0.3):
        super().__init__()
        
        # --- Visual Stream ---
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # --- Temporal Stream ---
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # --- Tabular Stream ---
        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # --- Fusion Head ---
        fusion_dim = lstm_hidden + 128
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Predicts normalized scalar [0, 1]
        )

    def forward(self, video, features):
        B, T, C, H, W = video.shape
        
        # CNN
        c_in = video.view(B * T, C, H, W)
        c_out = self.cnn(c_in)
        
        # LSTM
        r_in = c_out.view(B, T, -1)
        lstm_out, _ = self.lstm(r_in)
        visual_embedding = lstm_out.mean(dim=1)
        
        # MLP
        geo_embedding = self.tabular_mlp(features)
        
        # Fusion
        combined = torch.cat([visual_embedding, geo_embedding], dim=1)
        output = self.head(combined)
        
        return output.squeeze(-1)