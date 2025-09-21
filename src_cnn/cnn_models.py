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
        energy = self.energy(lstm_output).squeeze(-1)  # -> (batch, seq_len)

        # Get attention weights by applying softmax
        attention_weights = F.softmax(energy, dim=1)  # -> (batch, seq_len)

        # Reshape weights for batch matrix multiplication
        # -> (batch, 1, seq_len)
        attention_weights = attention_weights.unsqueeze(1)

        # Calculate the context vector (weighted sum of lstm_output)
        # (batch, 1, seq_len) @ (batch, seq_len, hidden_size) -> (batch, 1, hidden_size)
        context_vector = torch.bmm(attention_weights, lstm_output)

        # Remove the middle dimension
        return context_vector.squeeze(1)  # -> (batch, hidden_size)


class UltimateHybridRegressor(nn.Module):
    """
    A model that uses a pre-trained CNN to extract features from each frame
    of a sequence, and an LSTM to learn temporal patterns from those features.
    """

    def __init__(self, num_context_features, num_dynamic_features,
                 lstm_hidden_size=256, lstm_layers=3, pretrained=True, cnn_in_channels=3):  # Added cnn_in_channels
        super().__init__()

        self.cnn_in_channels = cnn_in_channels

        # --- CNN Feature Extractor ---
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        original_conv1 = resnet.conv1

        resnet.conv1 = nn.Conv2d(
            in_channels=self.cnn_in_channels,
            out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained:
            with torch.no_grad():
                if self.cnn_in_channels == 3:
                    resnet.conv1.weight.data = original_conv1.weight.data
                elif self.cnn_in_channels == 2:
                    avg_weights = original_conv1.weight.data.mean(
                        dim=1, keepdim=True)
                    resnet.conv1.weight.data = avg_weights.repeat(1, 2, 1, 1)
                elif self.cnn_in_channels == 1:
                    avg_weights = original_conv1.weight.data.mean(
                        dim=1, keepdim=True)
                    resnet.conv1.weight.data = avg_weights.repeat(1, 1, 1, 1)

        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.cnn[7].parameters():
            param.requires_grad = True

        # --- LSTM for Temporal Features ---
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.4 if lstm_layers > 1 else 0
        )
        self.attention = Attention(lstm_hidden_size)

        self.context_mlp = nn.Sequential(
            nn.Linear(num_context_features, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8)
        )
        self.dynamic_mlp = nn.Sequential(
            # Layer 1: Input -> 32 features
            nn.Linear(num_dynamic_features, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32), 

            # Layer 2: 32 features -> 16 features
            nn.Linear(32, 16),  
            nn.ReLU(),
            nn.BatchNorm1d(16),

            # Layer 3: 16 features -> 8 features (final output)
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8)
        )

        combined_feature_size = lstm_hidden_size + 8 + 8
        self.head = nn.Sequential(
            nn.Linear(combined_feature_size, 64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, 1)
        )

    def forward(self, image_sequence, context_data, dynamic_data):
        batch_size, seq_len, C, H, W = image_sequence.shape
        image_sequence = image_sequence.view(batch_size * seq_len, C, H, W)
        cnn_out = self.cnn(image_sequence)
        cnn_out = cnn_out.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(cnn_out)
        attention_out = self.attention(lstm_out)
        context_features = self.context_mlp(context_data)
        dynamic_features = self.dynamic_mlp(dynamic_data)
        combined = torch.cat(
            [attention_out, context_features, dynamic_features], dim=1)
        prediction = self.head(combined)
        return prediction.squeeze(-1)


class UltimateHybridClassifier(nn.Module):
    def __init__(self, num_context_features, num_dynamic_features, num_classes,
                 lstm_hidden_size=256, lstm_layers=3, pretrained=True):
        super().__init__()

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # --- START OF FIX ---

        # The classifier will run on the 2-channel flow data
        self.cnn_in_channels = 2
        original_conv1 = resnet.conv1

        resnet.conv1 = nn.Conv2d(
            in_channels=self.cnn_in_channels, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained:
            # Use the weight averaging strategy for 2-channel input
            with torch.no_grad():
                avg_weights = original_conv1.weight.data.mean(
                    dim=1, keepdim=True)
                resnet.conv1.weight.data = avg_weights.repeat(1, 2, 1, 1)

        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.cnn[7].parameters():
            param.requires_grad = True

        # --- LSTM, Attention, MLPs (Identical to Regressor) ---
        self.lstm = nn.LSTM(
            input_size=512, hidden_size=lstm_hidden_size,
            num_layers=lstm_layers, batch_first=True,
            dropout=0.4 if lstm_layers > 1 else 0
        )
        self.attention = Attention(lstm_hidden_size)
        self.context_mlp = nn.Sequential(
            nn.Linear(num_context_features, 8), nn.ReLU(), nn.BatchNorm1d(8)
        )
        self.dynamic_mlp = nn.Sequential(
            nn.Linear(num_dynamic_features, 16), nn.ReLU(), nn.BatchNorm1d(16),
            nn.Linear(16, 8), nn.ReLU(), nn.BatchNorm1d(8)
        )

        # --- Final Combined Head (The only change is the last layer) ---
        combined_feature_size = lstm_hidden_size + 8 + 8
        self.head = nn.Sequential(
            nn.Linear(combined_feature_size, 64),
            nn.ReLU(),
            nn.Dropout(0.6),
            # <-- CRITICAL CHANGE: Output num_classes instead of 1
            nn.Linear(64, num_classes)
        )

    def forward(self, image_sequence, context_data, dynamic_data):
        # --- This forward pass is IDENTICAL to the regressor ---
        batch_size, seq_len, C, H, W = image_sequence.shape
        image_sequence = image_sequence.view(batch_size * seq_len, C, H, W)
        cnn_out = self.cnn(image_sequence)
        cnn_out = cnn_out.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(cnn_out)
        attention_out = self.attention(lstm_out)
        context_features = self.context_mlp(context_data)
        dynamic_features = self.dynamic_mlp(dynamic_data)
        combined = torch.cat(
            [attention_out, context_features, dynamic_features], dim=1)

        # --- The output is now raw scores (logits) for each class ---
        prediction = self.head(combined)
        return prediction


class SimplifiedCnnAvgRegressor(nn.Module):
    """
    A simplified diagnostic model. Replaces LSTM/Attention with mean-pooling.
    """

    def __init__(self, num_context_features, num_dynamic_features, pretrained=True, cnn_in_channels=3):
        super().__init__()

        self.cnn_in_channels = cnn_in_channels

        # --- CNN Feature Extractor (Adapts to input channels) ---
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        original_conv1 = resnet.conv1

        resnet.conv1 = nn.Conv2d(
            in_channels=self.cnn_in_channels, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained:
            with torch.no_grad():
                if self.cnn_in_channels == 3:
                    resnet.conv1.weight.data = original_conv1.weight.data
                else:  # For 1 or 2 channels, use the averaging strategy
                    avg_weights = original_conv1.weight.data.mean(
                        dim=1, keepdim=True)
                    resnet.conv1.weight.data = avg_weights.repeat(
                        1, self.cnn_in_channels, 1, 1)

        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.cnn[7].parameters():
            param.requires_grad = True

        # --- MLPs for Tabular Data (Identical) ---
        self.context_mlp = nn.Sequential(
            nn.Linear(num_context_features, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8)
        )
        self.dynamic_mlp = nn.Sequential(
            nn.Linear(num_dynamic_features, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8)
        )

        # --- Final Combined Head (Input size is different!) ---
        # CNN output (512) + context (8) + dynamic (8)
        combined_feature_size = 512 + 8 + 8

        self.head = nn.Sequential(
            nn.Linear(combined_feature_size, 64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, 1)
        )

    def forward(self, image_sequence, context_data, dynamic_data):
        batch_size, seq_len, C, H, W = image_sequence.shape
        image_sequence = image_sequence.view(batch_size * seq_len, C, H, W)

        cnn_out = self.cnn(image_sequence)
        # Shape: (Batch, Sequence, 512)
        cnn_out = cnn_out.view(batch_size, seq_len, -1)

        # --- KEY CHANGE: Replace LSTM/Attention with simple averaging ---
        averaged_cnn_features = torch.mean(
            cnn_out, dim=1)  # Shape: (Batch, 512)

        context_features = self.context_mlp(context_data)
        dynamic_features = self.dynamic_mlp(dynamic_data)

        combined = torch.cat(
            [averaged_cnn_features, context_features, dynamic_features], dim=1)

        prediction = self.head(combined)
        return prediction.squeeze(-1)
