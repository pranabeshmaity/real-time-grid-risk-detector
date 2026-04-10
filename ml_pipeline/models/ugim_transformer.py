import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple

class UGIMTransformer(nn.Module):
    """
    A state-of-the-art Transformer model for spatio-temporal grid prediction.
    """
    def __init__(self, input_dim: int = 15, d_model: int = 128, nhead: int = 8, num_layers: int = 4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.risk_head = nn.Linear(d_model, 1)
        self.uncertainty_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        risk = torch.sigmoid(self.risk_head(x))
        uncertainty = nn.functional.softplus(self.uncertainty_head(x))
        return risk.squeeze(-1), uncertainty.squeeze(-1)

class UltimatePredictor:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UGIMTransformer().to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            risk, uncertainty = self.model(x)
            return {
                "risk_score": risk.item(),
                "uncertainty": uncertainty.item(),
                "confidence": 1.0 - uncertainty.item()
            }
