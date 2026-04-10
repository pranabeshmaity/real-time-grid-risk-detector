"""
UGIM Core - Ultimate Grid Intelligence Model
State-of-the-art Transformer-GNN Hybrid for Grid Oscillation Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from dataclasses import dataclass, field

@dataclass
class PredictionResult:
    risk_score: float
    confidence: float
    uncertainty: float
    oscillation_mode: int
    mode_probabilities: List[float]
    attention_weights: Optional[List[float]] = None
    feature_importance: Optional[Dict[str, float]] = None

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_weights = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_output))
        return x, attn_weights

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ff = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        x = self.norm(x + self.dropout(ff))
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, nhead, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, attn_weights = self.attention(x)
        x = self.ffn(x)
        return x, attn_weights

class UGIMCore(nn.Module):
    def __init__(
        self,
        input_dim: int = 15,
        d_model: int = 256,
        nhead: int = 16,
        num_layers: int = 6,
        num_modes: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_modes = num_modes
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        
        self.risk_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        
        self.mode_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_modes)
        )
        
        self.feature_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        
        attention_weights = []
        for block in self.encoder_blocks:
            x, attn = block(x)
            if return_attention:
                attention_weights.append(attn)
        
        x_pooled = x.mean(dim=1)
        
        risk = self.risk_head(x_pooled).squeeze(-1)
        uncertainty = self.uncertainty_head(x_pooled).squeeze(-1)
        mode_logits = self.mode_head(x_pooled)
        mode_probs = F.softmax(mode_logits, dim=-1)
        predicted_mode = torch.argmax(mode_probs, dim=-1)
        
        result = {
            "risk": risk,
            "uncertainty": uncertainty,
            "mode_logits": mode_logits,
            "mode_probs": mode_probs,
            "predicted_mode": predicted_mode
        }
        
        if return_attention:
            result["attention_weights"] = attention_weights
        
        return result

class UGIMPredictor:
    def __init__(self, model_path: Optional[str] = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UGIMCore().to(self.device)
        
        if model_path:
            self.load_model(model_path)
        
        self.model.eval()

    def predict(self, features: np.ndarray) -> PredictionResult:
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).to(self.device)
            
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            if len(x.shape) == 3:
                x = x.unsqueeze(1) if x.shape[1] != 1 else x
            
            output = self.model(x)
            
            risk = output["risk"].item()
            uncertainty = output["uncertainty"].item()
            confidence = 1.0 - min(0.3, uncertainty)
            predicted_mode = output["predicted_mode"].item()
            mode_probs = output["mode_probs"].cpu().numpy().tolist()[0]
            
            return PredictionResult(
                risk_score=risk,
                confidence=confidence,
                uncertainty=uncertainty,
                oscillation_mode=int(predicted_mode),
                mode_probabilities=mode_probs
            )

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.input_dim,
                'd_model': self.model.d_model,
                'num_modes': self.model.num_modes
            }
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
