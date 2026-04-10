# ml_pipeline/models/ultimate_predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Comprehensive prediction output with uncertainty."""
    oscillation_risk: float
    blackout_probability: float
    oscillation_mode: int
    confidence_interval: Tuple[float, float]  # 95% CI
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    attention_weights: Optional[np.ndarray] = None

class SpatioTemporalTransformer(nn.Module):
    """
    A novel hybrid model for spatio-temporal graph data.
    Combments: GAT for spatial, Transformer for temporal.
    """
    def __init__(self, in_channels: int = 15, hidden_dim: int = 128, out_channels: int = 1, num_heads: int = 8):
        super().__init__()
        # 1. Spatial Encoder (Graph Attention Network)
        self.spatial_gat = GATConv(in_channels, hidden_dim, heads=num_heads, concat=False)
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        
        # 2. Temporal Encoder (Transformer)
        self.temporal_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(self.temporal_encoder_layer, num_layers=3)
        
        # 3. Output Heads
        self.risk_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.mode_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 8)) # 8 modes
        self.uncertainty_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 2), nn.Softplus()) # Epistemic + Aleatoric

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [batch, time, nodes, features]
        batch_size, time_steps, num_nodes, _ = x.shape
        
        # --- Spatial Encoding ---
        # Process each timestep independently with the GAT
        spatial_out = []
        for t in range(time_steps):
            # Reshape to [batch * nodes, features]
            node_features = x[:, t, :, :].reshape(-1, x.shape[-1])
            # Apply GAT
            processed = self.spatial_gat(node_features, edge_index)
            processed = self.spatial_norm(processed)
            # Reshape back to [batch, nodes, hidden_dim]
            processed = processed.reshape(batch_size, num_nodes, -1)
            spatial_out.append(processed)
        
        # Stack back to [batch, time, nodes, hidden_dim]
        spatial_out = torch.stack(spatial_out, dim=1)
        
        # --- Temporal Encoding ---
        # Combine node and time dimensions for the Transformer: [batch, time * nodes, hidden_dim]
        temporal_input = spatial_out.flatten(1, 2)
        temporal_out = self.temporal_encoder(temporal_input)
        
        # --- Global Pooling & Predictions ---
        # Global average pooling over nodes and time
        global_embedding = temporal_out.mean(dim=1)
        
        risk = self.risk_head(global_embedding).squeeze()
        mode_logits = self.mode_head(global_embedding)
        uncertainty_params = self.uncertainty_head(global_embedding)
        
        return {"risk": risk, "mode_logits": mode_logits, "uncertainty": uncertainty_params}

class UltimatePredictor:
    """A research-grade predictor with Monte Carlo Dropout for uncertainty."""
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create a dummy edge_index for the 118-bus system (a ring topology as an example)
        edge_index = torch.tensor([[i, i+1] for i in range(117)] + [[i+1, i] for i in range(117)], dtype=torch.long).t().contiguous()
        self.model = SpatioTemporalTransformer().to(self.device)
        self.model.eval()
        self.edge_index = edge_index.to(self.device)
        
        if model_path:
            self.load_model(model_path)
        logger.info("UltimatePredictor initialized with SpatioTemporalTransformer.")

    def predict(self, features: np.ndarray) -> PredictionResult:
        """Runs inference with Monte Carlo Dropout for uncertainty."""
        self.model.train() # Enable dropout for uncertainty estimation
        x = torch.tensor(features, dtype=torch.float32).to(self.device) # [1, time, nodes, features]
        
        all_risks = []
        all_mode_logits = []
        all_uncertainties = []
        
        with torch.no_grad():
            for _ in range(20): # MC Dropout samples
                out = self.model(x, self.edge_index)
                all_risks.append(out["risk"])
                all_mode_logits.append(out["mode_logits"])
                all_uncertainties.append(out["uncertainty"])
        
        self.model.eval()
        
        # Aggregate predictions
        risk_pred = torch.stack(all_risks).mean().item()
        risk_std = torch.stack(all_risks).std().item()
        
        mode_logits_mean = torch.stack(all_mode_logits).mean(dim=0)
        mode_pred = torch.argmax(mode_logits_mean).item()
        
        # 95% Confidence Interval
        ci_lower = max(0.0, risk_pred - 1.96 * risk_std)
        ci_upper = min(1.0, risk_pred + 1.96 * risk_std)
        
        # Uncertainty decomposition
        uncertainty_mean = torch.stack(all_uncertainties).mean(dim=0)
        epistemic, aleatoric = uncertainty_mean[0].item(), uncertainty_mean[1].item()
        
        blackout_prob = risk_pred * 0.65 # Placeholder for a more sophisticated model
        
        return PredictionResult(
            oscillation_risk=risk_pred,
            blackout_probability=blackout_prob,
            oscillation_mode=mode_pred,
            confidence_interval=(ci_lower, ci_upper),
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric
        )
    
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
