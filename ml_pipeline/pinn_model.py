"""
Physics-Informed Neural Network for Power Grid Oscillation Prediction
Implements swing equation constraints with graph neural networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, EdgeConv
from torch_geometric.utils import add_self_loops, degree
import numpy as np
from typing import Dict, List, Tuple, Optional
import sympy as sp
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PhysicsParameters:
    """Physics parameters for power system dynamics"""
    inertia: torch.Tensor  # M - inertia constant
    damping: torch.Tensor   # D - damping coefficient
    reactance: torch.Tensor # X - line reactance
    conductance: torch.Tensor # G - line conductance
    voltage_setpoint: float = 1.0
    frequency_nominal: float = 60.0

class SwingEquationLayer(nn.Module):
    """
    Differentiable swing equation layer with physics constraints
    Implements: M * d²δ/dt² + D * dδ/dt = P_m - P_e
    """
    
    def __init__(self, hidden_dim: int, n_buses: int = 118):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_buses = n_buses
        
        # Learnable physics parameters with physical bounds
        self.inertia_logits = nn.Parameter(torch.randn(n_buses) * 0.1)
        self.damping_logits = nn.Parameter(torch.randn(n_buses) * 0.1)
        
        # Neural ODE for temporal dynamics
        self.ode_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.time_constant = nn.Parameter(torch.ones(1) * 0.01)
        
        # Physics residual network
        self.physics_residual = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)  # Predict dδ/dt and dω/dt
        )
        
    def get_physical_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get physically-constrained parameters"""
        # Ensure positive inertia and damping
        inertia = torch.sigmoid(self.inertia_logits) * 10.0 + 1.0  # Range [1, 11]
        damping = torch.sigmoid(self.damping_logits) * 5.0 + 0.5   # Range [0.5, 5.5]
        return inertia, damping
    
    def forward(self, h: torch.Tensor, delta: torch.Tensor, omega: torch.Tensor, 
                P_m: torch.Tensor, P_e: torch.Tensor, dt: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with swing equation integration
        h: hidden states [batch, nodes, features]
        delta: rotor angle [batch, nodes]
        omega: angular velocity [batch, nodes]
        P_m: mechanical power [batch, nodes]
        P_e: electrical power [batch, nodes]
        """
        batch_size, n_nodes, _ = h.shape
        
        # Get physical parameters
        inertia, damping = self.get_physical_parameters()
        inertia = inertia.unsqueeze(0).expand(batch_size, -1)
        damping = damping.unsqueeze(0).expand(batch_size, -1)
        
        # Compute acceleration from swing equation
        # d²δ/dt² = (P_m - P_e - D * ω) / M
        power_imbalance = P_m - P_e
        acceleration = (power_imbalance - damping * omega) / (inertia + 1e-8)
        
        # Physics-informed residual correction
        physics_input = torch.cat([h.mean(dim=1), h.std(dim=1)], dim=-1)
        physics_correction = self.physics_residual(physics_input)
        
        # Apply correction to acceleration and velocity
        acceleration = acceleration + physics_correction[:, 0:1] * 0.1
        omega_correction = physics_correction[:, 1:2] * 0.05
        
        # Euler integration
        new_omega = omega + acceleration * dt + omega_correction
        new_delta = delta + new_omega * dt
        
        # Update hidden states with neural ODE
        h_flat = h.view(-1, self.hidden_dim)
        new_h_flat = self.ode_gru(h_flat, h_flat)
        new_h = new_h_flat.view(batch_size, n_nodes, -1)
        
        return new_delta, new_omega, new_h

class GraphPhysicsLayer(nn.Module):
    """
    Graph neural network layer with power flow physics
    Implements Kirchhoff's laws and power flow equations
    """
    
    def __init__(self, in_dim: int, out_dim: int, n_buses: int = 118):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_buses = n_buses
        
        # Graph convolution with physics constraints
        self.gcn = GCNConv(in_dim, out_dim)
        self.gat = GATConv(in_dim, out_dim, heads=4, concat=False)
        
        # Power flow layer
        self.power_flow_proj = nn.Linear(out_dim * 2, out_dim)
        
        # Learnable admittance matrix
        self.admittance_real = nn.Parameter(torch.randn(n_buses, n_buses) * 0.01)
        self.admittance_imag = nn.Parameter(torch.randn(n_buses, n_buses) * 0.01)
        
    def compute_power_flow(self, voltages: torch.Tensor, angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute power flow using learned admittance matrix
        P_i = V_i * Σ(V_j * (G_ij * cos(θ_i - θ_j) + B_ij * sin(θ_i - θ_j)))
        Q_i = V_i * Σ(V_j * (G_ij * sin(θ_i - θ_j) - B_ij * cos(θ_i - θ_j)))
        """
        batch_size = voltages.shape[0]
        
        # Construct complex admittance
        Y = torch.complex(self.admittance_real, self.admittance_imag)
        
        # Complex voltages
        V_complex = voltages * torch.exp(1j * angles)
        
        # Compute power injection
        S = V_complex * torch.conj(torch.matmul(Y, V_complex.unsqueeze(-1)).squeeze(-1))
        
        P = torch.real(S)
        Q = torch.imag(S)
        
        return P, Q
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                voltages: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with physics-constrained message passing
        """
        # Standard GCN message passing
        h_gcn = self.gcn(x, edge_index)
        h_gcn = F.relu(h_gcn)
        
        # Attention-based message passing
        h_gat = self.gat(x, edge_index)
        h_gat = F.elu(h_gat)
        
        # Compute power flow constraints
        P_flow, Q_flow = self.compute_power_flow(voltages, angles)
        
        # Combine with power flow information
        power_features = torch.cat([P_flow.unsqueeze(-1), Q_flow.unsqueeze(-1)], dim=-1)
        power_features = self.power_flow_proj(power_features)
        
        # Fuse all information
        h = (h_gcn + h_gat + power_features) / 3
        
        return h

class PhysicsInformedGNN(nn.Module):
    """
    Complete Physics-Informed Graph Neural Network for oscillation prediction
    Novel: Combines swing equation constraints with graph attention and power flow physics
    """
    
    def __init__(self, n_buses: int = 118, n_features: int = 12, 
                 hidden_dim: int = 256, n_oscillation_modes: int = 8,
                 n_time_steps: int = 100):
        super().__init__()
        
        self.n_buses = n_buses
        self.hidden_dim = hidden_dim
        self.n_time_steps = n_time_steps
        
        # Input encoding with physical significance
        self.voltage_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.angle_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.frequency_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.power_encoder = nn.Sequential(
            nn.Linear(2, 64),  # P and Q
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Feature fusion
        self.fusion_layer = nn.Linear(32*3 + 64, hidden_dim)
        
        # Graph physics layers
        self.graph_physics_1 = GraphPhysicsLayer(hidden_dim, hidden_dim, n_buses)
        self.graph_physics_2 = GraphPhysicsLayer(hidden_dim, hidden_dim, n_buses)
        self.graph_physics_3 = GraphPhysicsLayer(hidden_dim, hidden_dim, n_buses)
        
        # Swing equation layers (temporal physics)
        self.swing_layer = SwingEquationLayer(hidden_dim, n_buses)
        
        # Temporal attention for oscillation detection
        self.temporal_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Oscillation mode classifier
        self.oscillation_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, n_oscillation_modes)
        )
        
        # Blackout risk predictor (multi-horizon)
        self.risk_predictor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for _ in range(5)  # 5 prediction horizons: 1min, 5min, 15min, 30min, 60min
        ])
        
        # Uncertainty estimator using Monte Carlo Dropout
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(32, 2)  # Mean and log variance
        )
        
        # Attention weights for interpretability
        self.register_buffer('attention_weights', torch.zeros(n_buses, n_buses))
        
    def forward(self, voltages: torch.Tensor, angles: torch.Tensor, 
                frequencies: torch.Tensor, powers: torch.Tensor,
                edge_index: torch.Tensor, edge_attr: torch.Tensor,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with full physics constraints
        
        Args:
            voltages: [batch, time, nodes] voltage magnitudes
            angles: [batch, time, nodes] voltage angles
            frequencies: [batch, time, nodes] frequency deviations
            powers: [batch, time, nodes, 2] active and reactive power
            edge_index: [2, edges] graph connectivity
            edge_attr: [edges, 4] line parameters (R, X, G, B)
        
        Returns:
            Dictionary with predictions, uncertainties, and physics residuals
        """
        batch_size, n_time, n_nodes = voltages.shape
        
        # Encode each physical quantity
        v_encoded = self.voltage_encoder(voltages.permute(0, 2, 1).unsqueeze(-1))
        a_encoded = self.angle_encoder(angles.permute(0, 2, 1).unsqueeze(-1))
        f_encoded = self.frequency_encoder(frequencies.permute(0, 2, 1).unsqueeze(-1))
        p_encoded = self.power_encoder(powers.permute(0, 3, 1, 2))
        
        # Reshape for fusion
        v_encoded = v_encoded.permute(0, 2, 1, 3).reshape(batch_size, n_time, n_nodes, -1)
        a_encoded = a_encoded.permute(0, 2, 1, 3).reshape(batch_size, n_time, n_nodes, -1)
        f_encoded = f_encoded.permute(0, 2, 1, 3).reshape(batch_size, n_time, n_nodes, -1)
        
        # Fuse features
        h = torch.cat([v_encoded, a_encoded, f_encoded, p_encoded], dim=-1)
        h = self.fusion_layer(h)
        
        # Process each time step with graph physics
        h_time = []
        delta_history = []
        omega_history = []
        
        # Initialize states
        delta = angles[:, 0, :]  # Initial angle
        omega = frequencies[:, 0, :] - 60.0  # Frequency deviation
        
        for t in range(n_time):
            # Graph physics layers
            h_t = h[:, t, :, :]
            
            # Apply graph physics with current voltage and angle
            h_t = self.graph_physics_1(h_t, edge_index, voltages[:, t, :], angles[:, t, :])
            h_t = F.gelu(h_t)
            h_t = self.graph_physics_2(h_t, edge_index, voltages[:, t, :], angles[:, t, :])
            h_t = F.gelu(h_t)
            h_t = self.graph_physics_3(h_t, edge_index, voltages[:, t, :], angles[:, t, :])
            
            # Apply swing equation physics
            P_m = powers[:, t, :, 0]  # Mechanical power
            P_e = powers[:, t, :, 1]  # Electrical power
            delta, omega, h_t = self.swing_layer(h_t, delta, omega, P_m, P_e)
            
            delta_history.append(delta)
            omega_history.append(omega)
            h_time.append(h_t)
        
        # Stack temporal features
        h_stack = torch.stack(h_time, dim=1)  # [batch, time, nodes, hidden]
        
        # Temporal attention for oscillation pattern recognition
        h_flat = h_stack.view(batch_size, n_time, -1)
        h_attended, attn_weights = self.temporal_attention(h_flat, h_flat, h_flat)
        
        # Global pooling over time and nodes
        h_pooled = h_attended.mean(dim=1)  # [batch, hidden*nodes]
        
        # Multi-horizon risk predictions
        risk_predictions = []
        for horizon_head in self.risk_predictor:
            risk = horizon_head(h_pooled)
            risk_predictions.append(risk)
        
        # Oscillation mode classification
        oscillation_logits = self.oscillation_classifier(h_pooled)
        oscillation_probs = F.softmax(oscillation_logits, dim=-1)
        
        # Uncertainty estimation
        uncertainty_params = self.uncertainty_head(h_pooled)
        uncertainty_mean = uncertainty_params[:, 0]
        uncertainty_logvar = uncertainty_params[:, 1]
        uncertainty_std = torch.exp(0.5 * uncertainty_logvar)
        
        # Store attention for interpretability
        if return_attention:
            self.attention_weights = attn_weights.mean(dim=0)
        
        # Compute physics residuals (for loss function)
        physics_residuals = self.compute_physics_residuals(
            torch.stack(delta_history, dim=1),
            torch.stack(omega_history, dim=1),
            powers
        )
        
        return {
            'oscillation_probs': oscillation_probs,
            'oscillation_mode': torch.argmax(oscillation_probs, dim=-1),
            'risk_predictions': torch.stack(risk_predictions, dim=-1),  # [batch, horizons]
            'risk_1min': risk_predictions[0].squeeze(),
            'risk_5min': risk_predictions[1].squeeze(),
            'risk_15min': risk_predictions[2].squeeze(),
            'risk_30min': risk_predictions[3].squeeze(),
            'risk_60min': risk_predictions[4].squeeze(),
            'uncertainty_mean': uncertainty_mean,
            'uncertainty_std': uncertainty_std,
            'physics_residuals': physics_residuals,
            'delta_history': torch.stack(delta_history, dim=1),
            'omega_history': torch.stack(omega_history, dim=1),
            'hidden_states': h_stack,
            'attention_weights': attn_weights if return_attention else None
        }
    
    def compute_physics_residuals(self, delta_history: torch.Tensor, 
                                   omega_history: torch.Tensor,
                                   powers: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-informed loss residuals
        Ensures the model respects swing equation constraints
        """
        # Compute angular acceleration
        d_delta = torch.diff(delta_history, dim=1)
        d_omega = torch.diff(omega_history, dim=1)
        
        # Time step (assumed constant)
        dt = 0.01
        
        # Numerical derivatives
        d2delta_dt2 = d_omega / dt
        
        # Swing equation residual: M * d²δ/dt² + D * dδ/dt - (P_m - P_e) = 0
        P_m = powers[:, 1:, :, 0]  # Mechanical power (excluding first time step)
        P_e = powers[:, 1:, :, 1]  # Electrical power
        
        # Get physical parameters (simplified)
        M = torch.ones_like(d2delta_dt2) * 5.0  # Placeholder inertia
        D = torch.ones_like(d2delta_dt2) * 2.0  # Placeholder damping
        
        swing_residual = M * d2delta_dt2 + D * d_omega / dt - (P_m - P_e)
        
        # L2 norm of residuals
        residual_loss = torch.mean(swing_residual ** 2)
        
        return residual_loss

class PhysicsConstrainedLoss(nn.Module):
    """
    Multi-objective loss function with physics constraints
    """
    
    def __init__(self, lambda_physics: float = 0.1, lambda_uncertainty: float = 0.05):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.lambda_uncertainty = lambda_uncertainty
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute total loss with physics constraints
        """
        # Classification loss for oscillation mode
        ce_loss = F.cross_entropy(
            predictions['oscillation_probs'], 
            targets['oscillation_mode']
        )
        
        # Regression loss for risk predictions (multi-horizon)
        mse_loss = 0.0
        for i in range(5):
            horizon_key = f'risk_{["1min", "5min", "15min", "30min", "60min"][i]}'
            mse_loss += F.mse_loss(
                predictions[horizon_key], 
                targets[f'risk_{["1min", "5min", "15min", "30min", "60min"][i]}']
            )
        mse_loss = mse_loss / 5
        
        # Physics-informed loss
        physics_loss = predictions['physics_residuals']
        
        # Uncertainty loss (negative log likelihood)
        uncertainty_loss = 0.5 * (
            torch.log(predictions['uncertainty_std']**2) + 
            ((targets['risk_15min'] - predictions['risk_15min'])**2) / (predictions['uncertainty_std']**2 + 1e-8)
        ).mean()
        
        # Total loss
        total_loss = ce_loss + mse_loss + self.lambda_physics * physics_loss + self.lambda_uncertainty * uncertainty_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'mse_loss': mse_loss,
            'physics_loss': physics_loss,
            'uncertainty_loss': uncertainty_loss
        }
