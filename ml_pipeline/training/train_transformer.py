import sys
import os

# Add the models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../models'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split

from ugim_transformer import UGIMTransformer

def generate_synthetic_training_data(n_samples=10000):
    np.random.seed(42)
    demand = np.random.uniform(1400, 4306, n_samples)
    load_factor = demand / 4306
    frequency = 50.0 - (load_factor - 0.5) * 0.3 + np.random.normal(0, 0.05, n_samples)
    frequency = np.clip(frequency, 49.5, 50.5)
    voltage = 1.0 - (load_factor - 0.5) * 0.1 + np.random.normal(0, 0.01, n_samples)
    voltage = np.clip(voltage, 0.94, 1.06)
    risk = load_factor * 0.33 + np.random.normal(0, 0.03, n_samples)
    risk = np.clip(risk, 0, 1)
    return demand, frequency, voltage, risk

def train():
    print("="*60)
    print("Training UGIM Transformer Model")
    print("="*60)
    
    demand, frequency, voltage, risk = generate_synthetic_training_data(10000)
    
    X = np.stack([demand, frequency, voltage], axis=1)
    y = risk
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train[:, 0]),
        torch.FloatTensor(X_train[:, 1]),
        torch.FloatTensor(X_train[:, 2]),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val[:, 0]),
        torch.FloatTensor(X_val[:, 1]),
        torch.FloatTensor(X_val[:, 2]),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = UGIMTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for d, f, v, y in train_loader:
            optimizer.zero_grad()
            pred = model(d, f, v)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for d, f, v, y in val_loader:
                pred = model(d, f, v)
                val_loss += criterion(pred, y).item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {total_loss/len(train_loader):.4f}, Val Loss = {val_loss/len(val_loader):.4f}")
    
    # Save model to backend folder for easy access
    torch.save(model.state_dict(), '../models/ugim_transformer_v2.pt')
    print("\nModel saved to ml_pipeline/models/ugim_transformer_v2.pt")
    
    model.eval()
    test_demand = torch.tensor([2600.0])
    test_freq = torch.tensor([49.98])
    test_voltage = torch.tensor([0.99])
    test_risk = model(test_demand, test_freq, test_voltage)
    print(f"\nTest prediction for demand=2600 MW, freq=49.98 Hz: Risk = {test_risk.item()*100:.1f}%")

if __name__ == "__main__":
    train()
