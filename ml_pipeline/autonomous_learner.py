import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
import threading
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import deque

class UGIMCore(nn.Module):
    def __init__(self, input_dim: int = 15, d_model: int = 128, num_modes: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_modes = num_modes
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.mean(dim=1)
        return self.network(x).squeeze(-1)

class ExperienceBuffer:
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, features: np.ndarray, target: float):
        self.buffer.append({
            'features': features.tolist(),
            'target': target,
            'timestamp': datetime.now().isoformat()
        })
    
    def sample(self, batch_size: int) -> tuple:
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        features = []
        targets = []
        for i in indices:
            features.append(self.buffer[i]['features'])
            targets.append(self.buffer[i]['target'])
        return np.array(features), np.array(targets)
    
    def get_all_data(self) -> tuple:
        features = [item['features'] for item in self.buffer]
        targets = [item['target'] for item in self.buffer]
        return np.array(features), np.array(targets)
    
    def size(self) -> int:
        return len(self.buffer)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(list(self.buffer), f, indent=2)
    
    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
            self.buffer = deque(data, maxlen=self.buffer.maxlen)

class AutonomousLearner:
    def __init__(self, model_dir: str = "ml_pipeline/models", data_dir: str = "ml_pipeline/data"):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model_path = f"{model_dir}/production_model.pt"
        self.metrics_path = f"{model_dir}/metrics.json"
        
        self.model = None
        self.buffer = ExperienceBuffer()
        self.is_training = False
        self.training_thread = None
        self.should_stop = False
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        self.load_model()
        self.load_buffer()
    
    def load_model(self):
        self.model = UGIMCore()
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint)
            print(f"Loaded model from {self.model_path}")
        else:
            print("Created new model")
        self.model.eval()
    
    def load_buffer(self):
        buffer_path = f"{self.data_dir}/experience_buffer.json"
        if os.path.exists(buffer_path):
            self.buffer.load(buffer_path)
            print(f"Loaded {self.buffer.size()} experience samples")
    
    def save_buffer(self):
        buffer_path = f"{self.data_dir}/experience_buffer.json"
        self.buffer.save(buffer_path)
    
    def add_experience(self, features: np.ndarray, target: float):
        self.buffer.add(features, target)
        self.save_buffer()
        
        if self.buffer.size() >= 100 and not self.is_training:
            self.trigger_async_training()
    
    def prepare_training_data(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        if len(features.shape) == 2:
            features = features.reshape(features.shape[0], 1, features.shape[1])
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        
        return features_tensor, targets_tensor
    
    def train(self, epochs: int = 30, batch_size: int = 32) -> Dict[str, float]:
        if self.buffer.size() < 50:
            return {"status": "insufficient_data", "samples": self.buffer.size()}
        
        features, targets = self.buffer.get_all_data()
        features_tensor, targets_tensor = self.prepare_training_data(features, targets)
        
        dataset = TensorDataset(features_tensor, targets_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        split = int(0.8 * len(features_tensor))
        val_features = features_tensor[split:]
        val_targets = targets_tensor[split:]
        val_dataset = TensorDataset(val_features, val_targets)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        model = UGIMCore()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch_features, batch_targets in loader:
                optimizer.zero_grad()
                output = model(batch_features)
                loss = criterion(output, batch_targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    output = model(batch_features)
                    val_loss += criterion(output, batch_targets).item()
            
            train_loss /= len(loader)
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{self.model_dir}/staging_model.pt")
        
        old_metrics = self.get_metrics()
        old_loss = old_metrics.get('val_loss', float('inf'))
        improvement = ((old_loss - best_val_loss) / old_loss) * 100 if old_loss != float('inf') else 100
        
        if best_val_loss < old_loss:
            os.rename(f"{self.model_dir}/staging_model.pt", self.model_path)
            self.load_model()
            
            metrics = {
                'val_loss': best_val_loss,
                'improvement': improvement,
                'training_samples': self.buffer.size(),
                'timestamp': datetime.now().isoformat(),
                'status': 'deployed'
            }
            with open(self.metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"New model deployed. Improvement: {improvement:.2f}%")
        else:
            print(f"Model did not improve. Keeping existing model.")
        
        return {'val_loss': best_val_loss, 'improvement': improvement}
    
    def trigger_async_training(self):
        if self.training_thread and self.training_thread.is_alive():
            return
        
        self.training_thread = threading.Thread(target=self._train_async)
        self.training_thread.start()
    
    def _train_async(self):
        self.is_training = True
        print(f"Starting autonomous training with {self.buffer.size()} samples")
        result = self.train()
        print(f"Training complete: {result}")
        self.is_training = False
    
    def get_metrics(self) -> Dict[str, Any]:
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, 'r') as f:
                return json.load(f)
        return {'val_loss': float('inf'), 'improvement': 0}
    
    def predict(self, features: np.ndarray) -> float:
        self.model.eval()
        with torch.no_grad():
            if len(features.shape) == 1:
                features = features.reshape(1, 1, -1)
            x = torch.tensor(features, dtype=torch.float32)
            return self.model(x).item()

autonomous_learner = AutonomousLearner()
