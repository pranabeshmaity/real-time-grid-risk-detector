import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from datetime import datetime

print("="*60)
print("UGIM Training Pipeline for IEEE Publication")
print("="*60)
print("\nTo generate real results, you need:")
print("1. Historical PMU data from NERLDC (India grid)")
print("2. Oscillation event labels")
print("3. Training/validation/test split")

latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Performance comparison for grid oscillation prediction}
\label{tab:comparison}
\begin{tabular}{lccc}
\hline
Model & RMSE $\downarrow$ & MAE $\downarrow$ & R² $\uparrow$ \\
\hline
LSTM & 0.087 & 0.065 & 0.89 \\
Transformer & 0.072 & 0.054 & 0.91 \\
Standard GNN & 0.063 & 0.048 & 0.93 \\
\textbf{UGIM (Ours)} & \textbf{0.041} & \textbf{0.029} & \textbf{0.96} \\
\hline
\end{tabular}
\end{table}
"""

print(latex_table)
print("\nReplace example numbers with your actual trained results")
