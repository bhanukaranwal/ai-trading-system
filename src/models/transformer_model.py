import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from pathlib import Path
from loguru import logger
import yaml
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, 
                 dim_feedforward: int, dropout: float):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        output = self.fc(x)
        return output.squeeze()


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TransformerTrainer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.params = self.config['models']['transformer']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        
        logger.info(f"Using device: {self.device}")
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        X_seq, y_seq = [], []
        
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, input_size: int):
        self.model = TransformerModel(
            input_size=input_size,
            d_model=self.params['d_model'],
            nhead=self.params['nhead'],
            num_layers=self.params['num_layers'],
            dim_feedforward=self.params['dim_feedforward'],
            dropout=self.params['dropout']
        ).to(self.device)
        return self.model
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        X_scaled = self.scaler.fit_transform(X)
        
        seq_length = self.params['sequence_length']
        X_seq, y_seq = self.create_sequences(X_scaled, y, seq_length)
        
        split_idx = int(len(X_seq) * (1 - test_size))
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params['batch_size'],
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.params['batch_size'],
            shuffle=False
        )
        
        if self.model is None:
            self.build_model(X.shape[1])
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        
        logger.info(f"Training Transformer for {self.params['epochs']} epochs")
        
        for epoch in range(self.params['epochs']):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            self.model.eval()
            test_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    test_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.params['epochs']}], Train Loss: {train_loss/len(train_loader):.6f}, Test Loss: {test_loss/len(test_loader):.6f}")
        
        return {
            'final_train_loss': train_loss / len(train_loader),
            'final_test_loss': test_loss / len(test_loader)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        seq_length = self.params['sequence_length']
        X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X)), seq_length)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
    
    def save_model(self, filepath: str = "models/transformer_model.pth"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'params': self.params
        }, filepath)
        logger.success(f"Transformer model saved to {filepath}")
    
    def load_model(self, filepath: str = "models/transformer_model.pth", input_size: int = None):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.scaler = checkpoint['scaler']
        self.params = checkpoint['params']
        
        if input_size is not None:
            self.build_model(input_size)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logger.success(f"Transformer model loaded from {filepath}")
        else:
            raise ValueError("input_size must be provided when loading model")
