"""
FinScope AI - Deep Neural Network Model (PyTorch)

Multi-layer feed-forward neural network for credit risk prediction
with dropout regularization and batch normalization.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader, TensorDataset

from models.base_model import BaseModel


class CreditRiskDNN(nn.Module):
    """PyTorch neural network architecture for credit risk."""

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (128, 64, 32), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Output layer (single neuron, sigmoid for probability)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class DNNModel(BaseModel):
    """Deep Neural Network model for credit risk prediction."""

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 100,
        patience: int = 10,
    ):
        super().__init__(name="dnn")
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.input_dim: Optional[int] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network: Optional[CreditRiskDNN] = None

    def _build_model(self, input_dim: int) -> None:
        """Initialize the neural network."""
        self.input_dim = input_dim
        self.network = CreditRiskDNN(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

    def _create_dataloader(self, X: np.ndarray, y: Optional[np.ndarray] = None, shuffle: bool = True) -> DataLoader:
        """Create PyTorch DataLoader from numpy arrays."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        if y is not None:
            y_tensor = torch.FloatTensor(y).to(self.device)
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs,
    ) -> Dict[str, float]:
        """Train the DNN with early stopping."""
        logger.info(
            f"Training {self.name} | train={X_train.shape} | val={X_val.shape} | "
            f"device={self.device} | epochs={self.epochs}"
        )

        self._build_model(X_train.shape[1])

        # Class weight for imbalanced data
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        pos_weight = torch.FloatTensor([n_neg / max(n_pos, 1)]).to(self.device)

        criterion = nn.BCELoss(weight=None)  # Using balanced batches approach
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        val_loader = self._create_dataloader(X_val, y_val, shuffle=False)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training
            self.network.train()
            train_losses = []
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.network(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            self.network.eval()
            val_losses = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.network(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            scheduler.step(avg_val_loss)

            if (epoch + 1) % 10 == 0:
                logger.debug(
                    f"Epoch {epoch + 1}/{self.epochs} | "
                    f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}"
                )

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Restore best model
        if best_state is not None:
            self.network.load_state_dict(best_state)
            self.network.to(self.device)

        self._is_trained = True

        # Final evaluation
        train_proba = self.predict_proba(X_train)
        val_proba = self.predict_proba(X_val)

        metrics = {
            "train_auc": roc_auc_score(y_train, train_proba),
            "val_auc": roc_auc_score(y_val, val_proba),
            "train_logloss": log_loss(y_train, train_proba),
            "val_logloss": log_loss(y_val, val_proba),
            "best_epoch": epoch + 1 - patience_counter,
            "total_epochs": epoch + 1,
        }

        logger.info(
            f"{self.name} trained | train_auc={metrics['train_auc']:.4f} | "
            f"val_auc={metrics['val_auc']:.4f} | best_epoch={metrics['best_epoch']}"
        )
        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return default probability for each sample."""
        if not self._is_trained or self.network is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        self.network.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            # Process in batches to avoid OOM
            predictions = []
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i : i + self.batch_size]
                pred = self.network(batch).cpu().numpy()
                predictions.append(pred)

        return np.concatenate(predictions)

    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / f"{self.name}.pt"
        torch.save(
            {
                "model_state_dict": self.network.state_dict(),
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
            },
            filepath,
        )
        logger.info(f"Model saved to {filepath}")

    def load(self, path: str) -> "DNNModel":
        """Load model from disk."""
        path = Path(path)
        filepath = path / f"{self.name}.pt"
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.input_dim = checkpoint["input_dim"]
        self.hidden_dims = checkpoint["hidden_dims"]
        self.dropout = checkpoint["dropout"]

        self._build_model(self.input_dim)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.network.eval()
        self._is_trained = True

        logger.info(f"Model loaded from {filepath}")
        return self
