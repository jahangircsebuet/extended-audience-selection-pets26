# privaRAG_agent/federated.py
import torch
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset
from config import DP_NOISE, CLIP_NORM, BATCH_SIZE

def train_federated(model, X, y, lr=1e-3, epochs=5):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        target_epsilon=3.0,
        target_delta=1e-5,
        max_grad_norm=CLIP_NORM,
    )

    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = torch.nn.BCELoss()(model(xb).squeeze(), yb)
            loss.backward()
            optimizer.step()
    return model
