# privaRAG_agent/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PostClassifier(nn.Module):
    """Binary classifier for post sensitivity"""
    def __init__(self, input_dim, hidden_dim=64):
        super(PostClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

class FriendInclusionModel(nn.Module):
    """Logistic regression for friend inclusion"""
    def __init__(self, input_dim=4):
        super(FriendInclusionModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))
