# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuronClassifier1D(nn.Module):
    """Estágio 1: Classificador de padrões de disparo (RS, CH, FS)"""
    def __init__(self, window_size=2000, num_classes=3):
        super(NeuronClassifier1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        
        def conv1d_out_size(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        conv1_out = conv1d_out_size(window_size)
        conv2_out = conv1d_out_size(conv1_out)
        self.flatten_size = 32 * conv2_out
        
        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PredictorExpert(nn.Module):
    """Estágio 2: Especialista para previsão de dinâmica de um subtipo específico"""
    def __init__(self, window_size=2000):
        super(PredictorExpert, self).__init__()
        self.fc1 = nn.Linear(window_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1) # Prevê o próximo v(t) ou métrica específica

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)