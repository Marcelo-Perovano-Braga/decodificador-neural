import torch
import torch.nn as nn
import torch.optim as optim
import os
from models import NeuronClassifier1D, PredictorExpert

# ==========================================
# INTEGRADOR VETORIZADO PARA GERAÇÃO DE DADOS
# ==========================================
def simulate_batch(a, b, c, d, I_injected, time_ms=200, dt=0.1, device='cpu'):
    steps = int(time_ms / dt)
    N = a.shape[0]
    v = torch.zeros((steps, N), device=device)
    u = torch.zeros((steps, N), device=device)
    v[0], u[0] = -65.0, b * -65.0

    for t in range(steps - 1):
        is_spiking = v[t] >= 30.0
        v[t, is_spiking] = 30.0
        dv = 0.04 * v[t]**2 + 5.0 * v[t] + 140.0 - u[t] + I_injected
        du = a * (b * v[t] - u[t])
        v[t+1] = v[t] + dv * dt
        u[t+1] = u[t] + du * dt
        v[t+1, is_spiking] = c[is_spiking]
        u[t+1, is_spiking] = u[t, is_spiking] + d[is_spiking]
    return v.transpose(0, 1)

# ==========================================
# LOOP DE TREINAMENTO BASE (CORRIGIDO)
# ==========================================
def train_networks():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[SISTEMA] Motor de Hardware: {device}")
    
    # Hiperparâmetros
    SAMPLES_PER_CLASS = 4000  # Total de 12.000 neurônios simulados simultaneamente
    TIME_MS = 200
    DT = 0.1
    WINDOW_SIZE = int(TIME_MS / DT)
    EPOCHS = 1000
    LR = 0.001

    print("[DADOS] Sintetizando matrizes biológicas na VRAM...")
    
    # Geração dos Parâmetros Izhikevich
    # RS (0) | CH (1) | FS (2)
    a = torch.cat([torch.full((SAMPLES_PER_CLASS,), 0.02), torch.full((SAMPLES_PER_CLASS,), 0.02), torch.full((SAMPLES_PER_CLASS,), 0.1)]).to(device)
    b = torch.cat([torch.full((SAMPLES_PER_CLASS,), 0.2),  torch.full((SAMPLES_PER_CLASS,), 0.2),  torch.full((SAMPLES_PER_CLASS,), 0.2)]).to(device)
    c = torch.cat([torch.full((SAMPLES_PER_CLASS,), -65.0),torch.full((SAMPLES_PER_CLASS,), -50.0),torch.full((SAMPLES_PER_CLASS,), -65.0)]).to(device)
    d = torch.cat([torch.full((SAMPLES_PER_CLASS,), 8.0),  torch.full((SAMPLES_PER_CLASS,), 2.0),  torch.full((SAMPLES_PER_CLASS,), 2.0)]).to(device)
    
    # Corrente variada para evitar overfitting (5 a 20)
    I_injected = (torch.rand(SAMPLES_PER_CLASS * 3) * 15 + 5).to(device)
    
    # Rótulos (Labels) para o Classificador
    labels = torch.cat([torch.zeros(SAMPLES_PER_CLASS), torch.ones(SAMPLES_PER_CLASS), torch.full((SAMPLES_PER_CLASS,), 2)]).long().to(device)

    # Executa a simulação biológica em paralelo na GPU
    v_data = simulate_batch(a, b, c, d, I_injected, time_ms=TIME_MS, dt=DT, device=device)
    
    # O Classificador espera dimensão (Batch, Canais, Janela) -> (12000, 1, 2000)
    v_inputs = v_data.unsqueeze(1)
    
    # Para o Especialista, definiremos como alvo prever a tensão final baseada nos dados anteriores
    # Isso simula o seu "Prevê o próximo v(t)"
    expert_targets = v_data[:, -1].unsqueeze(1) 

    # 1. Inicializa Modelos e Otimizadores
    classifier = NeuronClassifier1D(window_size=WINDOW_SIZE).to(device)
    experts = {
        0: PredictorExpert(window_size=WINDOW_SIZE).to(device), # RS
        1: PredictorExpert(window_size=WINDOW_SIZE).to(device), # CH
        2: PredictorExpert(window_size=WINDOW_SIZE).to(device)  # FS
    }

    opt_classifier = optim.Adam(classifier.parameters(), lr=LR)
    opts_experts = {
        0: optim.Adam(experts[0].parameters(), lr=LR),
        1: optim.Adam(experts[1].parameters(), lr=LR),
        2: optim.Adam(experts[2].parameters(), lr=LR)
    }

    criterion_class = nn.CrossEntropyLoss()
    criterion_expert = nn.MSELoss()

    print("[TREINAMENTO] Iniciando otimização de pesos (Backpropagation)...")
    
    # Loop de Treinamento
    for epoch in range(EPOCHS):
        # Treino do Classificador (Estágio 1)
        opt_classifier.zero_grad()
        class_preds = classifier(v_inputs)
        loss_class = criterion_class(class_preds, labels)
        loss_class.backward()
        opt_classifier.step()

        # Treino dos Especialistas (Estágio 2)
        loss_experts_total = 0.0
        for class_idx in range(3):
            # Isola os tensores pertencentes apenas ao especialista atual
            mask = (labels == class_idx)
            expert_in = v_inputs[mask]
            expert_tgt = expert_targets[mask]
            
            opts_experts[class_idx].zero_grad()
            expert_pred = experts[class_idx](expert_in)
            loss_exp = criterion_expert(expert_pred, expert_tgt)
            loss_exp.backward()
            opts_experts[class_idx].step()
            
            loss_experts_total += loss_exp.item()

        if (epoch + 1) % 100 == 0:
            print(f"Época {epoch+1:04d}/{EPOCHS} | Loss Classificador: {loss_class.item():.4f} | Loss Especialistas Médio: {loss_experts_total/3:.4f}")

    # 2. Salva os pesos no disco
    os.makedirs('weights', exist_ok=True)
    torch.save(classifier.state_dict(), 'weights/classifier.pth')
    torch.save(experts[0].state_dict(), 'weights/expert_rs.pth')
    torch.save(experts[1].state_dict(), 'weights/expert_ch.pth')
    torch.save(experts[2].state_dict(), 'weights/expert_fs.pth')
    print("[SISTEMA] Arquitetura MoE otimizada com sucesso. Matrizes exportadas para /weights.")

if __name__ == "__main__":
    train_networks()