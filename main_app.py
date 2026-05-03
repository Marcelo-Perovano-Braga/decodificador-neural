# main_app.py
import sys
import os
import numpy as np
import torch
import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QComboBox, QLabel, QSlider, QGroupBox)
from PyQt6.QtCore import Qt
import qdarktheme

# Importa as estruturas neurais vazias
from models import NeuronClassifier1D, PredictorExpert

# ==========================================
# ROTEADOR DE INFERÊNCIA (MIXTURE OF EXPERTS)
# ==========================================
def predict_live_neuron_state(live_voltage_window, classifier, experts_dict):
    """Passa os dados pelo Classificador e roteia para o Especialista correto."""
    with torch.no_grad():
        class_logits = classifier(live_voltage_window)
        predicted_class_idx = torch.argmax(class_logits, dim=1).item()
        
        class_map = {0: 'RS', 1: 'CH', 2: 'FS'}
        neuron_type = class_map[predicted_class_idx]
        
        selected_expert = experts_dict[neuron_type]
        future_prediction = selected_expert(live_voltage_window)
        
        return neuron_type, future_prediction.item()

# (Motor de simulação reduzido para gerar dados da UI)
def simulate_single(a, b, c, d, I_injected, time_ms=200, dt=0.1):
    steps = int(time_ms / dt)
    v, u = np.zeros(steps), np.zeros(steps)
    v[0], u[0] = -65.0, b * -65.0
    for t in range(steps - 1):
        if v[t] >= 30.0:
            v[t], v[t+1], u[t+1] = 30.0, c, u[t] + d
        else:
            dv = 0.04 * v[t]**2 + 5 * v[t] + 140 - u[t] + I_injected
            du = a * (b * v[t] - u[t])
            v[t+1], u[t+1] = v[t] + dv * dt, u[t] + du * dt
    return np.arange(steps)*dt, v

# ==========================================
# INTERFACE GRÁFICA
# ==========================================
class IzhikevichApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FAPES - Exposição: Dinâmica Neuronal (MoE AI)")
        self.resize(1100, 650)
        self.device = torch.device('cpu') # Força CPU para estabilidade da UI no .exe
        
        # 1. Carrega as Redes Neurais (Estágio 1 e 2)
        self.load_ai_models()
        
        # 2. Configura a Interface
        self.setup_ui()
        self.update_data()

    def load_ai_models(self):
        """Instancia as redes vazias e carrega os pesos do disco."""
        window_size = 2000 # 200ms com dt=0.1
        
        self.classifier = NeuronClassifier1D(window_size=window_size).to(self.device)
        self.experts = {
            'RS': PredictorExpert(window_size=window_size).to(self.device),
            'CH': PredictorExpert(window_size=window_size).to(self.device),
            'FS': PredictorExpert(window_size=window_size).to(self.device)
        }
        
        # Tenta carregar pesos (Lida graciosamente se os arquivos não existirem ainda)
        try:
            # Para o .exe, o PyInstaller descompacta arquivos em sys._MEIPASS
            base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            weight_dir = os.path.join(base_path, 'weights')
            
            self.classifier.load_state_dict(torch.load(os.path.join(weight_dir, 'classifier.pth'), map_location=self.device))
            self.experts['RS'].load_state_dict(torch.load(os.path.join(weight_dir, 'expert_rs.pth'), map_location=self.device))
            self.experts['CH'].load_state_dict(torch.load(os.path.join(weight_dir, 'expert_ch.pth'), map_location=self.device))
            self.experts['FS'].load_state_dict(torch.load(os.path.join(weight_dir, 'expert_fs.pth'), map_location=self.device))
            
            self.classifier.eval()
            for exp in self.experts.values(): exp.eval()
            self.ai_ready = True
        except Exception as e:
            print(f"[AVISO] Pesos da IA não encontrados. Modo simulação apenas. Erro: {e}")
            self.ai_ready = False

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        control_group = QGroupBox("Controles e Análise de IA")
        control_group.setStyleSheet("font-size: 14px; font-weight: bold; color: lightgray;")
        control_panel = QVBoxLayout()
        control_group.setLayout(control_panel)
        
        self.combo_type = QComboBox()
        self.combo_type.addItems(["Regular Spiking", "Chattering", "Fast Spiking"])
        self.combo_type.currentIndexChanged.connect(self.update_data)
        
        self.slider_current = QSlider(Qt.Orientation.Horizontal)
        self.slider_current.setRange(0, 300)
        self.slider_current.setValue(100)
        self.slider_current.valueChanged.connect(self.update_data)
        
        self.label_ai_pred = QLabel("IA Identificação: Aguardando...")
        self.label_ai_pred.setStyleSheet("color: #ff00ff; margin-top: 30px; font-size: 16px;")

        control_panel.addWidget(QLabel("Tipo Biológico:"))
        control_panel.addWidget(self.combo_type)
        control_panel.addWidget(QLabel("Intensidade da Corrente:"))
        control_panel.addWidget(self.slider_current)
        control_panel.addWidget(self.label_ai_pred)
        control_panel.addStretch()
        
        pg.setConfigOptions(antialias=True)
        self.plot_graph = pg.PlotWidget(background='#1e1e1e')
        self.plot_graph.setYRange(-90, 40, padding=0)
        self.plot_graph.setXRange(0, 200, padding=0)
        self.curve = self.plot_graph.plot([], [], pen=pg.mkPen(color='#00ffcc', width=2.5))
        
        layout.addWidget(control_group, stretch=1)
        layout.addWidget(self.plot_graph, stretch=3)

    def update_data(self):
        neuron_type = self.combo_type.currentText()
        current = self.slider_current.value() / 10.0
        
        params = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8}
        if "Chattering" in neuron_type: params.update({'c': -50, 'd': 2})
        elif "Fast" in neuron_type: params.update({'a': 0.1, 'd': 2})
            
        time, v = simulate_single(params['a'], params['b'], params['c'], params['d'], current)
        self.curve.setData(time, v)
        
        # Fluxo de dados para a IA
        if self.ai_ready:
            # Formata os dados no shape esperado pela 1D CNN: (Batch=1, Canal=1, Janela=2000)
            v_tensor = torch.tensor(v, dtype=torch.float32).view(1, 1, -1).to(self.device)
            identidade, prox_estado = predict_live_neuron_state(v_tensor, self.classifier, self.experts)
            self.label_ai_pred.setText(f"IA Detectou: {identidade} | Previsão V(t+1): {prox_estado:.2f}mV")

if __name__ == "__main__":
    torch.set_num_threads(1)
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")
    window = IzhikevichApp()
    window.show()
    sys.exit(app.exec())