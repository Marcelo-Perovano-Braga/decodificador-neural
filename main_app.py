"""Fluent desktop interface for simulated Izhikevich pattern classification."""

from __future__ import annotations

import sys
from pathlib import Path

import pyqtgraph as pg
import torch
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    ComboBox,
    InfoBadge,
    InfoLevel,
    ProgressBar,
    SimpleCardWidget,
    Slider,
    StrongBodyLabel,
    SubtitleLabel,
    SwitchButton,
    Theme,
    TitleLabel,
    setTheme,
    setThemeColor,
)

from config import CLASS_CODES, CLASS_NAMES, IZHIKEVICH_PARAMETERS, SimulationConfig
from inference import classify_trace
from models import NeuronClassifier1D
from simulation import count_spikes, simulate_izhikevich_single


DARK_PALETTE = {
    "window": "#0f1115",
    "plot": "#15191f",
    "text": "#d9e2ec",
    "muted": "#94a3b8",
    "curve": "#00e5c0",
    "threshold": "#536170",
}

LIGHT_PALETTE = {
    "window": "#f3f6f9",
    "plot": "#ffffff",
    "text": "#344054",
    "muted": "#667085",
    "curve": "#008f7a",
    "threshold": "#98a2b3",
}

WINDOWS_APP_ID = "Izhikevich.NeuralPatternClassifier"


def resource_root() -> Path:
    bundled = getattr(sys, "_MEIPASS", None)
    return Path(bundled) if bundled else Path(__file__).resolve().parent


def application_icon() -> QIcon:
    """Load the best icon format for the current platform."""

    icon_name = "app_icon.ico" if sys.platform == "win32" else "app_icon.png"
    icon_path = resource_root() / "assets" / icon_name
    return QIcon(str(icon_path)) if icon_path.is_file() else QIcon()


def configure_windows_process_identity() -> None:
    """Prevent Windows from grouping the window under the generic Python icon."""

    if sys.platform != "win32":
        return

    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
    except (AttributeError, OSError):
        # The window icon still works if this optional Windows API is unavailable.
        pass


def load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with PyTorch versions before weights_only.
        checkpoint = torch.load(path, map_location=device)
    required = {"model_state", "normalization", "simulation", "class_codes", "class_names"}
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(f"Checkpoint missing fields: {sorted(missing)}")
    return checkpoint


class MetricCard(SimpleCardWidget):
    """Compact summary card shown above the voltage trace."""

    def __init__(self, title: str, value: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(145)
        self.setFixedHeight(78)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(2)

        self.title_label = CaptionLabel(title, self)
        self.value_label = SubtitleLabel(value, self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class IzhikevichApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Dinâmica Neuronal — Classificador Izhikevich")
        self.setWindowIcon(application_icon())
        self.resize(1180, 720)
        self.setMinimumSize(980, 640)

        self.device = torch.device("cpu")
        self.model: NeuronClassifier1D | None = None
        self.checkpoint: dict[str, object] | None = None
        self.config = SimulationConfig()
        self.model_error: str | None = None
        self.is_light_theme = False

        self.load_ai_model()
        self.setup_ui()
        self.apply_visual_theme()
        self.update_data()

    def load_ai_model(self) -> None:
        checkpoint_path = resource_root() / "weights" / "classifier.pt"
        try:
            checkpoint = load_checkpoint(checkpoint_path, self.device)
            model = NeuronClassifier1D(num_classes=len(checkpoint["class_names"]))
            model.load_state_dict(checkpoint["model_state"])
            model.to(self.device).eval()
            simulation = checkpoint["simulation"]
            self.config = SimulationConfig(
                time_ms=float(simulation["time_ms"]),
                dt_ms=float(simulation["dt_ms"]),
                current_min=float(simulation["current_min"]),
                current_max=float(simulation["current_max"]),
                parameter_jitter=float(simulation.get("parameter_jitter", 0.0)),
                observation_noise_std=float(simulation.get("observation_noise_std", 0.0)),
                minimum_spikes=int(simulation["minimum_spikes"]),
            )
            self.model = model
            self.checkpoint = checkpoint
        except Exception as exc:
            self.model = None
            self.checkpoint = None
            self.model_error = str(exc)

    def setup_ui(self) -> None:
        self.root = QWidget()
        self.root.setObjectName("appRoot")
        self.setCentralWidget(self.root)

        page_layout = QVBoxLayout(self.root)
        page_layout.setContentsMargins(26, 22, 26, 24)
        page_layout.setSpacing(18)
        page_layout.addLayout(self._build_header())

        content_layout = QHBoxLayout()
        content_layout.setSpacing(18)
        content_layout.addWidget(self._build_control_panel(), 0)
        content_layout.addWidget(self._build_visualization_panel(), 1)
        page_layout.addLayout(content_layout, 1)

    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setSpacing(16)

        title_column = QVBoxLayout()
        title_column.setSpacing(2)
        title_column.addWidget(TitleLabel("Dinâmica neuronal simulada"))
        title_column.addWidget(
            CaptionLabel("Classificação de padrões RS, CH e FS com uma CNN 1D")
        )
        header.addLayout(title_column)
        header.addStretch(1)

        if self.model is None:
            self.model_badge = InfoBadge(
                "MODELO NÃO CARREGADO", level=InfoLevel.WARNING
            )
        else:
            self.model_badge = InfoBadge("MODELO PRONTO", level=InfoLevel.SUCCESS)
        self.model_badge.adjustSize()
        header.addWidget(self.model_badge, 0, Qt.AlignmentFlag.AlignVCenter)

        self.theme_switch = SwitchButton()
        self.theme_switch.setOnText("Tema claro")
        self.theme_switch.setOffText("Tema claro")
        self.theme_switch.setToolTip("Alternar entre os temas claro e escuro")
        self.theme_switch.checkedChanged.connect(self.on_theme_changed)
        header.addWidget(self.theme_switch, 0, Qt.AlignmentFlag.AlignVCenter)
        return header

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(370)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(14)

        simulation_card = SimpleCardWidget()
        simulation_layout = QVBoxLayout(simulation_card)
        simulation_layout.setContentsMargins(20, 18, 20, 20)
        simulation_layout.setSpacing(10)
        simulation_layout.addWidget(StrongBodyLabel("Configuração da simulação"))
        simulation_layout.addWidget(
            CaptionLabel("Escolha o comportamento neuronal e ajuste o estímulo.")
        )
        simulation_layout.addSpacing(4)
        simulation_layout.addWidget(CaptionLabel("Padrão neuronal"))

        self.combo_type = ComboBox()
        self.combo_type.addItems([
            f"{name} ({code})"
            for name, code in zip(CLASS_NAMES, CLASS_CODES, strict=True)
        ])
        self.combo_type.setToolTip("Padrão utilizado para gerar o traço simulado")
        self.combo_type.currentIndexChanged.connect(self.update_data)
        simulation_layout.addWidget(self.combo_type)

        current_header = QHBoxLayout()
        current_header.addWidget(CaptionLabel("Estímulo aplicado"))
        current_header.addStretch(1)
        self.label_current = StrongBodyLabel("10,0")
        current_header.addWidget(self.label_current)
        current_header.addWidget(CaptionLabel("unidades"))
        simulation_layout.addSpacing(6)
        simulation_layout.addLayout(current_header)

        self.slider_current = Slider(Qt.Orientation.Horizontal)
        self.slider_current.setRange(0, 300)
        self.slider_current.setSingleStep(1)
        self.slider_current.setPageStep(10)
        self.slider_current.setValue(100)
        self.slider_current.setToolTip("Intensidade de entrada aplicada ao neurônio")
        self.slider_current.valueChanged.connect(self.update_data)
        simulation_layout.addWidget(self.slider_current)

        self.label_domain = CaptionLabel(
            f"Domínio treinado: {self.config.current_min:.1f}–"
            f"{self.config.current_max:.1f} unidades"
        )
        self.label_domain.setWordWrap(True)
        simulation_layout.addWidget(self.label_domain)
        panel_layout.addWidget(simulation_card)

        result_card = SimpleCardWidget()
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(20, 18, 20, 20)
        result_layout.setSpacing(10)

        result_header = QHBoxLayout()
        result_header.addWidget(StrongBodyLabel("Leitura da inteligência artificial"))
        result_header.addStretch(1)
        self.result_badge = InfoBadge("AGUARDANDO", level=InfoLevel.INFOAMTION)
        self.result_badge.adjustSize()
        result_header.addWidget(self.result_badge)
        result_layout.addLayout(result_header)

        self.prediction_label = SubtitleLabel("—")
        self.prediction_label.setWordWrap(True)
        result_layout.addWidget(self.prediction_label)

        self.result_detail = CaptionLabel("A classificação aparecerá aqui.")
        self.result_detail.setWordWrap(True)
        result_layout.addWidget(self.result_detail)
        result_layout.addSpacing(4)

        confidence_header = QHBoxLayout()
        confidence_header.addWidget(CaptionLabel("Confiança da previsão"))
        confidence_header.addStretch(1)
        self.confidence_label = StrongBodyLabel("0,0%")
        confidence_header.addWidget(self.confidence_label)
        result_layout.addLayout(confidence_header)

        self.confidence_bar = ProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setFixedHeight(7)
        self.confidence_bar.setCustomBarColor("#009b75", "#3dd9a5")
        result_layout.addWidget(self.confidence_bar)
        result_layout.addSpacing(6)

        stats = QHBoxLayout()
        stats.setSpacing(24)
        expected_column, self.expected_value = self._build_stat("Esperado", "—")
        spikes_column, self.spikes_value = self._build_stat("Disparos", "0")
        stats.addLayout(expected_column)
        stats.addLayout(spikes_column)
        stats.addStretch(1)
        result_layout.addLayout(stats)

        panel_layout.addWidget(result_card, 1)
        return panel

    @staticmethod
    def _build_stat(title: str, initial_value: str) -> tuple[QVBoxLayout, BodyLabel]:
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.addWidget(CaptionLabel(title))
        value = BodyLabel(initial_value)
        layout.addWidget(value)
        return layout, value

    def _build_visualization_panel(self) -> QWidget:
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(14)

        summary = QHBoxLayout()
        summary.setSpacing(12)
        self.pattern_card = MetricCard("Padrão simulado", "RS")
        self.current_card = MetricCard("Estímulo", "10,0")
        self.spike_card = MetricCard("Disparos observados", "0")
        summary.addWidget(self.pattern_card, 1)
        summary.addWidget(self.current_card, 1)
        summary.addWidget(self.spike_card, 1)
        panel_layout.addLayout(summary)

        plot_card = SimpleCardWidget()
        plot_card.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        plot_layout = QVBoxLayout(plot_card)
        plot_layout.setContentsMargins(18, 16, 18, 18)
        plot_layout.setSpacing(8)

        plot_layout.addWidget(StrongBodyLabel("Potencial de membrana"))
        self.plot_subtitle = CaptionLabel("Traçado temporal do neurônio simulado")
        plot_layout.addWidget(self.plot_subtitle)

        pg.setConfigOptions(antialias=True)
        self.plot_graph = pg.PlotWidget()
        self.plot_graph.setMinimumHeight(340)
        self.plot_graph.setYRange(-100, 40, padding=0)
        self.plot_graph.setXRange(0, self.config.time_ms, padding=0)
        self.plot_graph.showGrid(x=True, y=True, alpha=0.18)
        self.plot_graph.getPlotItem().hideButtons()
        self.plot_graph.setMenuEnabled(False)
        self.plot_graph.setLabel("left", "Potencial de membrana", units="mV")
        self.plot_graph.setLabel("bottom", "Tempo", units="ms")
        self.curve = self.plot_graph.plot([], [], pen=pg.mkPen(width=2.2))
        self.threshold_line = pg.InfiniteLine(
            pos=30,
            angle=0,
            movable=False,
            pen=pg.mkPen(width=1, style=Qt.PenStyle.DashLine),
        )
        self.plot_graph.addItem(self.threshold_line)
        plot_layout.addWidget(self.plot_graph, 1)
        panel_layout.addWidget(plot_card, 1)
        return panel

    def on_theme_changed(self, use_light_theme: bool) -> None:
        self.is_light_theme = use_light_theme
        setTheme(Theme.LIGHT if use_light_theme else Theme.DARK)
        self.apply_visual_theme()

    def apply_visual_theme(self) -> None:
        palette = LIGHT_PALETTE if self.is_light_theme else DARK_PALETTE
        self.root.setStyleSheet(
            f"QWidget#appRoot {{ background-color: {palette['window']}; }}"
        )
        self.plot_graph.setBackground(palette["plot"])
        self.curve.setPen(pg.mkPen(palette["curve"], width=2.2))
        self.threshold_line.setPen(
            pg.mkPen(
                palette["threshold"], width=1, style=Qt.PenStyle.DashLine
            )
        )

        for axis_name in ("left", "bottom"):
            axis = self.plot_graph.getAxis(axis_name)
            axis.setPen(pg.mkPen(palette["muted"]))
            axis.setTextPen(pg.mkPen(palette["text"]))

        self.plot_graph.setLabel(
            "left", "Potencial de membrana", units="mV", color=palette["text"]
        )
        self.plot_graph.setLabel(
            "bottom", "Tempo", units="ms", color=palette["text"]
        )

    @staticmethod
    def _set_badge(badge: InfoBadge, text: str, level: InfoLevel) -> None:
        badge.setText(text)
        badge.setLevel(level)
        badge.adjustSize()

    def _set_confidence(self, confidence: float, state: str) -> None:
        confidence = min(max(float(confidence), 0.0), 1.0)
        self.confidence_label.setText(
            f"{confidence * 100:.1f}%".replace(".", ",")
        )
        self.confidence_bar.setValue(int(round(confidence * 100)))
        if state == "error":
            self.confidence_bar.setError(True)
        elif state == "warning":
            self.confidence_bar.resume()
            self.confidence_bar.pause()
        else:
            self.confidence_bar.resume()

    def update_data(self) -> None:
        class_index = self.combo_type.currentIndex()
        expected_code = CLASS_CODES[class_index]
        expected_name = CLASS_NAMES[class_index]
        current = self.slider_current.value() / 10.0

        current_display = f"{current:.1f}".replace(".", ",")
        self.label_current.setText(current_display)
        self.pattern_card.set_value(expected_code)
        self.current_card.set_value(current_display)
        self.expected_value.setText(f"{expected_name} ({expected_code})")

        params = IZHIKEVICH_PARAMETERS[expected_code]
        timeline, voltage = simulate_izhikevich_single(
            params["a"],
            params["b"],
            params["c"],
            params["d"],
            current,
            time_ms=self.config.time_ms,
            dt_ms=self.config.dt_ms,
        )
        spike_count = int(count_spikes(voltage)[0])
        self.spikes_value.setText(str(spike_count))
        self.spike_card.set_value(str(spike_count))
        self.plot_subtitle.setText(
            f"{expected_name} ({expected_code}) · estímulo {current_display} · "
            f"{spike_count} disparos"
        )
        self.curve.setData(timeline, voltage)
        self.plot_graph.setXRange(0, self.config.time_ms, padding=0)

        if self.model is None or self.checkpoint is None:
            self._set_badge(self.result_badge, "SEM MODELO", InfoLevel.WARNING)
            self.prediction_label.setText("Treinamento necessário")
            self.result_detail.setText(
                "O simulador está ativo, mas a classificação requer "
                "weights/classifier.pt. Execute train_pipeline.py."
            )
            if self.model_error:
                self.result_detail.setToolTip(self.model_error)
            self._set_confidence(0.0, "warning")
            return

        normalization = self.checkpoint["normalization"]
        result = classify_trace(
            voltage,
            self.model,
            normalization_mean=float(normalization["mean"]),
            normalization_std=float(normalization["std"]),
            class_codes=self.checkpoint["class_codes"],
            class_names=self.checkpoint["class_names"],
            minimum_spikes=self.config.minimum_spikes,
            confidence_threshold=float(self.checkpoint.get("confidence_threshold", 0.70)),
            device=self.device,
            current=current,
            trained_current_range=(self.config.current_min, self.config.current_max),
        )

        self.spikes_value.setText(str(result.spike_count))
        self.spike_card.set_value(str(result.spike_count))

        if not result.determined:
            self._set_badge(
                self.result_badge, "INDETERMINADO", InfoLevel.WARNING
            )
            hypothesis = (
                f"Hipótese: {result.class_name} ({result.class_code})"
                if result.class_name and result.class_code
                else "Sem hipótese confiável"
            )
            self.prediction_label.setText(hypothesis)
            self.result_detail.setText(f"Motivo: {result.reason}.")
            self._set_confidence(result.confidence, "warning")
            return

        correct = result.class_code == expected_code
        if correct:
            self._set_badge(self.result_badge, "CORRETO", InfoLevel.SUCCESS)
            state = "success"
            detail = f"A previsão coincide com o padrão esperado {expected_code}."
        else:
            self._set_badge(self.result_badge, "DIVERGENTE", InfoLevel.ERROR)
            state = "error"
            detail = f"A previsão diverge do padrão esperado {expected_code}."

        self.prediction_label.setText(
            f"{result.class_name} ({result.class_code})"
        )
        self.result_detail.setText(detail)
        self._set_confidence(result.confidence, state)


def main() -> int:
    configure_windows_process_identity()
    torch.set_num_threads(1)
    app = QApplication(sys.argv)
    app.setApplicationName("Classificador Izhikevich")
    icon = application_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)
    setTheme(Theme.DARK)
    setThemeColor("#00a884")
    window = IzhikevichApp()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
