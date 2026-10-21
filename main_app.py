"""Responsive Fluent desktop interface for simulated neuron populations."""

from __future__ import annotations

import argparse
import secrets
import sys
from functools import partial
from pathlib import Path

import pyqtgraph as pg
import torch
from PyQt6.QtCore import QEvent, QTimer, Qt
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLayout,
    QMainWindow,
    QSizePolicy,
    QTabWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    DoubleSpinBox,
    FluentIcon,
    InfoBadge,
    InfoLevel,
    ProgressBar,
    PushButton,
    ScrollArea,
    SimpleCardWidget,
    Slider,
    SpinBox,
    StrongBodyLabel,
    SubtitleLabel,
    SwitchButton,
    TableView,
    TableWidget,
    Theme,
    TitleLabel,
    ToolButton,
    setTheme,
    setThemeColor,
)

from config import CLASS_CODES, CLASS_NAMES, SimulationConfig
from inference import default_cpu_threads
from models import NeuronClassifier1D
from neural_background import NeuralBackground
from network import MAX_OUT_DEGREE, MAX_SYNAPSES, NetworkConfig
from population import MAX_NEURONS_PER_CLASS, PopulationRequest
from population_table import PopulationTableModel
from population_worker import PopulationResult, PopulationRunner, process_population


DARK_PALETTE = {
    "window": "#0f1115", "plot": "#15191f", "text": "#d9e2ec",
    "muted": "#94a3b8", "curve": "#00e5c0", "border": "#303944",
}
LIGHT_PALETTE = {
    "window": "#f3f6f9", "plot": "#ffffff", "text": "#344054",
    "muted": "#667085", "curve": "#008f7a", "border": "#d5dde5",
}
CLASS_COLORS = {"RS": "#16a779", "CH": "#e49329", "FS": "#428bff", None: "#94a3b8"}
WINDOWS_APP_ID = "Izhikevich.NeuralPatternClassifier"


def resource_root() -> Path:
    bundled = getattr(sys, "_MEIPASS", None)
    return Path(bundled) if bundled else Path(__file__).resolve().parent


def application_icon() -> QIcon:
    icon_name = "app_icon.ico" if sys.platform == "win32" else "app_icon.png"
    icon_path = resource_root() / "assets" / icon_name
    return QIcon(str(icon_path)) if icon_path.is_file() else QIcon()


def configure_windows_process_identity() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
    except (AttributeError, OSError):
        pass


def load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    required = {"model_state", "normalization", "simulation", "class_codes", "class_names"}
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(f"Checkpoint missing fields: {sorted(missing)}")
    return checkpoint


def caption(text: str, parent=None) -> CaptionLabel:
    label = CaptionLabel(text, parent)
    label.setWordWrap(True)
    return label


class MetricCard(SimpleCardWidget):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(120)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(2)
        self.title_label = caption(title)
        layout.addWidget(self.title_label)
        self.value_label = SubtitleLabel("—")
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class IzhikevichApp(QMainWindow):
    def __init__(self, checkpoint_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Dinâmica Neuronal — Classificador Izhikevich")
        self.setWindowIcon(application_icon())
        self.resize(1280, 880)
        self.setMinimumSize(1040, 760)
        self.device = torch.device("cpu")
        self.model: NeuronClassifier1D | None = None
        self.checkpoint: dict[str, object] | None = None
        self.config = SimulationConfig()
        self.model_error: str | None = None
        self.result: PopulationResult | None = None
        self.is_light_theme = False
        self._previous_window_state = Qt.WindowState.WindowNoState
        self._closing = False
        self.load_ai_model(checkpoint_path or resource_root() / "weights" / "classifier.pt")
        self.setup_ui()
        self.fullscreen_shortcut = QShortcut(QKeySequence("F11"), self)
        self.fullscreen_shortcut.setAutoRepeat(False)
        self.fullscreen_shortcut.activated.connect(self.toggle_fullscreen)
        self.escape_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.escape_shortcut.setEnabled(False)
        self.escape_shortcut.activated.connect(self.exit_fullscreen)
        self.apply_visual_theme()

        processor = partial(
            process_population, config=self.config, model=self.model,
            checkpoint=self.checkpoint, device=self.device,
        )
        self.runner = PopulationRunner(processor, self)
        self.runner.ready.connect(self._show_result)
        self.runner.failed.connect(self._show_error)
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(60)
        self.update_timer.timeout.connect(self._submit_request)
        for spin in self.count_spins:
            spin.valueChanged.connect(self._schedule_update)
        self.seed_spin.valueChanged.connect(self._schedule_update)
        self.variation_check.stateChanged.connect(self._schedule_update)
        self.network_check.stateChanged.connect(self._network_changed)
        self.network_preset_button.clicked.connect(self._network_preset)
        for spin in self.network_spins:
            spin.valueChanged.connect(self._schedule_update)
        self.slider_current.valueChanged.connect(self._schedule_update)
        self.slider_current.sliderReleased.connect(self._submit_request)
        self.neuron_selector.valueChanged.connect(self._show_selected_neuron)
        self.table.clicked.connect(self._select_table_neuron)
        self.new_seed_button.clicked.connect(self._new_seed)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)
        self._schedule_update()

    def load_ai_model(self, path: Path) -> None:
        try:
            checkpoint = load_checkpoint(path, self.device)
            if (
                len(checkpoint["class_codes"]) != len(CLASS_CODES)
                or set(checkpoint["class_codes"]) != set(CLASS_CODES)
                or len(checkpoint["class_names"]) != len(CLASS_CODES)
            ):
                raise RuntimeError("O checkpoint deve conter as classes RS, CH e FS.")
            model = NeuronClassifier1D(num_classes=len(checkpoint["class_names"]))
            model.load_state_dict(checkpoint["model_state"])
            model.to(self.device).eval()
            simulation = checkpoint["simulation"]
            self.config = SimulationConfig(**{
                name: simulation[name] for name in SimulationConfig.__dataclass_fields__
                if name in simulation
            })
            self.model = model.for_inference()
            self.checkpoint = checkpoint
        except Exception as exc:
            self.model_error = str(exc)

    def setup_ui(self) -> None:
        self.root = NeuralBackground(resource_root() / "assets" / "neural_background.png")
        self.root.setObjectName("appRoot")
        self.setCentralWidget(self.root)
        layout = QVBoxLayout(self.root)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        layout.addLayout(self._build_header())
        self.status_label = caption("Preparando a população…")
        self.status_label.setMinimumHeight(30)
        layout.addWidget(self.status_label)
        content = QHBoxLayout()
        content.setSpacing(16)
        content.addWidget(self._build_controls())
        content.addWidget(self._build_results(), 1)
        layout.addLayout(content, 1)

    def _build_header(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(TitleLabel("Dinâmica neuronal simulada"))
        titles.addWidget(caption("Populações independentes ou conectadas · padrões RS, CH e FS"))
        layout.addLayout(titles, 1)
        self.model_badge = InfoBadge(
            "MODELO PRONTO" if self.model is not None else "SEM MODELO",
            level=InfoLevel.SUCCESS if self.model is not None else InfoLevel.WARNING,
        )
        if self.model_error:
            self.model_badge.setToolTip(self.model_error)
        layout.addWidget(self.model_badge)
        self.processing_badge = InfoBadge("AGUARDANDO", level=InfoLevel.INFOAMTION)
        layout.addWidget(self.processing_badge)
        self.theme_switch = SwitchButton()
        self.theme_switch.setOnText("Tema claro")
        self.theme_switch.setOffText("Tema claro")
        self.theme_switch.checkedChanged.connect(self.on_theme_changed)
        layout.addWidget(self.theme_switch)
        self.background_button = ToolButton(FluentIcon.PHOTO)
        self.background_button.setFixedSize(36, 36)
        self.background_button.setCheckable(True)
        self.background_button.setChecked(True)
        self.background_button.setToolTip("Fundo neural: ativar / desativar")
        self.background_button.setAccessibleName("Fundo neural")
        self.background_button.toggled.connect(self.root.set_artwork_enabled)
        layout.addWidget(self.background_button)
        self.fullscreen_button = ToolButton(FluentIcon.FULL_SCREEN)
        self.fullscreen_button.setFixedSize(36, 36)
        self.fullscreen_button.setToolTip("Tela cheia (F11)")
        self.fullscreen_button.setAccessibleName("Tela cheia")
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        layout.addWidget(self.fullscreen_button)
        return layout

    def _build_controls(self) -> ScrollArea:
        scroll = ScrollArea()
        scroll.setFixedWidth(330)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        panel = QWidget()
        panel.setObjectName("controlPanel")
        scroll.setWidget(panel)
        scroll.viewport().setObjectName("controlViewport")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(12)

        card = SimpleCardWidget()
        controls = QVBoxLayout(card)
        controls.setContentsMargins(18, 16, 18, 18)
        controls.setSpacing(10)
        controls.addWidget(StrongBodyLabel("Composição da população"))
        controls.addWidget(caption("Quantidade de neurônios por padrão"))
        self.count_spins = []
        for code, name in zip(CLASS_CODES, CLASS_NAMES, strict=True):
            row = QHBoxLayout()
            labels = QVBoxLayout()
            labels.setSpacing(0)
            labels.addWidget(StrongBodyLabel(code))
            labels.addWidget(CaptionLabel(name))
            row.addLayout(labels, 1)
            spin = SpinBox()
            spin.setRange(0, MAX_NEURONS_PER_CLASS)
            spin.setValue(10)
            spin.setKeyboardTracking(False)
            spin.setFixedWidth(168)
            row.addWidget(spin)
            self.count_spins.append(spin)
            controls.addLayout(row)
        controls.addWidget(caption(f"Até {MAX_NEURONS_PER_CLASS} por classe. Conexões opcionais no controle de rede."))
        controls.addSpacing(6)
        row = QHBoxLayout()
        row.addWidget(StrongBodyLabel("Estímulo aplicado"), 1)
        self.label_current = StrongBodyLabel("10,0")
        row.addWidget(self.label_current)
        controls.addLayout(row)
        self.slider_current = Slider(Qt.Orientation.Horizontal)
        self.slider_current.setRange(0, round(max(30.0, self.config.current_max) * 10))
        initial_current = max(self.config.current_min, min(10.0, self.config.current_max))
        self.slider_current.setValue(round(initial_current * 10))
        self.slider_current.setToolTip("Mesmo estímulo para todos os neurônios, em unidades do modelo")
        controls.addWidget(self.slider_current)
        controls.addWidget(caption(
            f"Domínio treinado: {self.config.current_min:g}–{self.config.current_max:g} unidades"
        ))
        self.variation_check = CheckBox("Variação entre neurônios")
        self.variation_check.setChecked(True)
        self.variation_check.setToolTip(
            "Aplica a variação paramétrica e o ruído de observação do treinamento. "
            "Desmarque para inspecionar as parametrizações centrais."
        )
        controls.addWidget(self.variation_check)
        controls.addWidget(CaptionLabel("Semente da população"))
        self.seed_spin = SpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(42)
        self.seed_spin.setKeyboardTracking(False)
        controls.addWidget(self.seed_spin)
        self.new_seed_button = PushButton("Nova variação")
        controls.addWidget(self.new_seed_button)
        layout.addWidget(card)
        layout.addWidget(self._build_network_controls())
        explanation = SimpleCardWidget()
        explanation_layout = QVBoxLayout(explanation)
        explanation_layout.setContentsMargins(18, 16, 18, 16)
        explanation_layout.addWidget(StrongBodyLabel("Como inspecionar"))
        explanation_layout.addWidget(caption(
            "O raster mostra os instantes de disparo. Escolha um neurônio pelo número "
            "ou clique em sua linha na tabela para examinar o sinal completo."
        ))
        explanation_layout.addWidget(caption(
            "As estatísticas usam toda a população, inclusive os neurônios que não "
            "aparecem na amostra visual do raster."
        ))
        if self.model is None:
            explanation_layout.addWidget(caption(
                "Simulação disponível. Para classificar, copie o checkpoint treinado "
                "para weights/classifier.pt ou execute train_pipeline.py."
            ))
        layout.addWidget(explanation)
        layout.addStretch(1)
        return scroll

    def _build_network_controls(self) -> SimpleCardWidget:
        card = SimpleCardWidget()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.addWidget(StrongBodyLabel("Conexões sinápticas"))
        self.network_check = CheckBox("Conectar neurônios")
        layout.addWidget(self.network_check)
        self.network_preset_button = PushButton("Preset: 10.000 neurônios")
        layout.addWidget(self.network_preset_button)
        self.network_controls = QWidget()
        fields = QVBoxLayout(self.network_controls)
        fields.setContentsMargins(0, 0, 0, 0)
        self.out_degree_spin = SpinBox()
        self.out_degree_spin.setRange(0, MAX_OUT_DEGREE)
        self.out_degree_spin.setValue(100)
        self.out_degree_spin.setKeyboardTracking(False)
        definitions = [
            ("Saídas por neurônio", self.out_degree_spin),
        ]
        for name, text, low, high, value, step in (
            ("weight_spin", "Força excitatória", 0.0, 5.0, 0.5, 0.1),
            ("inhibitory_spin", "Inibitórios (%)", 0.0, 100.0, 20.0, 5.0),
            ("inhibitory_gain_spin", "Ganho inibitório", 0.0, 10.0, 4.0, 0.5),
            ("delay_spin", "Atraso (ms)", 0.1, 20.0, 1.0, 0.1),
            ("tau_spin", "Decaimento (ms)", 0.5, 50.0, 5.0, 0.5),
        ):
            spin = DoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(2)
            spin.setSingleStep(step)
            spin.setValue(value)
            spin.setKeyboardTracking(False)
            setattr(self, name, spin)
            definitions.append((text, spin))
        self.network_spins = []
        for text, spin in definitions:
            row = QHBoxLayout()
            row.addWidget(CaptionLabel(text), 1)
            spin.setFixedWidth(132)
            row.addWidget(spin)
            fields.addLayout(row)
            self.network_spins.append(spin)
        self.network_controls.setEnabled(False)
        layout.addWidget(self.network_controls)
        layout.addWidget(caption(
            "Sinapses de corrente com atraso. Papéis excitatório/inibitório são "
            "sorteados separadamente dos parâmetros RS/CH/FS."
        ))
        layout.addWidget(caption(
            f"Até {MAX_SYNAPSES:,} sinapses, sem autoconexões nem destinos repetidos. "
            "Em populações pequenas, as saídas são limitadas a N − 1."
        ))
        layout.addWidget(caption(
            "Na rede, as previsões da IA são experimentais: o modelo foi treinado "
            "com neurônios independentes. Os pesos não são retreinados."
        ))
        return card

    def _network_changed(self, *_args) -> None:
        self.network_controls.setEnabled(self.network_check.isChecked())
        self._schedule_update()

    def _network_preset(self) -> None:
        for spin, value in zip(self.count_spins, (3334, 3333, 3333), strict=True):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        for spin, value in zip(self.network_spins, (100, 0.5, 20, 4, 1, 5), strict=True):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.network_check.setChecked(True)
        self._schedule_update()

    def _build_results(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        summaries = QHBoxLayout()
        summaries.setSpacing(10)
        self.total_card = MetricCard("Neurônios")
        self.classified_card = MetricCard("Classificados")
        self.accuracy_card = MetricCard("Acerto dos classificados")
        self.accuracy_card.setToolTip("Acertos / neurônios classificados. Indeterminados são contabilizados na cobertura.")
        self.spike_card = MetricCard("Disparos totais")
        for card in (self.total_card, self.classified_card, self.accuracy_card, self.spike_card):
            summaries.addWidget(card, 1)
        layout.addLayout(summaries)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_signal_tab(), "Sinais")
        self.tabs.addTab(self._build_table_tab(), "Classificações individuais")
        self.tabs.addTab(self._build_statistics_tab(), "Estatísticas")
        self.tabs.addTab(self._build_network_tab(), "Rede")
        layout.addWidget(self.tabs, 1)
        return panel

    def _build_signal_tab(self) -> QWidget:
        page = ScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QFrame.Shape.NoFrame)
        panel = QWidget()
        panel.setObjectName("signalsPanel")
        page.viewport().setObjectName("signalsViewport")
        page.setWidget(panel)
        layout = QVBoxLayout(panel)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        layout.setContentsMargins(10, 10, 10, 10)
        raster_card = SimpleCardWidget()
        raster_layout = QVBoxLayout(raster_card)
        raster_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        raster_layout.addWidget(StrongBodyLabel("Atividade da população"))
        self.raster_subtitle = caption("Preparando o raster…")
        self.raster_subtitle.setMinimumHeight(32)
        raster_layout.addWidget(self.raster_subtitle)
        self.raster_graph = pg.PlotWidget()
        self.raster_graph.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.raster_graph.setMinimumHeight(160)
        self.raster_graph.setLabel("left", "Neurônio")
        self.raster_graph.setLabel("bottom", "Tempo", units="ms")
        self.raster_graph.setXRange(0, self.config.time_ms, padding=0)
        self.raster_graph.setMouseEnabled(x=True, y=True)
        self.raster_graph.setMenuEnabled(False)
        self.raster_graph.getPlotItem().hideButtons()
        self.raster_curves = {}
        for code in (*CLASS_CODES, None):
            self.raster_curves[code] = self.raster_graph.plot(
                [], [], pen=None, symbol="|", symbolSize=6,
                symbolPen=pg.mkPen(CLASS_COLORS[code], width=1),
                symbolBrush=None, antialias=False,
            )
        raster_layout.addWidget(self.raster_graph, 1)
        layout.addWidget(raster_card, 1)

        trace_card = SimpleCardWidget()
        trace_layout = QVBoxLayout(trace_card)
        trace_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        row = QHBoxLayout()
        row.addWidget(StrongBodyLabel("Sinal do neurônio"), 1)
        self.neuron_selector = SpinBox()
        self.neuron_selector.setRange(1, 1)
        self.neuron_selector.setValue(1)
        self.neuron_selector.setEnabled(False)
        self.neuron_selector.setKeyboardTracking(False)
        self.neuron_selector.setFixedWidth(168)
        row.addWidget(self.neuron_selector)
        trace_layout.addLayout(row)
        self.trace_subtitle = caption("Selecione uma população para inspecionar.")
        self.trace_subtitle.setMinimumHeight(32)
        trace_layout.addWidget(self.trace_subtitle)
        self.plot_graph = pg.PlotWidget()
        self.plot_graph.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.plot_graph.setMinimumHeight(160)
        self.plot_graph.setYRange(-110, 40, padding=0)
        self.plot_graph.setXRange(0, self.config.time_ms, padding=0)
        self.plot_graph.setLabel("left", "Potencial", units="mV")
        self.plot_graph.setLabel("bottom", "Tempo", units="ms")
        self.plot_graph.showGrid(x=True, y=True, alpha=0.15)
        self.plot_graph.setMenuEnabled(False)
        self.plot_graph.getPlotItem().hideButtons()
        self.curve = self.plot_graph.plot([], [], pen=pg.mkPen(width=1), antialias=False)
        self.curve.setDownsampling(auto=True, method="peak")
        self.curve.setClipToView(True)
        self.threshold_line = pg.InfiniteLine(
            pos=30, angle=0, movable=False,
            pen=pg.mkPen(width=1, style=Qt.PenStyle.DashLine),
        )
        self.plot_graph.addItem(self.threshold_line)
        trace_layout.addWidget(self.plot_graph, 1)
        result_row = QHBoxLayout()
        self.selected_prediction = BodyLabel("—")
        result_row.addWidget(self.selected_prediction, 1)
        self.selected_badge = InfoBadge("AGUARDANDO", level=InfoLevel.INFOAMTION)
        result_row.addWidget(self.selected_badge)
        trace_layout.addLayout(result_row)
        self.selected_detail = caption("A classificação aparecerá aqui.")
        self.selected_detail.setMinimumHeight(32)
        trace_layout.addWidget(self.selected_detail)
        self.confidence_bar = ProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setFixedHeight(5)
        trace_layout.addWidget(self.confidence_bar)
        layout.addWidget(trace_card, 2)
        return page

    def _build_network_tab(self) -> QWidget:
        page = ScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QFrame.Shape.NoFrame)
        panel = QWidget()
        panel.setObjectName("networkPanel")
        page.viewport().setObjectName("networkViewport")
        page.setWidget(panel)
        layout = QVBoxLayout(panel)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.addWidget(StrongBodyLabel("Dinâmica da rede"))
        self.network_detail = caption("Modo independente: ative as conexões para inspecionar a rede.")
        layout.addWidget(self.network_detail)
        layout.addWidget(caption("Taxa de disparo de toda a população · janelas de até 5 ms"))
        self.rate_graph = pg.PlotWidget()
        self.rate_graph.setFixedHeight(190)
        self.rate_graph.setLabel("left", "Taxa média", units="Hz")
        self.rate_graph.setMenuEnabled(False)
        self.rate_curve = self.rate_graph.plot([], [], pen=pg.mkPen("#e49329", width=2))
        layout.addWidget(self.rate_graph)
        layout.addWidget(caption("Corrente sináptica média · unidades do modelo"))
        self.synaptic_graph = pg.PlotWidget()
        self.synaptic_graph.setFixedHeight(150)
        self.synaptic_graph.setLabel("left", "Corrente média")
        self.synaptic_graph.setMenuEnabled(False)
        self.synaptic_curve = self.synaptic_graph.plot([], [], pen=pg.mkPen("#428bff", width=1))
        self.synaptic_curve.setDownsampling(auto=True, method="peak")
        self.synaptic_curve.setClipToView(True)
        layout.addWidget(self.synaptic_graph)
        layout.addWidget(StrongBodyLabel("Conexões: origem → destino"))
        layout.addWidget(caption("Contagens por parâmetros intrínsecos, calculadas sobre todas as sinapses."))
        self.connectivity_table = self._statistics_table(3, list(CLASS_CODES), list(CLASS_CODES))
        layout.addWidget(self.connectivity_table)
        self.connection_detail = caption("Selecione um neurônio na aba Sinais para inspecionar suas conexões.")
        layout.addWidget(self.connection_detail)
        layout.addWidget(caption(
            "Rede exploratória de sinapses de corrente. Não modela junções elétricas, "
            "plasticidade nem aquisição de sinais biológicos."
        ))
        return page

    def _build_table_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(caption(
            "Uma linha por neurônio. Clique para abrir seu sinal. (?) indica uma hipótese "
            "abaixo do limiar de confiança; ela não entra na contagem de classificados."
        ))
        self.table_model = PopulationTableModel(self)
        self.table = TableView()
        self.table.setModel(self.table_model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)
        return page

    @staticmethod
    def _statistics_table(rows: int, columns: list[str], row_labels: list[str]) -> TableWidget:
        table = TableWidget()
        table.setRowCount(rows)
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setVerticalHeaderLabels(row_labels)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setDefaultSectionSize(38)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setMinimumHeight(160)
        return table

    def _build_statistics_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        layout.addWidget(StrongBodyLabel("Resultados desta simulação"))
        layout.addWidget(caption(
            "Estas contagens descrevem a população exibida. Elas não substituem "
            "as métricas científicas do conjunto de teste retido."
        ))
        self.statistics_detail = caption("Aguardando resultados.")
        layout.addWidget(self.statistics_detail)
        self.counts_table = self._statistics_table(
            3, ["Esperados", "Previstos (classificados)"], list(CLASS_CODES),
        )
        layout.addWidget(self.counts_table)
        self.confusion_heading = StrongBodyLabel("Esperado × previsto")
        layout.addWidget(self.confusion_heading)
        self.confusion_table = self._statistics_table(
            3, [*CLASS_CODES, "Indeterminado"], [f"{code} esperado" for code in CLASS_CODES],
        )
        layout.addWidget(self.confusion_table)
        self.performance_detail = caption("—")
        layout.addWidget(self.performance_detail)
        layout.addStretch(1)
        return page

    @staticmethod
    def _set_badge(badge: InfoBadge, text: str, level: InfoLevel) -> None:
        badge.setText(text)
        badge.setLevel(level)
        badge.adjustSize()

    def _schedule_update(self, *_args) -> None:
        if self._closing:
            return
        current = self.slider_current.value() / 10.0
        self.label_current.setText(f"{current:.1f}".replace(".", ","))
        self.runner.invalidate()
        self._set_badge(self.processing_badge, "ATUALIZANDO", InfoLevel.WARNING)
        self.status_label.setText(
            "Atualizando a população… Gráficos e tabelas mostram a última simulação concluída."
        )
        # Coalesce bursts without accumulating a simulation for each slider tick.
        if not self.update_timer.isActive():
            self.update_timer.start()

    def _submit_request(self) -> None:
        if self._closing:
            return
        self.update_timer.stop()
        try:
            request = PopulationRequest(
                counts=tuple(spin.value() for spin in self.count_spins),
                current=self.slider_current.value() / 10.0,
                seed=self.seed_spin.value(),
                varied=self.variation_check.isChecked(),
                network=NetworkConfig(
                    out_degree=self.out_degree_spin.value(),
                    weight=self.weight_spin.value(),
                    inhibitory_fraction=self.inhibitory_spin.value() / 100.0,
                    inhibitory_gain=self.inhibitory_gain_spin.value(),
                    delay_ms=self.delay_spin.value(),
                    tau_ms=self.tau_spin.value(),
                ) if self.network_check.isChecked() else None,
            )
        except ValueError as exc:
            self.runner.invalidate()
            self._clear_result()
            self._show_error(str(exc))
            return
        self.runner.submit(request)

    def _new_seed(self) -> None:
        self.seed_spin.setValue((self.seed_spin.value() + 1 + secrets.randbelow(2**31 - 1)) % 2**31)

    def _clear_result(self) -> None:
        self.result = None
        self.table_model.set_result(None)
        self.neuron_selector.setEnabled(False)
        self.curve.setData([], [])
        for curve in self.raster_curves.values():
            curve.setData([], [])
        for card in (self.total_card, self.classified_card, self.accuracy_card, self.spike_card):
            card.set_value("—")
        self.counts_table.clearContents()
        self.confusion_table.clearContents()
        self._show_network(None)
        self.statistics_detail.setText("Sem população válida.")
        self.performance_detail.setText("—")
        self.raster_subtitle.setText("Sem população válida.")
        self.trace_subtitle.setText("Selecione pelo menos um neurônio.")
        self.selected_prediction.setText("—")
        self.selected_detail.setText("Sem classificação.")
        self.confidence_bar.setValue(0)
        self._set_badge(self.selected_badge, "AGUARDANDO", InfoLevel.INFOAMTION)

    def _show_error(self, message: str) -> None:
        self._set_badge(self.processing_badge, "VERIFICAR", InfoLevel.ERROR)
        suffix = " Exibindo a última simulação concluída." if self.result is not None else ""
        self.status_label.setText(f"{message}{suffix}")

    def _show_result(self, result: PopulationResult) -> None:
        if self._closing:
            return
        self.result = result
        population, stats = result.population, result.statistics
        self.table_model.set_result(result)
        self.neuron_selector.blockSignals(True)
        self.neuron_selector.setRange(1, population.size)
        self.neuron_selector.setEnabled(True)
        self.neuron_selector.blockSignals(False)
        self.total_card.set_value(str(stats.total))
        self.classified_card.set_value(str(stats.classified) if result.predictions is not None else "—")
        self.accuracy_card.set_value(f"{stats.accuracy:.1%}" if stats.accuracy is not None else "—")
        self.spike_card.set_value(str(stats.total_spikes))
        connected = population.network_activity is not None
        self.accuracy_card.title_label.setText(
            "Concordância experimental" if connected else "Acerto dos classificados"
        )
        for series in result.raster.series:
            self.raster_curves[series.class_code].setData(series.times, series.neuron_ids)
        self.raster_graph.setYRange(0, population.size + 1, padding=0)
        if result.raster.neuron_count < population.size:
            preview_text = f"Amostra visual: {result.raster.neuron_count} de {population.size} neurônios."
        else:
            preview_text = f"Todos os {population.size} neurônios."
        legend = " · ".join(
            f'<span style="color:{CLASS_COLORS[code]}">{code}</span>' for code in CLASS_CODES
        )
        self.raster_subtitle.setText(
            f"{preview_text} Cores por previsão: {legend}; cinza = indeterminado / sem modelo."
        )
        self._show_statistics(result)
        self._show_network(result)
        self._show_selected_neuron()
        self._set_badge(self.processing_badge, "PRONTO", InfoLevel.SUCCESS)
        self.status_label.setText(
            f"{'Rede conectada' if connected else 'População independente'} concluída · "
            f"{population.size} neurônios · estímulo "
            f"{population.request.current:g} · semente {population.request.seed}."
            + (" Classificação experimental; modelo treinado sem sinapses." if connected else "")
        )

    def _show_network(self, result: PopulationResult | None) -> None:
        activity = result.population.network_activity if result is not None else None
        if activity is None:
            self.rate_curve.setData([], [])
            self.synaptic_curve.setData([], [])
            self.connectivity_table.clearContents()
            self.network_detail.setText("Modo independente: ative as conexões para inspecionar a rede.")
            self.connection_detail.setText("Sem conexões nesta população.")
            return
        network = activity.network
        inhibited = int(network.inhibitory.sum())
        self.network_detail.setText(
            f"{network.size:,} neurônios · {network.synapse_count:,} sinapses dirigidas · "
            f"{network.out_degree} saídas por neurônio\n"
            f"Excitatórios: {network.size - inhibited:,} · Inibitórios: {inhibited:,} · "
            f"Atraso efetivo: {activity.effective_delay_ms:g} ms · "
            f"Eventos sinápticos entregues: {activity.delivered_events:,}"
        )
        self.rate_curve.setData(activity.rate_times_ms, activity.rate_hz)
        self.synaptic_curve.setData(result.population.timeline, activity.mean_synaptic_current)
        for row in range(3):
            for column in range(3):
                self._set_cell(self.connectivity_table, row, column, int(network.class_connections[row, column]))

    def _show_statistics(self, result: PopulationResult) -> None:
        stats = result.statistics
        connected = result.population.network_activity is not None
        self.counts_table.setHorizontalHeaderLabels([
            "Parâmetros" if connected else "Esperados", "Previstos (classificados)",
        ])
        self.confusion_heading.setText("Parâmetros × previsão experimental" if connected else "Esperado × previsto")
        self.confusion_table.setVerticalHeaderLabels([
            f"{code} {'parâmetros' if connected else 'esperado'}" for code in CLASS_CODES
        ])
        self.accuracy_card.setToolTip(
            "Concordância com parâmetros de origem. Modelo sem validação em redes conectadas."
            if connected else "Acertos / neurônios classificados. Indeterminados são contabilizados na cobertura."
        )
        for row, code in enumerate(CLASS_CODES):
            self._set_cell(self.counts_table, row, 0, stats.expected_counts[code])
            predicted = stats.predicted_counts[code] if result.predictions is not None else "—"
            self._set_cell(self.counts_table, row, 1, predicted)
            for column in range(4):
                value = int(stats.confusion_matrix[row, column]) if result.predictions is not None else "—"
                self._set_cell(self.confusion_table, row, column, value)
        if result.predictions is None:
            self.statistics_detail.setText(f"{stats.unclassified} neurônios sem classificação: checkpoint não carregado.")
        else:
            confidence = f"{stats.mean_confidence:.1%}" if stats.mean_confidence is not None else "—"
            self.statistics_detail.setText(
                f"Cobertura: {stats.classified}/{stats.total} ({stats.coverage:.1%}) · "
                f"{'Concordantes' if connected else 'Acertos'}: {stats.correct}/{stats.classified} classificados · "
                f"Indeterminados: {stats.indeterminate}\n"
                f"Confiança média dos classificados: {confidence} · "
                f"Taxa média de disparo: {stats.mean_rate_hz:.1f} Hz"
            )
        if connected:
            self.statistics_detail.setText(
                self.statistics_detail.text() + "\nExploratório: concordância com os parâmetros RS/CH/FS, "
                "sem validação do classificador em redes conectadas."
            )
        self.performance_detail.setText(
            f"Processamento: {result.elapsed_ms:.0f} ms para toda a população "
            "(simulação, inferência e preparação dos resultados; exclui desenho)."
        )

    @staticmethod
    def _set_cell(table: TableWidget, row: int, column: int, value) -> None:
        item = QTableWidgetItem(str(value))
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        table.setItem(row, column, item)

    def _show_selected_neuron(self, *_args) -> None:
        if self.result is None:
            return
        population = self.result.population
        row = self.neuron_selector.value() - 1
        code = CLASS_CODES[int(population.labels[row])]
        name = CLASS_NAMES[int(population.labels[row])]
        self.curve.setData(population.timeline, population.traces[row])
        self.trace_subtitle.setText(
            f"Neurônio {row + 1} · parâmetros: {name} ({code}) · "
            f"{int(population.spike_counts[row])} disparos · "
            f"estímulo {population.request.current:g}"
        )
        prediction = self.result.predictions[row] if self.result.predictions is not None else None
        if prediction is None:
            self.selected_prediction.setText("Classificação indisponível")
            self.selected_detail.setText("O simulador está ativo. O checkpoint treinado não foi carregado.")
            self.confidence_bar.setValue(0)
            self._set_badge(self.selected_badge, "SEM MODELO", InfoLevel.WARNING)
        elif not prediction.determined:
            hypothesis = f"Hipótese: {prediction.class_name}" if prediction.class_name else "Sem hipótese confiável"
            self.selected_prediction.setText(hypothesis)
            self.selected_detail.setText(
                f"{prediction.reason}. Confiança: {prediction.confidence:.1%}."
            )
            self.confidence_bar.setValue(round(prediction.confidence * 100))
            self._set_badge(self.selected_badge, "INDETERMINADO", InfoLevel.WARNING)
        else:
            correct = prediction.class_code == code
            self.selected_prediction.setText(f"Previsto: {prediction.class_name} ({prediction.class_code})")
            self.selected_detail.setText(f"Confiança da previsão: {prediction.confidence:.1%}")
            self.confidence_bar.setValue(round(prediction.confidence * 100))
            self._set_badge(
                self.selected_badge, "CORRETO" if correct else "DIVERGENTE",
                InfoLevel.SUCCESS if correct else InfoLevel.ERROR,
            )
        activity = population.network_activity
        if activity is not None:
            network = activity.network
            role = "inibitório" if network.inhibitory[row] else "excitatório"
            destinations = ", ".join(str(int(target) + 1) for target in network.targets[row, :16]) or "nenhum"
            suffix = " (primeiros 16)" if network.out_degree > 16 else ""
            self.connection_detail.setText(
                f"Neurônio {row + 1}: {role} · peso por saída {network.source_weights[row]:g}\n"
                f"Entradas: {network.in_degrees[row]} · Saídas: {network.out_degree}\n"
                f"Destinos{suffix}: {destinations}"
            )
            if prediction is not None:
                self.selected_detail.setText(self.selected_detail.text() + " · Previsão experimental em rede.")
                if prediction.determined:
                    self._set_badge(
                        self.selected_badge, "CONCORDANTE" if prediction.class_code == code else "DIVERGENTE",
                        InfoLevel.INFOAMTION,
                    )

    def _select_table_neuron(self, index) -> None:
        if index.isValid():
            self.neuron_selector.setValue(index.row() + 1)
            self.tabs.setCurrentIndex(0)

    def on_theme_changed(self, use_light_theme: bool) -> None:
        self.is_light_theme = use_light_theme
        setTheme(Theme.LIGHT if use_light_theme else Theme.DARK)
        self.apply_visual_theme()

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.exit_fullscreen()
        else:
            self._previous_window_state = self.windowState()
            self.showFullScreen()

    def exit_fullscreen(self) -> None:
        if not self.isFullScreen():
            return
        if self._previous_window_state & Qt.WindowState.WindowMaximized:
            self.showMaximized()
        else:
            self.showNormal()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange and hasattr(self, "fullscreen_button"):
            fullscreen = self.isFullScreen()
            self.fullscreen_button.setIcon(FluentIcon.CONSTRACT if fullscreen else FluentIcon.FULL_SCREEN)
            self.fullscreen_button.setToolTip("Sair da tela cheia (Esc / F11)" if fullscreen else "Tela cheia (F11)")
            self.fullscreen_button.setAccessibleName("Sair da tela cheia" if fullscreen else "Tela cheia")
            if hasattr(self, "escape_shortcut"):
                self.escape_shortcut.setEnabled(fullscreen)

    def apply_visual_theme(self) -> None:
        palette = LIGHT_PALETTE if self.is_light_theme else DARK_PALETTE
        self.root.set_light_theme(self.is_light_theme)
        self.root.setStyleSheet(
            "QWidget#controlPanel, QWidget#controlViewport, "
            "QWidget#signalsPanel, QWidget#signalsViewport, QWidget#networkPanel, QWidget#networkViewport "
            "{ background-color: transparent; }"
        )
        # QScrollArea enables automatic background filling on its child when
        # setWidget() is called. Disable it so transparent gaps show the art
        # instead of the platform's default (often white) window brush.
        for scroll in self.root.findChildren(ScrollArea):
            scroll.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")
            scroll.viewport().setAutoFillBackground(False)
            if scroll.widget() is not None:
                scroll.widget().setAutoFillBackground(False)
        card_color = "rgba(255, 255, 255, 242)" if self.is_light_theme else "rgba(21, 25, 31, 242)"
        for card in self.root.findChildren(SimpleCardWidget):
            card.setStyleSheet(
                f"SimpleCardWidget {{ background-color: {card_color}; "
                f"border: 1px solid {palette['border']}; border-radius: 8px; }}"
            )
        pane_color = "rgba(243, 246, 249, 235)" if self.is_light_theme else "rgba(15, 17, 21, 235)"
        self.tabs.setStyleSheet(
            f"QTabWidget::pane {{ background-color: {pane_color}; "
            f"border: 1px solid {palette['border']}; border-radius: 8px; }}"
            f"QTabBar::tab {{ color: {palette['muted']}; background: {palette['window']}; "
            "padding: 9px 12px; margin-right: 3px; }"
            f"QTabBar::tab:selected {{ color: {palette['text']}; background: {palette['plot']}; "
            "border-bottom: 2px solid #00a884; }"
        )
        for graph in (self.plot_graph, self.raster_graph, self.rate_graph, self.synaptic_graph):
            graph.setBackground(palette["plot"])
            for axis_name in ("left", "bottom"):
                axis = graph.getAxis(axis_name)
                axis.setPen(pg.mkPen(palette["muted"]))
                axis.setTextPen(pg.mkPen(palette["text"]))
            graph.setLabel("bottom", "Tempo", units="ms", color=palette["text"])
        self.plot_graph.setLabel("left", "Potencial", units="mV", color=palette["text"])
        self.raster_graph.setLabel("left", "Neurônio", color=palette["text"])
        self.curve.setPen(pg.mkPen(palette["curve"], width=1))
        self.threshold_line.setPen(pg.mkPen(palette["muted"], width=1, style=Qt.PenStyle.DashLine))

    def shutdown(self) -> None:
        if self._closing:
            return
        self._closing = True
        self.update_timer.stop()
        self.runner.shutdown()

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=default_cpu_threads(),
                        help="CPU inference threads (default: up to 8, leaving one logical CPU free)")
    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen; Esc exits, F11 toggles")
    args, qt_args = parser.parse_known_args()
    if args.threads < 1:
        parser.error("--threads must be positive.")
    configure_windows_process_identity()
    torch.set_num_threads(args.threads)
    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("Classificador Izhikevich")
    icon = application_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)
    setTheme(Theme.DARK)
    setThemeColor("#00a884")
    window = IzhikevichApp()
    if args.fullscreen:
        window.toggle_fullscreen()
    else:
        window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
