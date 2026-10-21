"""Lazy table rows: large populations do not allocate a widget for every cell."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt

from config import CLASS_CODES

if TYPE_CHECKING:
    from population_worker import PopulationResult


class PopulationTableModel(QAbstractTableModel):
    HEADERS = ("Neurônio", "Esperado", "Previsto", "Confiança", "Disparos", "Estado")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.result: PopulationResult | None = None

    def set_result(self, result: PopulationResult | None) -> None:
        self.beginResetModel()
        self.result = result
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return self.result.population.size if self.result is not None and not parent.isValid() else 0

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section] if 0 <= section < len(self.HEADERS) else None
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self.result is None or not 0 <= index.row() < self.rowCount():
            return None
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        if role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return None
        row, column = index.row(), index.column()
        population = self.result.population
        expected = CLASS_CODES[int(population.labels[row])]
        prediction = self.result.predictions[row] if self.result.predictions is not None else None
        if prediction is None:
            predicted, confidence, state = "—", "—", "Sem modelo"
        elif not prediction.determined:
            predicted = f"{prediction.class_code} (?)" if prediction.class_code else "—"
            confidence = f"{prediction.confidence:.1%}" if prediction.class_code else "—"
            state = "Indeterminado"
        else:
            predicted = prediction.class_code
            confidence = f"{prediction.confidence:.1%}"
            state = "Correto" if predicted == expected else "Divergente"
        if role == Qt.ItemDataRole.ToolTipRole and prediction is not None and prediction.reason:
            return prediction.reason
        values = (row + 1, expected, predicted, confidence, int(population.spike_counts[row]), state)
        return values[column] if 0 <= column < len(values) else None
