"""Static neural artwork, fitted once per viewport/theme rather than per frame."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QPainter, QPixmap
from PyQt6.QtWidgets import QWidget


class NeuralBackground(QWidget):
    """A cached, aspect-preserving background with a plain-color fallback.

    There is no timer or animation. Plot updates reuse the composed pixmap;
    resizing, a display-scale change, or a theme/toggle change rebuilds it.
    """

    def __init__(self, image_path: Path, parent=None) -> None:
        super().__init__(parent)
        self._artwork = QPixmap(str(image_path))
        self._light = False
        self._enabled = True
        self._cache_key = None
        self._cache = QPixmap()
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

    def set_light_theme(self, enabled: bool) -> None:
        if self._light != enabled:
            self._light = enabled
            self.update()

    def set_artwork_enabled(self, enabled: bool) -> None:
        if self._enabled != enabled:
            self._enabled = enabled
            self.update()

    def paintEvent(self, event) -> None:
        ratio = self.devicePixelRatioF()
        key = (self.width(), self.height(), ratio, self._light, self._enabled)
        if key != self._cache_key:
            pixels = QSize(max(1, round(self.width() * ratio)), max(1, round(self.height() * ratio)))
            self._cache = QPixmap(pixels)
            self._cache.setDevicePixelRatio(ratio)
            self._cache.fill(QColor("#f3f6f9" if self._light else "#0f1115"))
            if self._enabled and not self._artwork.isNull():
                fitted = self._artwork.scaled(
                    pixels, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation,
                )
                fitted.setDevicePixelRatio(ratio)
                painter = QPainter(self._cache)
                painter.drawPixmap(QPointF(
                    (self.width() - fitted.width() / ratio) / 2,
                    (self.height() - fitted.height() / ratio) / 2,
                ), fitted)
                tint = QColor(243, 246, 249, 222) if self._light else QColor(15, 17, 21, 80)
                painter.fillRect(QRectF(0, 0, self.width(), self.height()), tint)
                painter.end()
            self._cache_key = key
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._cache)
