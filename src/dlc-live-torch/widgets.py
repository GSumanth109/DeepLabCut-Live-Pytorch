# src/dlc_live_torch/widgets.py

import threading
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QRect, QPoint
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor

class CroppableLabel(QLabel):
    """ A custom QLabel that allows drawing a cropping rectangle. """
    crop_finished = pyqtSignal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True); self._is_cropping = False; self._is_enabled = False
        self._crop_rect = QRect(); self._start_point = QPoint(); self._end_point = QPoint()

    def setEnabled(self, enabled):
        self._is_enabled = enabled
        if not enabled: self.clear_crop()
        self.update()

    def mousePressEvent(self, event):
        if self._is_enabled and event.button() == Qt.LeftButton:
            self._is_cropping = True; self._start_point = event.pos(); self._end_point = event.pos(); self.update()

    def mouseMoveEvent(self, event):
        if self._is_enabled:
            self.setCursor(Qt.CrossCursor)
            if self._is_cropping: self._end_point = event.pos(); self.update()
        else: self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if self._is_enabled and event.button() == Qt.LeftButton:
            self._is_cropping = False; final_rect = QRect(self._start_point, self._end_point).normalized()
            if final_rect.width() > 5 and final_rect.height() > 5: self._crop_rect = final_rect; self.crop_finished.emit(self._crop_rect)
            else: self.clear_crop()
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._is_enabled:
            painter = QPainter(self); draw_rect = QRect()
            if self._is_cropping: draw_rect = QRect(self._start_point, self._end_point).normalized()
            elif not self._crop_rect.isNull(): draw_rect = self._crop_rect
            if not draw_rect.isNull() and draw_rect.isValid():
                pen = QPen(QColor(255,0,0,200), 2, Qt.SolidLine); brush = QBrush(QColor(255,0,0,50))
                painter.setPen(pen); painter.setBrush(brush); painter.drawRect(draw_rect)
            painter.end()

    def set_existing_crop(self, qrect):
        self._crop_rect = qrect
        if not qrect.isNull(): self._start_point = qrect.topLeft(); self._end_point = qrect.bottomRight()
        else: self._start_point = QPoint(); self._end_point = QPoint()
        self.update()

    def clear_crop(self):
        self._crop_rect = QRect(); self._start_point = QPoint(); self._end_point = QPoint(); self.update()

class CropManager:
    """ Thread-safe storage for crop coordinates (x1, y1, x2, y2). """
    def __init__(self): self._lock = threading.Lock(); self.crop_rect = None
    def set(self, rect_tuple):
        with self._lock: self.crop_rect = rect_tuple
    def get(self):
        with self._lock: return self.crop_rect
    def clear(self):
        with self._lock: self.crop_rect = None
    def get_tuple(self):
        with self._lock: return self.crop_rect
    def set_from_tuple(self, rect_tuple):
        if isinstance(rect_tuple, (list, tuple)) and len(rect_tuple) == 4:
            with self._lock: self.crop_rect = tuple(map(int, rect_tuple))
        else:
            with self._lock: self.crop_rect = None

class ActiveQueueReference:
    """ Thread-safe reference to the active multiprocessing queue. """
    def __init__(self): self._lock = threading.Lock(); self._queue = None
    def set(self, queue):
        with self._lock: self._queue = queue
    def get(self):
        with self._lock: return self._queue
