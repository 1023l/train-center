"""
统一标注工具：obj 模式 + ocr 模式，支持矩形/旋转框/多边形。

用法:
    python label_tool.py --mode obj --input data/raw/
    python label_tool.py --mode ocr --input data/det_text/images/train/
    python label_tool.py --mode ocr --input data/det_text/images/train/ --no-text

工具栏（左侧）:
    V  拖动模式（左键拖动画布）
    B  矩形框
    R  旋转框（4 点）

快捷键:
    W/S       上/下一张图
    D / Del   删除选中框
    右键       删除点击的框
    1/2       切换类别
    T         切换文字录入（ocr 模式）
    Ctrl+Z    撤销
    Ctrl+Y    重做
    Ctrl+S    保存
    Ctrl+E    导出
    F         适配窗口
    滚轮      缩放
    双击框     编辑文字
"""

import argparse
import copy
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PyQt5.QtCore import Qt, QPointF, QRectF, QSizeF, QLineF
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor, QFont, QPainter, QBrush, QPolygonF
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QFileDialog, QMessageBox,
    QListWidget, QListWidgetItem, QSplitter, QGroupBox,
    QCheckBox, QDoubleSpinBox, QStatusBar, QLineEdit, QInputDialog,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
    QGraphicsPolygonItem, QGraphicsTextItem, QMenu, QAction, QToolBar,
    QToolButton, QButtonGroup, QSizePolicy
)

ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def cv_imread(path) -> np.ndarray:
    """支持中文路径的 imread。"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), -1)


def cv_imwrite(path, img) -> bool:
    """支持中文路径的 imwrite。"""
    ext = Path(str(path)).suffix
    return cv2.imencode(ext, img)[1].tofile(str(path))

OBJ_CLASSES = ["fabric"]
OCR_CLASSES = ["text_h", "text_v"]

CLASS_COLORS = {
    "fabric": QColor(0, 220, 0),
    "text_h": QColor(255, 160, 0),
    "text_v": QColor(0, 160, 255),
}

# 工具模式
TOOL_PAN = "pan"       # 平移（左键拖动画布）
TOOL_RECT = "rect"     # 矩形框
TOOL_ROTATE = "rotate" # 旋转框（4 点）


class BaseBox:
    """标注基类，存 YOLO 坐标。"""
    def __init__(self, cls_name, points: list[QPointF], text=""):
        self.cls_name = cls_name
        self.points = points  # 坐标列表
        self.text = text
        self.shape = "rect"   # rect / rotate / polygon


class BoxItem(QGraphicsRectItem):
    """可交互的矩形标注。"""

    def __init__(self, cls_name, x, y, w, h, text="", canvas=None):
        super().__init__(x, y, w, h)
        self.cls_name = cls_name
        self.text = text
        self.canvas = canvas
        self._color = CLASS_COLORS.get(cls_name, QColor(255, 255, 0))

        self.setPen(QPen(self._color, 1))
        self.setBrush(QBrush(QColor(self._color.red(), self._color.green(), self._color.blue(), 30)))

        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setZValue(10)

    def set_text(self, text: str):
        self.text = text

    def setSelected(self, selected):
        super().setSelected(selected)
        self.setPen(QPen(self._color, 2 if selected else 1))

    def mouseDoubleClickEvent(self, event):
        if self.canvas and self.canvas.text_mode:
            text, ok = QInputDialog.getText(
                None, "编辑文字", f"文字内容（{self.cls_name}）:", text=self.text
            )
            if ok:
                self.set_text(text)
                self.canvas.on_box_changed()
        super().mouseDoubleClickEvent(event)

    def to_yolo(self, img_w, img_h):
        r = self.rect()
        cx = (r.x() + r.width() / 2) / img_w
        cy = (r.y() + r.height() / 2) / img_h
        w = r.width() / img_w
        h = r.height() / img_h
        return f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


class PolygonItem(QGraphicsPolygonItem):
    """多边形/旋转框标注。"""

    def __init__(self, cls_name, points: list[QPointF], text="", canvas=None):
        poly = QPolygonF(points)
        super().__init__(poly)
        self.cls_name = cls_name
        self.points = points
        self.text = text
        self.canvas = canvas
        self._color = CLASS_COLORS.get(cls_name, QColor(255, 255, 0))

        self.setPen(QPen(self._color, 1))
        self.setBrush(QBrush(QColor(self._color.red(), self._color.green(), self._color.blue(), 30)))

        self.setFlag(QGraphicsPolygonItem.ItemIsSelectable, True)
        self.setZValue(10)

    def set_text(self, text: str):
        self.text = text

    def setSelected(self, selected):
        super().setSelected(selected)
        self.setPen(QPen(self._color, 2 if selected else 1))

    def mouseDoubleClickEvent(self, event):
        if self.canvas and self.canvas.text_mode:
            text, ok = QInputDialog.getText(
                None, "编辑文字", f"文字内容（{self.cls_name}）:", text=self.text
            )
            if ok:
                self.set_text(text)
                self.canvas.on_box_changed()
        super().mouseDoubleClickEvent(event)

    def to_yolo(self, img_w, img_h):
        coords = []
        for p in self.points:
            coords.append(f"{p.x()/img_w:.6f} {p.y()/img_h:.6f}")
        return " ".join(coords)


class CanvasView(QGraphicsView):
    """画布。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(35, 35, 35))
        self.setInteractive(True)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.pixmap_item = None
        self.image_path = None
        self.image_w = 0
        self.image_h = 0

        self.items_list = []  # BoxItem / PolygonItem
        self.current_class = "fabric"
        self.text_mode = False
        self.tool = TOOL_PAN

        # 画框状态
        self.drawing = False
        self.start_pt = QPointF()
        self.temp_rect = None

        # 旋转框状态
        self.poly_points: list[QPointF] = []
        self.temp_lines = []
        self.temp_preview_line = None  # 鼠标到最后一点的预览线

        # 平移
        self.panning = False
        self.pan_start = QPointF()

        # 撤销/重做
        self.undo_stack: list[list] = []
        self.redo_stack: list[list] = []

    def set_image(self, path: str):
        self.image_path = path
        img = cv_imread(path)
        if img is None:
            return
        self.image_h, self.image_w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.scene.clear()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.pixmap_item.setZValue(0)
        self.scene.setSceneRect(QRectF(0, 0, w, h))
        self.items_list = []
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def add_rect(self, cls_name, x, y, w, h, text=""):
        item = BoxItem(cls_name, x, y, w, h, text, self)
        self.scene.addItem(item)
        self.items_list.append(item)
        return item

    def add_polygon(self, cls_name, points, text=""):
        item = PolygonItem(cls_name, points, text, self)
        self.scene.addItem(item)
        self.items_list.append(item)
        return item

    def snapshot(self):
        """保存当前状态用于撤销。"""
        state = []
        for item in self.items_list:
            if isinstance(item, BoxItem):
                r = item.rect()
                state.append(("rect", item.cls_name, r.x(), r.y(), r.width(), r.height(), item.text))
            elif isinstance(item, PolygonItem):
                pts = [(p.x(), p.y()) for p in item.points]
                state.append(("poly", item.cls_name, pts, item.text))
        self.undo_stack.append(state)
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return
        current = self._snapshot_current()
        self.redo_stack.append(current)
        state = self.undo_stack.pop()
        self._restore_state(state)

    def redo(self):
        if not self.redo_stack:
            return
        current = self._snapshot_current()
        self.undo_stack.append(current)
        state = self.redo_stack.pop()
        self._restore_state(state)

    def _snapshot_current(self):
        state = []
        for item in self.items_list:
            if isinstance(item, BoxItem):
                r = item.rect()
                state.append(("rect", item.cls_name, r.x(), r.y(), r.width(), r.height(), item.text))
            elif isinstance(item, PolygonItem):
                pts = [(p.x(), p.y()) for p in item.points]
                state.append(("poly", item.cls_name, pts, item.text))
        return state

    def _restore_state(self, state):
        for item in self.items_list:
            self.scene.removeItem(item)
        self.items_list = []
        for entry in state:
            if entry[0] == "rect":
                _, cls, x, y, w, h, text = entry
                self.add_rect(cls, x, y, w, h, text)
            elif entry[0] == "poly":
                _, cls, pts, text = entry
                points = [QPointF(x, y) for x, y in pts]
                self.add_polygon(cls, points, text)
        self.on_box_changed()

    def load_yolo_labels(self, txt_path: Path, class_names: list[str]):
        if not txt_path.is_file():
            return
        cls_map = {i: name for i, name in enumerate(class_names)}
        for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
            # 先用 tab 分隔坐标和文字
            if "\t" in line:
                coord_str, text = line.split("\t", 1)
            else:
                coord_str, text = line, ""
            parts = coord_str.split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            cls_name = cls_map.get(cid, class_names[0])

            if len(parts) == 5:
                # 矩形：cid cx cy w h
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x = (cx - bw / 2) * self.image_w
                y = (cy - bh / 2) * self.image_h
                w = bw * self.image_w
                h = bh * self.image_h
                self.add_rect(cls_name, x, y, w, h, text)
            elif len(parts) >= 7 and len(parts) % 2 == 1:
                # 多边形：cid x1 y1 x2 y2 ... xn yn
                n_pairs = (len(parts) - 1) // 2
                pts = []
                for i in range(n_pairs):
                    px = float(parts[1 + i * 2]) * self.image_w
                    py = float(parts[2 + i * 2]) * self.image_h
                    pts.append(QPointF(px, py))
                if len(pts) >= 3:
                    self.add_polygon(cls_name, pts, text)
                elif len(pts) == 2:
                    x1, y1 = pts[0].x(), pts[0].y()
                    x2, y2 = pts[1].x(), pts[1].y()
                    self.add_rect(cls_name, min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1), text)

    def get_yolo_labels(self, class_names: list[str]) -> list[str]:
        lines = []
        cls_map = {name: i for i, name in enumerate(class_names)}
        for item in self.items_list:
            if item.cls_name not in cls_map:
                continue
            cid = cls_map[item.cls_name]
            if isinstance(item, BoxItem):
                yolo = item.to_yolo(self.image_w, self.image_h)
                line = f"{cid} {yolo}"
                if self.text_mode and item.text:
                    line += f"\t{item.text}"
                lines.append(line)
            elif isinstance(item, PolygonItem):
                yolo = item.to_yolo(self.image_w, self.image_h)
                line = f"{cid} {yolo}"
                if self.text_mode and item.text:
                    line += f"\t{item.text}"
                lines.append(line)
        return lines

    def delete_selected(self):
        for item in self.scene.selectedItems():
            if isinstance(item, (BoxItem, PolygonItem)):
                self.snapshot()
                self.items_list.remove(item)
                self.scene.removeItem(item)
                self.on_box_changed()
                return

    def clear_poly_temp(self):
        for line in self.temp_lines:
            self.scene.removeItem(line)
        self.temp_lines.clear()
        if self.temp_preview_line:
            self.scene.removeItem(self.temp_preview_line)
            self.temp_preview_line = None
        self.poly_points.clear()

    def undo_last_point(self):
        """撤销旋转框的最后一个点。"""
        if not self.poly_points:
            return False
        self.poly_points.pop()
        if self.temp_lines:
            line = self.temp_lines.pop()
            self.scene.removeItem(line)
        if not self.poly_points:
            if self.temp_preview_line:
                self.scene.removeItem(self.temp_preview_line)
                self.temp_preview_line = None
        return True

    def on_box_changed(self):
        pass

    def on_mouse_move(self, pos, rect=None):
        pass

    def _item_at(self, pos):
        """获取 pos 处的标注项（处理子项）。"""
        item = self.itemAt(pos)
        if isinstance(item, (BoxItem, PolygonItem)):
            return item
        if item and item.parentItem() and isinstance(item.parentItem(), (BoxItem, PolygonItem)):
            return item.parentItem()
        return None

    def mousePressEvent(self, event):
        if not self.pixmap_item:
            return
        pos = self.mapToScene(event.pos())

        if event.button() == Qt.RightButton:
            # 右键：删除点击的框
            target = self._item_at(event.pos())
            if target:
                for it in self.scene.selectedItems():
                    it.setSelected(False)
                target.setSelected(True)
                self.delete_selected()
            return

        if event.button() == Qt.LeftButton:
            if self.tool == TOOL_PAN:
                # 平移模式：左键拖动平移
                self.panning = True
                self.pan_start = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
                # 点击框也可以选中
                target = self._item_at(event.pos())
                if target:
                    for it in self.scene.selectedItems():
                        it.setSelected(False)
                    target.setSelected(True)
                return

            if self.tool == TOOL_RECT:
                self.drawing = True
                self.start_pt = QPointF(pos)
                self.temp_rect = QGraphicsRectItem(QRectF(pos, QSizeF(1, 1)))
                self.temp_rect.setPen(QPen(QColor(255, 0, 255), 3, Qt.SolidLine))
                self.temp_rect.setZValue(100)
                self.scene.addItem(self.temp_rect)
                return

            if self.tool == TOOL_ROTATE:
                self.poly_points.append(QPointF(pos))
                if len(self.poly_points) >= 2:
                    line = self.scene.addLine(
                        QLineF(self.poly_points[-2], self.poly_points[-1]),
                        QPen(QColor(255, 0, 255), 3)
                    )
                    line.setZValue(100)
                    self.temp_lines.append(line)
                # 旋转框：4 个点自动完成
                if len(self.poly_points) == 4:
                    self._finish_polygon()
                return

    def mouseMoveEvent(self, event):
        pos = self.mapToScene(event.pos())
        # 坐标回调
        if self.on_mouse_move:
            self.on_mouse_move(pos)

        if self.panning:
            delta = event.pos() - self.pan_start
            self.pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return

        if self.drawing and self.temp_rect:
            x = min(self.start_pt.x(), pos.x())
            y = min(self.start_pt.y(), pos.y())
            w = abs(pos.x() - self.start_pt.x())
            h = abs(pos.y() - self.start_pt.y())
            self.temp_rect.setRect(QRectF(x, y, w, h))
            if self.on_mouse_move:
                self.on_mouse_move(pos, QRectF(x, y, w, h))
            return

        # 旋转框预览线（鼠标到最后一个点）
        if self.tool == TOOL_ROTATE and self.poly_points:
            if self.temp_preview_line:
                self.temp_preview_line.setLine(QLineF(self.poly_points[-1], pos))
            else:
                self.temp_preview_line = self.scene.addLine(
                    QLineF(self.poly_points[-1], pos),
                    QPen(QColor(255, 0, 255, 200), 2, Qt.DashLine)
                )
                self.temp_preview_line.setZValue(99)
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.panning:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)
            return

        if not self.drawing:
            super().mouseReleaseEvent(event)
            return

        self.drawing = False
        if self.temp_rect:
            self.scene.removeItem(self.temp_rect)
            self.temp_rect = None
            pos = self.mapToScene(event.pos())
            x = min(self.start_pt.x(), pos.x())
            y = min(self.start_pt.y(), pos.y())
            w = abs(pos.x() - self.start_pt.x())
            h = abs(pos.y() - self.start_pt.y())
            if w < 5 or h < 5:
                return
            self.snapshot()
            text = ""
            if self.text_mode:
                text, ok = QInputDialog.getText(
                    None, "文字录入", f"文字内容（{self.current_class}）:"
                )
                if not ok:
                    text = ""
            self.add_rect(self.current_class, x, y, w, h, text)
            self.on_box_changed()

    def mouseDoubleClickEvent(self, event):
        if self.tool == TOOL_ROTATE and len(self.poly_points) >= 3:
            self._finish_polygon()
        else:
            super().mouseDoubleClickEvent(event)

    def _finish_polygon(self):
        if len(self.poly_points) < 3:
            self.clear_poly_temp()
            return
        self.snapshot()
        text = ""
        if self.text_mode:
            text, ok = QInputDialog.getText(
                None, "文字录入", f"文字内容（{self.current_class}）:"
            )
            if not ok:
                text = ""
        self.add_polygon(self.current_class, list(self.poly_points), text)
        self.clear_poly_temp()
        self.on_box_changed()

    def wheelEvent(self, event):
        if not self.pixmap_item:
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)

    def contextMenuEvent(self, event):
        target = self._item_at(event.pos())
        if target:
            menu = QMenu(self)
            act_del = QAction("删除", self)
            act_del.triggered.connect(self.delete_selected)
            menu.addAction(act_del)
            if self.text_mode:
                act_edit = QAction("编辑文字", self)
                act_edit.triggered.connect(lambda: target.mouseDoubleClickEvent(None))
                menu.addAction(act_edit)
            menu.exec_(event.globalPos())
        else:
            super().contextMenuEvent(event)


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mode = args.mode
        self.input_dir = Path(args.input) if args.input else None
        self.images: list[Path] = []
        self.current_idx = 0
        self.text_enabled = (args.mode == "ocr" and not args.no_text)

        self.setWindowTitle(f"标注工具 - {self.mode.upper()}")
        self.resize(1500, 950)

        self._init_ui()
        self._load_classes()
        if self.input_dir and self.input_dir.is_dir():
            self._load_images()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # === 顶部工具栏 ===
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)

        self.btn_open = QPushButton("📂 打开目录")
        self.btn_open.setMinimumHeight(36)
        self.btn_open.clicked.connect(self._open_dir)
        toolbar.addWidget(self.btn_open)

        self.dir_label = QLabel("未选择")
        self.dir_label.setStyleSheet("color: #aaa; font-size: 13px; padding: 0 8px;")
        toolbar.addWidget(self.dir_label)

        toolbar.addSpacing(15)

        toolbar.addWidget(QLabel("模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["obj", "ocr"])
        self.mode_combo.setCurrentText(self.mode)
        self.mode_combo.setMinimumHeight(32)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        toolbar.addWidget(self.mode_combo)

        toolbar.addSpacing(10)

        toolbar.addWidget(QLabel("类别:"))
        self.cls_combo = QComboBox()
        self.cls_combo.setMinimumHeight(32)
        self.cls_combo.currentTextChanged.connect(self._on_class_changed)
        toolbar.addWidget(self.cls_combo)

        self.text_checkbox = QCheckBox("文字录入")
        self.text_checkbox.setChecked(self.text_enabled)
        self.text_checkbox.setStyleSheet("font-size: 14px;")
        self.text_checkbox.toggled.connect(self._on_text_toggled)
        toolbar.addWidget(self.text_checkbox)

        toolbar.addStretch()

        self.btn_prev = QPushButton("◀ 上一张")
        self.btn_prev.setMinimumHeight(36)
        self.btn_prev.clicked.connect(self._prev_image)
        toolbar.addWidget(self.btn_prev)

        self.img_label = QLabel("0/0")
        self.img_label.setStyleSheet("color: #8cf; font-size: 14px; padding: 0 8px;")
        toolbar.addWidget(self.img_label)

        self.btn_next = QPushButton("下一张 ▶")
        self.btn_next.setMinimumHeight(36)
        self.btn_next.clicked.connect(self._next_image)
        toolbar.addWidget(self.btn_next)

        self.btn_del = QPushButton("🗑 删除")
        self.btn_del.setMinimumHeight(36)
        self.btn_del.clicked.connect(self._delete_box)
        toolbar.addWidget(self.btn_del)

        self.btn_save = QPushButton("💾 保存")
        self.btn_save.setMinimumHeight(36)
        self.btn_save.clicked.connect(self._save)
        toolbar.addWidget(self.btn_save)

        self.btn_export = QPushButton("📦 导出")
        self.btn_export.setMinimumHeight(36)
        self.btn_export.clicked.connect(self._export)
        toolbar.addWidget(self.btn_export)

        layout.addLayout(toolbar)

        # === 中间区域 ===
        body = QHBoxLayout()
        body.setSpacing(4)

        # 左侧工具栏
        tool_bar = QVBoxLayout()
        tool_bar.setSpacing(4)

        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)

        self.btn_pan = QToolButton()
        self.btn_pan.setText("拖动\n(V)")
        self.btn_pan.setCheckable(True)
        self.btn_pan.setChecked(True)
        self.btn_pan.setMinimumSize(60, 50)
        self.btn_pan.setStyleSheet("QToolButton { font-size: 13px; }")
        self.tool_group.addButton(self.btn_pan, 0)
        tool_bar.addWidget(self.btn_pan)

        self.btn_rect = QToolButton()
        self.btn_rect.setText("矩形\n(B)")
        self.btn_rect.setCheckable(True)
        self.btn_rect.setMinimumSize(60, 50)
        self.btn_rect.setStyleSheet("QToolButton { font-size: 13px; }")
        self.tool_group.addButton(self.btn_rect, 1)
        tool_bar.addWidget(self.btn_rect)

        self.btn_rotate = QToolButton()
        self.btn_rotate.setText("旋转框\n(R)")
        self.btn_rotate.setCheckable(True)
        self.btn_rotate.setMinimumSize(60, 50)
        self.btn_rotate.setStyleSheet("QToolButton { font-size: 13px; }")
        self.tool_group.addButton(self.btn_rotate, 2)
        tool_bar.addWidget(self.btn_rotate)

        tool_bar.addStretch()

        # 撤销/重做按钮
        self.btn_undo = QToolButton()
        self.btn_undo.setText("⤺\n撤销")
        self.btn_undo.setMinimumSize(60, 50)
        self.btn_undo.setStyleSheet("QToolButton { font-size: 13px; }")
        self.btn_undo.clicked.connect(self._undo)
        tool_bar.addWidget(self.btn_undo)

        self.btn_redo = QToolButton()
        self.btn_redo.setText("⤻\n重做")
        self.btn_redo.setMinimumSize(60, 50)
        self.btn_redo.setStyleSheet("QToolButton { font-size: 13px; }")
        self.btn_redo.clicked.connect(self._redo)
        tool_bar.addWidget(self.btn_redo)

        self.tool_group.buttonClicked.connect(self._on_tool_changed)

        tool_widget = QWidget()
        tool_widget.setLayout(tool_bar)
        tool_widget.setFixedWidth(70)
        body.addWidget(tool_widget)

        # 画布
        self.canvas = CanvasView()
        self.canvas.text_mode = self.text_enabled
        self.canvas.on_box_changed = self._update_box_list
        self.canvas.on_mouse_move = self._on_mouse_move
        body.addWidget(self.canvas, 1)

        # 右侧栏
        sidebar = QWidget()
        sidebar.setFixedWidth(280)
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setSpacing(6)

        gb_boxes = QGroupBox("标注框列表")
        gb_layout = QVBoxLayout()
        self.box_list = QListWidget()
        self.box_list.currentRowChanged.connect(self._on_box_selected)
        self.box_list.setStyleSheet("font-size: 13px;")
        gb_layout.addWidget(self.box_list)
        gb_boxes.setLayout(gb_layout)
        sb_layout.addWidget(gb_boxes)

        gb_export = QGroupBox("导出设置")
        exp_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("训练比例:"))
        self.ratio_spin = QDoubleSpinBox()
        self.ratio_spin.setRange(0.1, 0.9)
        self.ratio_spin.setSingleStep(0.05)
        self.ratio_spin.setValue(0.8)
        row1.addWidget(self.ratio_spin)
        exp_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("随机种子:"))
        self.seed_edit = QLineEdit("42")
        row2.addWidget(self.seed_edit)
        exp_layout.addLayout(row2)

        gb_export.setLayout(exp_layout)
        sb_layout.addWidget(gb_export)

        gb_help = QGroupBox("快捷键")
        help_layout = QVBoxLayout()
        help_text = QLabel(
            "V  拖动（左键拖动画布）\n"
            "B  矩形框\n"
            "R  旋转框（4点）\n"
            "\n"
            "W/S  翻页\n"
            "D/Del  删除框\n"
            "右键  删除点击的框\n"
            "1/2  切换类别\n"
            "T  文字录入开关\n"
            "Ctrl+Z  撤销\n"
            "Ctrl+Y  重做\n"
            "Ctrl+S  保存\n"
            "Ctrl+E  导出\n"
            "F  适配窗口\n"
            "滚轮  缩放\n"
            "双击框  编辑文字"
        )
        help_text.setStyleSheet("font-size: 14px; color: #aaa;")
        help_layout.addWidget(help_text)
        gb_help.setLayout(help_layout)
        sb_layout.addWidget(gb_help)

        sb_layout.addStretch()
        body.addWidget(sidebar)

        layout.addLayout(body, 1)

        self.status = QStatusBar()
        self.status.setStyleSheet("font-size: 13px;")
        self.coord_label = QLabel("")
        self.coord_label.setStyleSheet("color: #0f0; font-size: 14px; padding: 0 8px;")
        self.status.addPermanentWidget(self.coord_label)
        self.setStatusBar(self.status)

    def _load_classes(self):
        self.cls_combo.clear()
        classes = OBJ_CLASSES if self.mode == "obj" else OCR_CLASSES
        self.cls_combo.addItems(classes)
        if classes:
            self.canvas.current_class = classes[0]
        self.text_checkbox.setVisible(self.mode == "ocr")

    def _on_mode_changed(self, mode: str):
        self.mode = mode
        self.setWindowTitle(f"标注工具 - {mode.upper()}")
        self._load_classes()

    def _on_class_changed(self, cls: str):
        self.canvas.current_class = cls

    def _on_text_toggled(self, checked: bool):
        self.text_enabled = checked
        self.canvas.text_mode = checked

    def _on_tool_changed(self, btn):
        btn_id = self.tool_group.id(btn)
        tools = [TOOL_PAN, TOOL_RECT, TOOL_ROTATE]
        self.canvas.tool = tools[btn_id]
        self.canvas.clear_poly_temp()
        self.status.showMessage(f"工具: {self.canvas.tool}")

    def _undo(self):
        self.canvas.undo()
        self._update_box_list()

    def _redo(self):
        self.canvas.redo()
        self._update_box_list()

    def _open_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if d:
            self.input_dir = Path(d)
            self._load_images()

    def _load_images(self):
        self.images = sorted([
            f for f in self.input_dir.rglob("*")
            if f.suffix.lower() in IMAGE_EXTS
        ])
        self.dir_label.setText(str(self.input_dir))
        self.current_idx = 0
        if self.images:
            self._show_image()
        else:
            self.status.showMessage("未找到图片")
            self.img_label.setText("0/0")

    def _show_image(self):
        if not self.images:
            return
        path = self.images[self.current_idx]
        self.canvas.set_image(str(path))
        self.canvas.load_yolo_labels(
            path.with_suffix(".txt"),
            OBJ_CLASSES if self.mode == "obj" else OCR_CLASSES
        )
        self.img_label.setText(f"{self.current_idx+1}/{len(self.images)}")
        self.status.showMessage(
            f"[{self.current_idx+1}/{len(self.images)}] {path.name}  "
            f"({self.canvas.image_w}×{self.canvas.image_h})"
        )
        self._update_box_list()

    def _save(self):
        if not self.canvas.image_path:
            return
        img_path = Path(self.canvas.image_path)
        txt_path = img_path.with_suffix(".txt")
        classes = OBJ_CLASSES if self.mode == "obj" else OCR_CLASSES
        lines = self.canvas.get_yolo_labels(classes)
        txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self.status.showMessage(f"已保存: {txt_path.name}  ({len(lines)} 个框)")

    def _next_image(self):
        if self.current_idx < len(self.images) - 1:
            self._save()
            self.current_idx += 1
            self._show_image()

    def _prev_image(self):
        if self.current_idx > 0:
            self._save()
            self.current_idx -= 1
            self._show_image()

    def _delete_box(self):
        self.canvas.delete_selected()
        self._update_box_list()

    def _on_box_selected(self, idx):
        for item in self.canvas.items_list:
            item.setSelected(False)
        if 0 <= idx < len(self.canvas.items_list):
            self.canvas.items_list[idx].setSelected(True)
        self.canvas.viewport().update()

    def _update_box_list(self):
        self.box_list.clear()
        for i, item in enumerate(self.canvas.items_list):
            shape_tag = "▢" if isinstance(item, BoxItem) else "◈"
            label = f"{i} {shape_tag} {item.cls_name}"
            if item.text:
                label += f": {item.text}"
            self.box_list.addItem(QListWidgetItem(label))
        self.status.showMessage(
            f"[{self.current_idx+1}/{len(self.images)}] "
            f"{Path(self.canvas.image_path).name if self.canvas.image_path else ''}  "
            f"({len(self.canvas.items_list)} 个框)"
        )

    def _on_mouse_move(self, pos, rect=None):
        """鼠标移动时显示像素坐标，画框时额外显示框尺寸。"""
        px, py = int(pos.x()), int(pos.y())
        if rect:
            w, h = int(rect.width()), int(rect.height())
            self.coord_label.setText(f"({px}, {py})  框: {w}×{h}")
        else:
            self.coord_label.setText(f"({px}, {py})")

    def _export(self):
        if not self.images:
            QMessageBox.warning(self, "导出", "没有图片可导出")
            return
        self._save()
        ratio = self.ratio_spin.value()
        seed = int(self.seed_edit.text() or "42")
        if self.mode == "obj":
            self._export_obj(ratio, seed)
        else:
            self._export_ocr(ratio, seed)

    def _get_annotated_images(self):
        """只返回有标注（txt 存在且非空）的图片路径列表。"""
        result = []
        for img_path in self.images:
            txt_path = img_path.with_suffix(".txt")
            if txt_path.is_file() and txt_path.read_text(encoding="utf-8").strip():
                result.append(img_path)
        return result

    def _strip_text_from_txt(self, txt_path, dest_path):
        """复制 txt 但去掉 tab 后的文字内容，只留 YOLO 坐标格式。"""
        lines_out = []
        for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
            if "\t" in line:
                line = line.split("\t", 1)[0]
            lines_out.append(line)
        dest_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")

    def _export_obj(self, ratio, seed):
        out_dir = ROOT / "data" / "det_fabric"
        det_text_img_dir = ROOT / "data" / "det_text" / "images"
        # 只导出有标注的图
        annotated = self._get_annotated_images()
        if not annotated:
            QMessageBox.warning(self, "导出", "没有已标注的图片（txt 为空或不存在）")
            return
        random.seed(seed)
        indices = list(range(len(annotated)))
        random.shuffle(indices)
        split_idx = max(1, int(len(indices) * ratio))
        for split in ("train", "val"):
            (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        det_text_img_dir.mkdir(parents=True, exist_ok=True)
        n_train = n_val = 0
        for i, img_idx in enumerate(indices):
            split = "train" if i < split_idx else "val"
            img_path = annotated[img_idx]
            txt_path = img_path.with_suffix(".txt")
            shutil.copy2(img_path, out_dir / "images" / split / img_path.name)
            self._strip_text_from_txt(txt_path, out_dir / "labels" / split / f"{img_path.stem}.txt")
            self._crop_pieces(img_path, txt_path, det_text_img_dir)
            if split == "train":
                n_train += 1
            else:
                n_val += 1
        self._write_yaml(out_dir, OBJ_CLASSES)
        QMessageBox.information(self, "导出完成",
            f"det_fabric → {out_dir}\n"
            f"  train: {n_train} 张, val: {n_val} 张\n"
            f"布片裁剪 → {det_text_img_dir}\n\n"
            f"下一步:\n1. label_tool.py --mode ocr --input {det_text_img_dir}\n"
            f"2. train.py --dataset det_fabric")

    def _export_ocr(self, ratio, seed):
        out_dir = ROOT / "data" / "det_text"
        rec_dir = ROOT / "data" / "rec_text"
        annotated = self._get_annotated_images()
        if not annotated:
            QMessageBox.warning(self, "导出", "没有已标注的图片（txt 为空或不存在）")
            return
        random.seed(seed)
        indices = list(range(len(annotated)))
        random.shuffle(indices)
        split_idx = max(1, int(len(indices) * ratio))
        for split in ("train", "val"):
            (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
            (rec_dir / "crop_img" / split).mkdir(parents=True, exist_ok=True)
        rec_lines = {"train": [], "val": []}
        n_train = n_val = 0
        for i, img_idx in enumerate(indices):
            split = "train" if i < split_idx else "val"
            img_path = annotated[img_idx]
            txt_path = img_path.with_suffix(".txt")
            shutil.copy2(img_path, out_dir / "images" / split / img_path.name)
            self._strip_text_from_txt(txt_path, out_dir / "labels" / split / f"{img_path.stem}.txt")
            if self.text_enabled:
                rec_lines[split].extend(
                    self._crop_text(img_path, txt_path, rec_dir / "crop_img" / split, split)
                )
            if split == "train":
                n_train += 1
            else:
                n_val += 1
        self._write_yaml(out_dir, OCR_CLASSES)
        for split in ("train", "val"):
            gt_path = rec_dir / f"{split}.txt"
            with gt_path.open("w", encoding="utf-8") as f:
                for line in rec_lines[split]:
                    f.write(line + "\n")
        msg = f"det_text → {out_dir}\n  train: {n_train} 张, val: {n_val} 张\n"
        if self.text_enabled:
            msg += f"rec_text → {rec_dir}\n\n下一步:\n1. train.py --dataset det_text\n2. PaddleOCR rec 增量训练"
        else:
            msg += f"\n下一步: train.py --dataset det_text"
        QMessageBox.information(self, "导出完成", msg)

    def _parse_box_coords(self, parts, w, h):
        """解析标注行，返回 (cid, x1, y1, x2, y2, text) 像素坐标的外接矩形。
        支持矩形(5值)和4点旋转框(9值)两种格式，用 tab 分隔文字。
        """
        cid = int(parts[0])
        if len(parts) == 5:
            # 矩形: cid cx cy w h
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            x2 = (cx + bw/2) * w
            y2 = (cy + bh/2) * h
            text = ""
        elif len(parts) >= 7 and len(parts) % 2 == 1:
            # 4点旋转框: cid x1 y1 x2 y2 x3 y3 x4 y4
            n_pairs = (len(parts) - 1) // 2
            xs, ys = [], []
            for i in range(n_pairs):
                xs.append(float(parts[1 + i * 2]) * w)
                ys.append(float(parts[2 + i * 2]) * h)
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            text = ""
        else:
            # 不支持的格式，跳过
            return None
        return cid, int(x1), int(y1), int(x2), int(y2), text

    def _crop_pieces(self, img_path, txt_path, out_dir):
        img = cv_imread(img_path)
        if img is None or not txt_path.is_file():
            return
        h, w = img.shape[:2]
        idx = 0
        for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
            if "\t" in line:
                coord_str = line.split("\t", 1)[0]
            else:
                coord_str = line
            parts = coord_str.split()
            if len(parts) < 5:
                continue
            result = self._parse_box_coords(parts, w, h)
            if result is None:
                continue
            cid, x1, y1, x2, y2, _ = result
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            piece = img[y1:y2, x1:x2]
            if piece.size > 0:
                name = f"{img_path.stem}_{idx}.jpg"
                cv_imwrite(str(out_dir / name), piece)
                idx += 1

    def _crop_text(self, img_path, txt_path, out_dir, split="train"):
        img = cv_imread(img_path)
        if img is None or not txt_path.is_file():
            return []
        h, w = img.shape[:2]
        lines_out = []
        idx = 0
        for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
            if "\t" in line:
                coord_str, text = line.split("\t", 1)
            else:
                coord_str, text = line, ""
            parts = coord_str.split()
            if len(parts) < 5:
                continue
            result = self._parse_box_coords(parts, w, h)
            if result is None:
                continue
            cid, x1, y1, x2, y2, _ = result
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                name = f"{img_path.stem}_{idx}.jpg"
                cv_imwrite(str(out_dir / name), crop)
                lines_out.append(f"crop_img/{split}/{name}\t{text}")
                idx += 1
        return lines_out

    def _write_yaml(self, out_dir, class_names):
        cfg = {
            "path": str(out_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(class_names),
            "names": {i: name for i, name in enumerate(class_names)},
        }
        (out_dir / "data.yaml").write_text(
            yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8"
        )

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()

        if mods & Qt.ControlModifier:
            if key == Qt.Key_S:
                self._save()
            elif key == Qt.Key_E:
                self._export()
            elif key == Qt.Key_Z:
                # 画旋转框时撤销最后一个点，否则撤销整个操作
                if self.canvas.tool == TOOL_ROTATE and self.canvas.poly_points:
                    self.canvas.undo_last_point()
                else:
                    self._undo()
            elif key == Qt.Key_Y:
                self._redo()
            return

        if key == Qt.Key_W:
            self._prev_image()
        elif key == Qt.Key_S:
            self._next_image()
        elif key in (Qt.Key_D, Qt.Key_Delete):
            self._delete_box()
        elif key == Qt.Key_1:
            self.cls_combo.setCurrentIndex(0)
        elif key == Qt.Key_2 and self.mode == "ocr":
            self.cls_combo.setCurrentIndex(1)
        elif key == Qt.Key_T and self.mode == "ocr":
            self.text_checkbox.toggle()
        elif key == Qt.Key_F:
            if self.canvas.pixmap_item:
                self.canvas.fitInView(self.canvas.scene.sceneRect(), Qt.KeepAspectRatio)
        elif key == Qt.Key_V:
            self.btn_pan.setChecked(True)
            self._on_tool_changed(self.btn_pan)
        elif key == Qt.Key_B:
            self.btn_rect.setChecked(True)
            self._on_tool_changed(self.btn_rect)
        elif key == Qt.Key_R:
            self.btn_rotate.setChecked(True)
            self._on_tool_changed(self.btn_rotate)
        else:
            super().keyPressEvent(event)


def parse_args():
    p = argparse.ArgumentParser(description="统一标注工具")
    p.add_argument("--mode", choices=["obj", "ocr"], default="obj")
    p.add_argument("--input", type=str, default=None)
    p.add_argument("--no-text", action="store_true")
    return p.parse_args()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QMainWindow { background: #2b2b2b; }
        QWidget { color: #ddd; font-size: 14px; }
        QPushButton {
            background: #3c3f41; border: 1px solid #555;
            padding: 6px 14px; border-radius: 4px; font-size: 14px;
        }
        QPushButton:hover { background: #4c5052; }
        QPushButton:pressed { background: #2d2d2d; }
        QToolButton {
            background: #3c3f41; border: 1px solid #555;
            border-radius: 4px; font-size: 13px;
        }
        QToolButton:checked { background: #4a7fb5; border-color: #6a9fd5; }
        QToolButton:hover { background: #4c5052; }
        QComboBox {
            background: #3c3f41; border: 1px solid #555;
            padding: 5px; border-radius: 4px; min-height: 20px;
        }
        QListWidget { background: #1e1e1e; border: 1px solid #555; font-size: 14px; }
        QGroupBox {
            border: 1px solid #555; border-radius: 4px;
            margin-top: 10px; padding-top: 14px; font-size: 14px;
        }
        QGroupBox::title { color: #8cf; left: 10px; padding: 0 4px; }
        QCheckBox { spacing: 6px; }
        QLineEdit {
            background: #1e1e1e; border: 1px solid #555;
            padding: 4px; border-radius: 3px;
        }
        QStatusBar { background: #1e1e1e; color: #8cf; }
        QLabel { font-size: 14px; }
    """)
    win = MainWindow(parse_args())
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
