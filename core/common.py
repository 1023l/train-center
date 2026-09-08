"""
通用常量与工具函数（不依赖 GUI），供 label_tool.py、server 等共享。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # train-center/
PROJECTS = ROOT.parent

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

OBJ_CLASSES = ["fabric"]
OCR_CLASSES = ["text_h", "text_v"]


def cv_imread(path) -> np.ndarray | None:
    """支持中文路径的 imread。"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), -1)


def cv_imwrite(path, img) -> bool:
    """支持中文路径的 imwrite。"""
    ext = Path(str(path)).suffix or ".jpg"
    if isinstance(img, np.ndarray) and img.size > 0:
        return cv2.imencode(ext, img)[1].tofile(str(path))
    return False


def list_images(root: Path):
    """递归查找目录下所有图片。"""
    root = Path(root)
    if not root.exists():
        return []
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTS else []
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return out


def parse_line(line: str):
    """把一行 YOLO txt 拆成 (坐标 parts, 文字 text)。tab 分隔文字。"""
    if "\t" in line:
        coord_str, text = line.split("\t", 1)
    else:
        coord_str, text = line, ""
    return coord_str.split(), text.strip()


def parse_box_coords(parts, w, h):
    """解析标注 parts 坐标，返回 (cid, x1, y1, x2, y2) 像素坐标（外接矩形）。
    支持:
      - 矩形(5 值): cid cx cy w h
      - 多边形/旋转框(奇数, >=7): cid x1 y1 x2 y2 ... xn yn（归一化）
    不支持返回 None。
    """
    cid = int(parts[0])
    if len(parts) == 5:
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
    elif len(parts) >= 7 and len(parts) % 2 == 1:
        n_pairs = (len(parts) - 1) // 2
        xs, ys = [], []
        for i in range(n_pairs):
            xs.append(float(parts[1 + i * 2]) * w)
            ys.append(float(parts[2 + i * 2]) * h)
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
    else:
        return None
    return cid, int(x1), int(y1), int(x2), int(y2)
