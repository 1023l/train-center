"""
数据导出工具（不依赖 GUI）：把标注好的图片 → 训练数据集。

导出逻辑和 label_tool.py 的 _export_obj / _export_ocr 保持一致，只是去掉了
QMessageBox 弹窗，返回结构化结果 dict。
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import yaml

from .common import (
    ROOT,
    OBJ_CLASSES,
    OCR_CLASSES,
    cv_imread,
    cv_imwrite,
    parse_box_coords,
    parse_line,
)


# ---- 内部辅助 ----

def _strip_text(txt_path, dest_path, img_shape=None):
    """
    把 txt 里的标注统一转成 YOLO 矩形 5 值（cid cx cy w h，归一化 0-1），去掉文字。
    - 原行是 5 值（矩形）：保持（已经是 cx cy w h 归一化格式）
    - 原行 ≥7 奇数（4 点/多边形）：parse_box_coords 算出轴对齐 bbox 像素坐标，再归一化
    - 若无法拿到 img_shape 且原行不是 5 值：该行跳过防止错数据
    """
    txt_path = Path(txt_path)
    dest_path = Path(dest_path)
    lines_out = []
    w = h = None
    if img_shape is not None:
        h, w = img_shape[0], img_shape[1]
    for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
        parts, _text = parse_line(line)
        if len(parts) < 5:
            continue
        # 5 值（cid cx cy w h）：直接用
        if len(parts) == 5:
            lines_out.append(" ".join(str(x) for x in parts))
            continue
        # ≥7 奇数（cid x1..yN）：按像素坐标取 AABB 后归一化（parse_box_coords 需要 w,h 像素拿到 bbox 像素）
        if len(parts) >= 7 and len(parts) % 2 == 1:
            if w is None or h is None:
                continue
            result = parse_box_coords(parts, w, h)
            if result is None:
                continue
            cid, x1, y1, x2, y2 = result
            cx = ((x1 + x2) * 0.5) / w
            cy = ((y1 + y2) * 0.5) / h
            bw = (max(x1, x2) - min(x1, x2)) / w
            bh = (max(y1, y2) - min(y1, y2)) / h
            # 夹到 [0,1] 安全
            cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
            bw, bh = min(max(bw, 1e-6), 1.0), min(max(bh, 1e-6), 1.0)
            lines_out.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(
        "\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8"
    )


def _write_yaml(out_dir: Path, class_names):
    cfg = {
        "path": str(Path(out_dir).resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }
    (out_dir / "data.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8"
    )


def _annotated_images(image_paths):
    """从图片列表里过滤出有标注（txt 存在且非空）的。"""
    result = []
    for p in image_paths:
        p = Path(p)
        txt = p.with_suffix(".txt")
        if txt.is_file() and txt.read_text(encoding="utf-8").strip():
            result.append(p)
    return result


def _split_indices(n, ratio, seed):
    """返回 (train_indices, val_indices)。保证 train 至少 1，集合 size>1 时 val 至少 1。"""
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    split_idx = max(1, int(n * ratio))
    if split_idx >= n and n > 1:
        split_idx = n - 1  # 保证 val 非空
    return indices[:split_idx], indices[split_idx:]


def _crop_pieces(img_path, txt_path, out_dir):
    """从原图按标注框裁剪布片（obj 导出用）。"""
    img = cv_imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = Path(txt_path)
    if not txt_path.is_file():
        return
    idx = 0
    for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
        parts, _ = parse_line(line)
        if len(parts) < 5:
            continue
        result = parse_box_coords(parts, w, h)
        if result is None:
            continue
        _, x1, y1, x2, y2 = result
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        piece = img[y1:y2, x1:x2]
        if piece.size > 0:
            name = f"{Path(img_path).stem}_{idx}.jpg"
            cv_imwrite(out_dir / name, piece)
            idx += 1


def _crop_text(img_path, txt_path, out_dir, split):
    """从布片按文字框裁剪小图，返回 rec gt_lines。"""
    img = cv_imread(img_path)
    if img is None:
        return []
    h, w = img.shape[:2]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = Path(txt_path)
    if not txt_path.is_file():
        return []
    lines_out = []
    idx = 0
    for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
        parts, text = parse_line(line)
        if len(parts) < 5:
            continue
        result = parse_box_coords(parts, w, h)
        if result is None:
            continue
        _, x1, y1, x2, y2 = result
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = img[y1:y2, x1:x2]
        if crop.size > 0:
            name = f"{Path(img_path).stem}_{idx}.jpg"
            cv_imwrite(out_dir / name, crop)
            lines_out.append(f"crop_img/{split}/{name}\t{text}")
            idx += 1
    return lines_out


# ---- 公开 API ----

def export_obj(image_paths, ratio: float = 0.8, seed: int = 42, out_dir=None, det_text_img_dir=None):
    """导出 obj 模式标注 → det_fabric + 布片裁剪到 det_text/images。"""
    out_dir = Path(out_dir) if out_dir else ROOT / "data" / "det_fabric"
    det_text_img_dir = Path(det_text_img_dir) if det_text_img_dir else ROOT / "data" / "det_text" / "images"
    annotated = _annotated_images(image_paths)
    if not annotated:
        return {"ok": False, "msg": "没有已标注图片"}

    train_idx, val_idx = _split_indices(len(annotated), ratio, seed)
    for s in ("train", "val"):
        (out_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / s).mkdir(parents=True, exist_ok=True)
    det_text_img_dir.mkdir(parents=True, exist_ok=True)

    n_train = n_val = 0
    for i in train_idx:
        img = annotated[i]
        txt = img.with_suffix(".txt")
        shutil.copy2(img, out_dir / "images" / "train" / img.name)
        _img = cv_imread(img)
        _shape = _img.shape[:2] if _img is not None else None
        _strip_text(txt, out_dir / "labels" / "train" / f"{img.stem}.txt", _shape)
        _crop_pieces(img, txt, det_text_img_dir)
        n_train += 1
    for i in val_idx:
        img = annotated[i]
        txt = img.with_suffix(".txt")
        shutil.copy2(img, out_dir / "images" / "val" / img.name)
        _img = cv_imread(img)
        _shape = _img.shape[:2] if _img is not None else None
        _strip_text(txt, out_dir / "labels" / "val" / f"{img.stem}.txt", _shape)
        _crop_pieces(img, txt, det_text_img_dir)
        n_val += 1

    _write_yaml(out_dir, OBJ_CLASSES)
    return {
        "ok": True,
        "out_dir": str(out_dir),
        "det_text_img_dir": str(det_text_img_dir),
        "train": n_train,
        "val": n_val,
        "class_names": OBJ_CLASSES,
    }


def export_ocr(image_paths, ratio: float = 0.8, seed: int = 42, text_enabled: bool = True,
               out_dir=None, rec_dir=None):
    """导出 ocr 模式标注 → det_text（文字检测） + rec_text（文字识别，可选）。"""
    out_dir = Path(out_dir) if out_dir else ROOT / "data" / "det_text"
    rec_dir = Path(rec_dir) if rec_dir else ROOT / "data" / "rec_text"
    annotated = _annotated_images(image_paths)
    if not annotated:
        return {"ok": False, "msg": "没有已标注图片"}

    train_idx, val_idx = _split_indices(len(annotated), ratio, seed)
    for s in ("train", "val"):
        (out_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / s).mkdir(parents=True, exist_ok=True)
        (rec_dir / "crop_img" / s).mkdir(parents=True, exist_ok=True)

    rec_lines = {"train": [], "val": []}
    n_train = n_val = 0
    for split, idxs in (("train", train_idx), ("val", val_idx)):
        for i in idxs:
            img = annotated[i]
            txt = img.with_suffix(".txt")
            shutil.copy2(img, out_dir / "images" / split / img.name)
            _img = cv_imread(img)
            _shape = _img.shape[:2] if _img is not None else None
            _strip_text(txt, out_dir / "labels" / split / f"{img.stem}.txt", _shape)
            if text_enabled:
                rec_lines[split].extend(
                    _crop_text(img, txt, rec_dir / "crop_img" / split, split)
                )
            if split == "train":
                n_train += 1
            else:
                n_val += 1

    _write_yaml(out_dir, OCR_CLASSES)

    if text_enabled:
        for s in ("train", "val"):
            gt_path = rec_dir / f"{s}.txt"
            with gt_path.open("w", encoding="utf-8") as f:
                for line in rec_lines[s]:
                    f.write(line + "\n")
        # 生成字典（rec 训练需要）
        chars = set()
        for lines in rec_lines.values():
            for ln in lines:
                if "\t" in ln:
                    for ch in ln.rsplit("\t", 1)[1]:
                        chars.add(ch)
        if chars:
            dict_path = rec_dir / "dict.txt"
            dict_path.write_text("\n".join(sorted(chars)) + "\n", encoding="utf-8")
        rec_crops = len(rec_lines["train"]) + len(rec_lines["val"])
    else:
        rec_crops = 0

    return {
        "ok": True,
        "out_dir": str(out_dir),
        "rec_dir": str(rec_dir),
        "train": n_train,
        "val": n_val,
        "class_names": OCR_CLASSES,
        "rec_crops": rec_crops,
        "text_enabled": text_enabled,
    }
