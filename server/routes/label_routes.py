"""
标注：读取 YOLO txt、保存 YOLO txt、导出训练集。
- obj 模式 → export.export_obj → det_fabric + det_text/images
- ocr 模式 → export.export_ocr → det_text + rec_text
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal

from core.common import (
    ROOT,
    list_images,
    OBJ_CLASSES,
    OCR_CLASSES,
    cv_imread,
)
from core import export as _export

router = APIRouter(prefix="/api/label", tags=["label"])


# --- YOLO 坐标格式 ---
# 矩形: {type:"rect", cls:"fabric", x1,y1,x2,y2: 像素}
# 旋转框: {type:"rotate", cls:"text_h", points:[[x,y]×4]}
# ocr 模式额外带 text

class Box(BaseModel):
    type: Literal["rect", "rotate"] = "rect"
    cls: str
    x1: int | None = None
    y1: int | None = None
    x2: int | None = None
    y2: int | None = None
    points: list[list[float]] | None = Field(default_factory=list)
    text: str = ""


class SavePayload(BaseModel):
    image_path: str
    boxes: list[Box] = []


def _to_yolo_lines(img_w: int, img_h: int, boxes: list[Box], class_names: list[str], with_text: bool = True) -> list[str]:
    cls_map = {n: i for i, n in enumerate(class_names)}
    lines = []
    for b in boxes:
        cid = cls_map.get(b.cls)
        if cid is None:
            continue
        parts = [str(cid)]
        if b.type == "rect" and all(v is not None for v in (b.x1, b.y1, b.x2, b.y2)):
            x1, y1, x2, y2 = map(int, (b.x1, b.y1, b.x2, b.y2))  # type: ignore[arg-type]
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            parts.extend(f"{v:.6f}" for v in (cx, cy, bw, bh))
        elif b.type == "rotate" and b.points and len(b.points) >= 3:
            for px, py in b.points:
                parts.append(f"{px / img_w:.6f}")
                parts.append(f"{py / img_h:.6f}")
        else:
            continue
        line = " ".join(parts)
        if with_text and b.text:
            line += f"\t{b.text}"
        lines.append(line)
    return lines


def _from_yolo_lines(img_w: int, img_h: int, txt_path: Path, class_names: list[str]) -> list[dict]:
    if not txt_path.is_file():
        return []
    cls_map = {i: n for i, n in enumerate(class_names)}
    result = []
    for raw in txt_path.read_text(encoding="utf-8").strip().splitlines():
        if "\t" in raw:
            coord_str, text = raw.split("\t", 1)
        else:
            coord_str, text = raw, ""
        parts = coord_str.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        cls = cls_map.get(cid, class_names[0])
        if len(parts) == 5:
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            result.append({"type": "rect", "cls": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "text": text})
        elif len(parts) >= 7 and len(parts) % 2 == 1:
            n_pairs = (len(parts) - 1) // 2
            pts = []
            for i in range(n_pairs):
                px = float(parts[1 + i * 2]) * img_w
                py = float(parts[2 + i * 2]) * img_h
                pts.append([round(px, 2), round(py, 2)])
            tag = "rotate" if len(pts) == 4 else "rotate"
            result.append({"type": tag, "cls": cls, "points": pts, "text": text})
    return result


# ---- 路由 ----

@router.get("/classes/{mode}")
def get_classes(mode: str):
    if mode == "obj":
        return {"mode": mode, "classes": OBJ_CLASSES}
    if mode == "ocr":
        return {"mode": mode, "classes": OCR_CLASSES}
    raise HTTPException(400, f"未知 mode: {mode}")


@router.get("/load")
def load_boxes(image_path: str, mode: str = Query("obj")):
    img_p = Path(image_path)
    if not img_p.is_file():
        raise HTTPException(404, "图片不存在")
    txt_p = img_p.with_suffix(".txt")
    img = cv_imread(img_p)
    if img is None:
        raise HTTPException(400, "无法读取图片")
    h, w = img.shape[:2]
    class_names = OBJ_CLASSES if mode == "obj" else OCR_CLASSES
    boxes = _from_yolo_lines(w, h, txt_p, class_names)
    return JSONResponse({
        "image_path": str(img_p),
        "image_w": w,
        "image_h": h,
        "boxes": boxes,
    })


@router.post("/save")
def save_boxes(payload: SavePayload, mode: str = Query("obj")):
    img_p = Path(payload.image_path)
    if not img_p.is_file():
        raise HTTPException(404, "图片不存在")
    img = cv_imread(img_p)
    if img is None:
        raise HTTPException(400, "无法读取图片")
    h, w = img.shape[:2]
    class_names = OBJ_CLASSES if mode == "obj" else OCR_CLASSES
    with_text = (mode == "ocr")
    lines = _to_yolo_lines(w, h, payload.boxes, class_names, with_text=with_text)
    txt_p = img_p.with_suffix(".txt")
    if lines:
        txt_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif txt_p.is_file():
        # 没标注就不留空 txt：标注结果只在有框时才存在
        try:
            txt_p.unlink()
        except OSError:
            pass
    return JSONResponse({"ok": True, "txt": str(txt_p), "n_boxes": len(lines)})


@router.get("/export/{mode}")
@router.post("/export/{mode}")
def export_dataset(
    mode: str,
    dataset: str = Query(..., description="数据集名（data/_uploads 下）或 目录绝对路径"),
    ratio: float = Query(0.8, ge=0.1, le=0.95),
    seed: int = Query(42),
    text_enabled: bool = Query(True, description="ocr 模式下是否生成 rec 数据"),
):
    # 确定图片根（解析规则与 data_routes 一致）
    name = dataset.strip()
    if "/" in name or "\\" in name or Path(name).is_absolute():
        p = Path(name)
        if not p.is_absolute():
            p = ROOT / p
    else:
        p = ROOT / "data" / "_uploads" / name
    p = p.resolve()
    root = ROOT.resolve()
    if root != p and root not in p.parents:
        raise HTTPException(403, f"dataset 不在项目根目录下: {dataset}")
    if not p.is_dir():
        raise HTTPException(404, f"找不到图片目录: {dataset}")
    imgs = list_images(p)
    if not imgs:
        raise HTTPException(400, "目录下没有图片")

    if mode == "obj":
        result = _export.export_obj(imgs, ratio=ratio, seed=seed)
    elif mode == "ocr":
        result = _export.export_ocr(imgs, ratio=ratio, seed=seed, text_enabled=text_enabled)
    else:
        raise HTTPException(400, f"未知 mode: {mode}")
    if not result.get("ok"):
        raise HTTPException(400, result.get("msg", "导出失败"))
    return JSONResponse(result)


def _resolve_dataset_dir(dataset: str) -> Path:
    """与 export_dataset 相同的路径解析规则。"""
    name = dataset.strip()
    if "/" in name or "\\" in name or Path(name).is_absolute():
        p = Path(name)
        if not p.is_absolute():
            p = ROOT / p
    else:
        p = ROOT / "data" / "_uploads" / name
    p = p.resolve()
    root = ROOT.resolve()
    if root != p and root not in p.parents:
        raise HTTPException(403, f"dataset 不在项目根目录下: {dataset}")
    if not p.is_dir():
        raise HTTPException(404, f"找不到图片目录: {dataset}")
    return p


@router.get("/finish")
@router.post("/finish")
def finish_dataset(
    mode: str,
    dataset: str = Query(..., description="数据集名（data/_uploads/fabric|ocrtext 下）或 目录绝对路径"),
    ratio: float = Query(0.8, ge=0.1, le=0.95),
    seed: int = Query(42),
    text_enabled: bool = Query(True, description="ocr 模式下是否生成 rec 数据"),
):
    """
    标注「完成」：当前数据集标注收尾。
    - obj 模式：未标注图（txt 空/缺失，视为质量差跳过）的 txt 删除 → 生成 data/det_fabric(train/val)
               同时把裁剪目标写入 _uploads/ocrtext/<数据集名>，供 OCR 标注页选择。
    - ocr 模式：同样跳过未标注图 → 生成 data/det_text + data/rec_text（text_enabled 时）。
    """
    p = _resolve_dataset_dir(dataset)
    imgs = list_images(p)
    if not imgs:
        raise HTTPException(400, "目录下没有图片")

    # 1) 区分已标注 / 未标注（txt 存在且非空）。未标注的图（质量差跳过）连同 txt 一起删除，不进入下一阶段。
    annotated, skipped = [], []
    for img in imgs:
        txt = img.with_suffix(".txt")
        if txt.is_file() and txt.read_text(encoding="utf-8").strip():
            annotated.append(img)
        else:
            skipped.append(img)
            try:
                img.unlink()
            except OSError:
                pass
            if txt.is_file():
                try:
                    txt.unlink()
                except OSError:
                    pass
    if not annotated:
        raise HTTPException(400, "没有任何已标注的图片，无法完成标注（请先标注至少一张）")

    if mode == "obj":
        # 2a) obj 完成：
        #   - 目标按 fabric 框裁剪 → _uploads/ocrtext/<数据集名>（OCR 标注数据源）
        #   - 标注数据本身追加进 det_fabric（train/val 划分，不删旧数据，可跨数据集累积）
        ocr_dir = ROOT / "data" / "_uploads" / "ocrtext" / p.name
        if ocr_dir.exists():
            shutil.rmtree(ocr_dir, ignore_errors=True)
        ocr_dir.mkdir(parents=True, exist_ok=True)
        det_fabric_dir = ROOT / "data" / "det_fabric"
        result = _export.export_obj(annotated, ratio=ratio, seed=seed,
                                    out_dir=det_fabric_dir, det_text_img_dir=ocr_dir)
        if not result.get("ok"):
            raise HTTPException(400, result.get("msg", "导出失败"))
        result["ocr_dir"] = str(ocr_dir)
        result["ocr_images"] = len(list_images(ocr_dir)) if ocr_dir.is_dir() else 0
    elif mode == "ocr":
        # 2b) ocr 完成：
        #   - 文字框标注数据追加进 det_text（train/val 划分，不删旧数据）
        #   - 按文字框裁剪小图追加进 rec_text（crop_img + train.txt/val.txt 追加，dict.txt 重建为并集）
        det_text_dir = ROOT / "data" / "det_text"
        rec_dir = ROOT / "data" / "rec_text"
        train_idx, val_idx = _export._split_indices(len(annotated), ratio, seed)
        for s in ("train", "val"):
            (det_text_dir / "images" / s).mkdir(parents=True, exist_ok=True)
            (det_text_dir / "labels" / s).mkdir(parents=True, exist_ok=True)
            (rec_dir / "crop_img" / s).mkdir(parents=True, exist_ok=True)
        n_train = n_val = 0
        for split, idxs in (("train", train_idx), ("val", val_idx)):
            for i in idxs:
                img = annotated[i]
                txt = img.with_suffix(".txt")
                shutil.copy2(img, det_text_dir / "images" / split / img.name)
                _img = cv_imread(img)
                _shape = _img.shape[:2] if _img is not None else None
                _export._strip_text(txt, det_text_dir / "labels" / split / f"{img.stem}.txt", _shape)
                if split == "train":
                    n_train += 1
                else:
                    n_val += 1
        _export._write_yaml(det_text_dir, OCR_CLASSES)
        # rec_text：本次裁剪的图片追加；train.txt/val.txt 行追加；dict.txt 全量重建
        n_rec = 0
        if text_enabled:
            for split, idxs in (("train", train_idx), ("val", val_idx)):
                lines = []
                for i in idxs:
                    img = annotated[i]
                    lines.extend(_export._crop_text(img, img.with_suffix(".txt"), rec_dir / "crop_img" / split, split))
                n_rec += len(lines)
                if lines:
                    gt = rec_dir / f"{split}.txt"
                    existing = gt.read_text(encoding="utf-8").strip().splitlines() if gt.is_file() else []
                    gt.write_text("\n".join(existing + lines) + "\n", encoding="utf-8")
            chars = set()
            for s in ("train", "val"):
                gt = rec_dir / f"{s}.txt"
                if gt.is_file():
                    for ln in gt.read_text(encoding="utf-8").strip().splitlines():
                        if "\t" in ln:
                            chars.update(ln.rsplit("\t", 1)[1])
            if chars:
                (rec_dir / "dict.txt").write_text("\n".join(sorted(chars)) + "\n", encoding="utf-8")
        result = {"ok": True, "det_text_dir": str(det_text_dir), "rec_dir": str(rec_dir),
                  "train": n_train, "val": n_val, "rec_crops": n_rec}
    else:
        raise HTTPException(400, f"未知 mode: {mode}")

    result.update({
        "mode": mode,
        "total": len(imgs),
        "annotated": len(annotated),
        "skipped": len(skipped),
        "skipped_images": [Path(x).name for x in skipped],
    })
    # 完成标记：标注页"完成"按钮据此显示已完成（绿）。再次完成会覆盖。
    try:
        (p / ".finished").write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
    except OSError:
        pass
    return JSONResponse(result)
