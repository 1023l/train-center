"""AI 预标注路由（切入点 3）：VLM 看图出粗框，人工修正。

接口：
  GET  /api/label/prelabel_status   - VLM 配置状态
  POST /api/label/prelabel          - 对一张已上传的数据集图片做预标注

设计：
  - 输入 image_path 为 train-center 数据集内已有图片（obj/ocr 标注页当前图）
  - VLM 输出 0-1000 相对坐标，服务端换算成像素并做类别白名单过滤
  - 返回 boxes 与 /api/label/load 的格式一致（rect + cls + text），前端直接并入画布
  - 预标注结果不直接写盘（由前端随人工修正一起走 /api/label/save 保存），保证人工终审
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.common import ROOT, OBJ_CLASSES, OCR_CLASSES, cv_imread
from services import vlm_client

router = APIRouter(prefix="/api/label", tags=["prelabel"])


class PrelabelBody(BaseModel):
    image_path: str
    mode: str = Field("obj", description="obj=检布片(fabric) / ocr=检文字区域(text_h/text_v)")
    prompt_extra: str = Field("", description="补充说明（如'只找完整布片'）")


def _check_image_path(image_path: str) -> Path:
    p = Path(image_path)
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()
    root = ROOT.resolve()
    if root not in p.parents:
        raise HTTPException(403, f"图片不在项目根目录下: {image_path}")
    if not p.is_file():
        raise HTTPException(404, f"图片不存在: {image_path}")
    return p


@router.get("/prelabel_status")
def prelabel_status():
    return vlm_client.status()


@router.post("/prelabel")
def prelabel(body: PrelabelBody):
    t0 = __import__("time").time()
    img_p = _check_image_path(body.image_path)
    img = cv_imread(img_p)
    if img is None:
        raise HTTPException(400, "无法读取图片")
    H, W = img.shape[:2]

    if body.mode == "obj":
        target = "所有布片/织物主体（一块完整的布，忽略边角碎片）"
        labels_hint = "fabric"
        allowed = set(OBJ_CLASSES)
    elif body.mode == "ocr":
        target = "布面上所有印刷文字区域（如鞋码、货号、L/R 标记等），横排和竖排分开标"
        labels_hint = "text_h（横排文字）或 text_v（竖排文字）"
        allowed = set(OCR_CLASSES)
    else:
        raise HTTPException(400, f"未知 mode: {body.mode}")

    try:
        raw_boxes = vlm_client.detect_boxes(img, target + (f"。{body.prompt_extra}" if body.prompt_extra else ""), labels_hint)
    except RuntimeError as e:  # 未配置 Key
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(502, f"VLM 调用失败：{e}") from e

    boxes = []
    for b in raw_boxes:
        cls = b["label"]
        if cls not in allowed:
            # VLM 偶尔输出别名/变体，尽力归一
            low = cls.lower()
            if "fabric" in low and "fabric" in allowed:
                cls = "fabric"
            elif "text_v" in low and "text_v" in allowed:
                cls = "text_v"
            elif "text" in low and "text_h" in allowed:
                cls = "text_h"
            else:
                continue
        x1, y1, x2, y2 = b["bbox_2d"]
        # prompt 要求 0-1000 相对坐标 → 像素；若 VLM 输出了像素坐标（>1000），按像素处理
        if max(x2, y2) > 1000:
            px1, py1, px2, py2 = x1, y1, x2, y2
        else:
            px1, py1 = x1 / 1000 * W, y1 / 1000 * H
            px2, py2 = x2 / 1000 * W, y2 / 1000 * H
        px1, px2 = sorted((max(0, min(W, px1)), max(0, min(W, px2))))
        py1, py2 = sorted((max(0, min(H, py1)), max(0, min(H, py2))))
        if (px2 - px1) < 4 or (py2 - py1) < 4:
            continue
        boxes.append({
            "type": "rect",
            "cls": cls,
            "x1": int(px1), "y1": int(py1), "x2": int(px2), "y2": int(py2),
            "text": b["text"] if body.mode == "ocr" else "",
        })

    return JSONResponse({
        "ok": True,
        "elapsed_ms": int((__import__("time").time() - t0) * 1000),
        "image_path": str(img_p),
        "image_w": W, "image_h": H,
        "boxes": boxes,
        "raw_count": len(raw_boxes),
    })
