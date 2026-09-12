"""VLM 客户端（预标注用）：OpenAI 兼容接口，把"看图检测框"封装成函数。

与 fabric-algo/vlm_qc.py 同一套模式，但 prompt 面向检测框输出（grounding）。

环境变量（或 train-center 根 .env）：
  TC_VLM_API_BASE - 默认 https://dashscope.aliyuncs.com/compatible-mode/v1
  TC_VLM_API_KEY  - API Key
  TC_VLM_MODEL    - 默认 qwen-vl-max
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # train-center/

DEFAULT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-vl-max"

_client = None
_client_cfg: tuple[str, str, str] | None = None


def _load_env() -> None:
    """加载 train-center 根 .env（TC_VLM_* 变量；已设置的环境变量优先）。"""
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines() if (ROOT / ".env").exists() else []:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().replace("export ", "")
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env()


def is_configured() -> bool:
    return bool(os.getenv("TC_VLM_API_KEY", "").strip())


def status() -> dict:
    return {
        "configured": is_configured(),
        "api_base": os.getenv("TC_VLM_API_BASE", DEFAULT_BASE),
        "model": os.getenv("TC_VLM_MODEL", DEFAULT_MODEL),
    }


def _get_client():
    global _client, _client_cfg
    base = os.getenv("TC_VLM_API_BASE", DEFAULT_BASE).strip()
    key = os.getenv("TC_VLM_API_KEY", "").strip()
    model = os.getenv("TC_VLM_MODEL", DEFAULT_MODEL).strip()
    cfg = (base, key, model)
    if _client is None or _client_cfg != cfg:
        if not key:
            raise RuntimeError(
                "TC_VLM_API_KEY 未设置。请在 train-center/.env 配置 "
                "TC_VLM_API_BASE / TC_VLM_API_KEY / TC_VLM_MODEL"
            )
        from openai import OpenAI
        _client = OpenAI(api_key=key, base_url=base, timeout=float(os.getenv("TC_VLM_TIMEOUT", "60")))
        _client_cfg = cfg
    return _client, model


def encode_jpg(img: np.ndarray, max_side: int = 1024, quality: int = 88) -> str:
    """ndarray → JPEG data URL（预标注需要更多细节，max_side 给大一点）。"""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _extract_json(text: str):
    """从模型回复提取 JSON（容忍围栏/杂文；支持对象或数组）。"""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", text, flags=re.S)
    if not m:
        m = re.search(r"(\[.*\]|\{.*\})", text, flags=re.S)
    if not m:
        raise ValueError(f"无法从模型输出中提取 JSON: {text[:200]}")
    return json.loads(m.group(1))


def detect_boxes(img: np.ndarray, target_desc: str, labels_hint: str) -> list[dict]:
    """让 VLM 检测图中目标，返回原始框列表。

    target_desc: 检测什么（中文描述）
    labels_hint: label 字段允许取值说明
    返回: [{"label": str, "bbox_2d": [x1,y1,x2,y2] 0-1000 相对坐标, "text": str}]
    """
    client, model = _get_client()
    data_url = encode_jpg(img)
    prompt = (
        f"请检测图片中的{target_desc}。\n"
        "要求：\n"
        "1. 只输出一个 JSON 数组，不要输出任何其他内容，不要用 markdown 代码块\n"
        '2. 每个元素格式：{"label": "...", "bbox_2d": [x1, y1, x2, y2], "text": "..."}\n'
        "3. bbox_2d 是 0-1000 的相对坐标（相对图片宽高的千分比），x1<y1 且 x2>y2\n"
        f"4. label 只允许取：{labels_hint}\n"
        '5. 检测不到任何目标时输出 []\n'
        "6. text 字段：如果目标上有可读文字就填文字内容，没有就填空字符串\n"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content or ""
    obj = _extract_json(raw)
    if isinstance(obj, dict):
        obj = obj.get("boxes") or obj.get("result") or []
    out = []
    for item in obj:
        if not isinstance(item, dict):
            continue
        box = item.get("bbox_2d") or item.get("bbox") or item.get("box")
        if not box or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in box]
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        out.append({
            "label": str(item.get("label", "")).strip(),
            "bbox_2d": [x1, y1, x2, y2],
            "text": str(item.get("text", "") or ""),
        })
    return out
