"""
训练 API：启动后台任务跑 train.py / train_rec.py，懒加载模型依赖（不顶层 import torch/paddle）。
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.common import ROOT
from services import task_manager
router = APIRouter(prefix="/api/train", tags=["train"])

PY_EXE = sys.executable


def _find_paddle_python() -> str:
    """定位 paddle-ocr 环境的 python.exe（rec 训练用）。

    探测顺序（便于移植到其他电脑）：
    1. 环境变量 TRAIN_PADDLE_PY（最优先，换机器时 set 一下即可）
    2. 本机既有路径 C:\\Users\\Administrator\\miniconda3\\envs\\paddle-ocr
    3. 常见 conda 布局自动扫描（如 C:\\miniconda3\\envs\\paddle-ocr 等）
    4. 找不到 → 回退到当前解释器（rec 训练启动时会给清晰报错）
    """
    import os

    # 1) 环境变量显式指定
    env = os.environ.get("TRAIN_PADDLE_PY")
    if env and Path(env).is_file():
        return env

    # 2) 本机已知路径
    for cand in [
        Path(r"C:\Users\Administrator\miniconda3\envs\paddle-ocr\python.exe"),
        Path(r"C:\Users\Administrator\miniconda3\envs\paddle-ocr\pythonw.exe"),
    ]:
        if cand.is_file():
            return str(cand)

    # 3) 自动扫描常见 conda 安装布局
    for base in [Path.home() / "miniconda3", Path.home() / "anaconda3",
                 Path("C:/miniconda3"), Path("C:/anaconda3"), Path("D:/miniconda3")]:
        p = base / "envs" / "paddle-ocr" / "python.exe"
        if p.is_file():
            return str(p)

    # 4) 回退当前解释器
    return PY_EXE


PADDLE_PY_EXE = _find_paddle_python()

# DATASETS 合法键名与 train.py 保持一致：{det_fabric, det_text, rec_text}
# 前端 dataset 可能传 "data/det_text" "data\\det_fabric" "det_text" 等多种形式，这里统一映射
_VALID_DATASET_KEYS = {"det_fabric", "det_text", "rec_text"}

# 训练 name → 辅助判断 det/rec 输出目录：在 cmd 里反推
# rec: 我们会根据 cmd 是否包含 train_rec.py，来决定走 rec 清理分支
# det: cmd 含 train.py（且非 --val-only） → 走 det 清理分支


def _normalize_dataset_key(name: str) -> str:
    """把 data/det_text、data\\det_fabric、det_text 统一映射成合法键名。"""
    if not name:
        raise HTTPException(400, "dataset 不能为空")
    # 去掉 data/ 前缀，去最后一段
    key = name.replace("\\", "/").strip().rstrip("/")
    if "/" in key:
        key = key.rsplit("/", 1)[-1]
    if key not in _VALID_DATASET_KEYS:
        raise HTTPException(400, f"dataset 参数 '{name}' 不合法，期望：{sorted(_VALID_DATASET_KEYS)}")
    return key


class DetParams(BaseModel):
    dataset: str = "det_fabric"          # det_fabric / det_text / souzhidata
    model: str | None = None            # 预训练权重路径
    epochs: int = 50
    batch: int = 8
    imgsz: int = 640
    device: str = "0"
    version: int = 1


class ValParams(BaseModel):
    model: str = ""                  # 已训练检测模型路径（models/det/xxx.pt）
    dataset: str = "det_fabric"      # 验证用数据集（det_fabric / det_text）
    imgsz: int = 640
    device: str = "0"
    workers: int = 2


class RecParams(BaseModel):
    model: str = "mobile"                 # mobile/server（预训练）或已训练 rec 模型名/路径（models/ocr/<name>）
    epochs: int = 50
    batch: int = 8
    rec_data_dir: str | None = None     # 默认 data/rec_text


def _resolve_rec_pretrained(name: str) -> tuple[str, str | None]:
    """
    解析 rec 训练模型选择：
    - mobile/server（或含 PP-OCRv5_mobile/server）→ 下载预训练权重
    - 其他 → 视为已训练 rec 模型：models/ocr/<name>/best.pdparams 或绝对路径
    返回 (model_key, pretrained_path)，两者至少一个有效。
    """
    if not name:
        return "mobile", None
    n = name.strip().lower().replace("-", "_").replace(" ", "")
    if "server" in n:
        return "server", None
    if "mobile" in n:
        return "mobile", None
    # 已训练 rec 模型
    cand = Path(name)
    if not cand.is_absolute():
        cand = ROOT / "models" / "ocr" / name / "best.pdparams"
        if not cand.is_file():
            alt = ROOT / name
            if alt.is_file():
                cand = alt
    if cand.is_file():
        return "", str(cand.resolve())
    raise HTTPException(
        400,
        f"model 参数 '{name}' 无法解析：既不是 mobile/server，也不是已训练 rec 模型"
        f"（期望 models/ocr/<rec 目录>/best.pdparams）",
    )


@router.get("/tasks")
def api_list_tasks():
    return JSONResponse({"tasks": task_manager.list_tasks()})


@router.get("/tasks/{task_id}")
def api_get_task(task_id: str, from_line: int = Query(0, ge=0)):
    t = task_manager.get_task(task_id)
    if not t:
        raise HTTPException(404, "任务不存在")
    return JSONResponse(t.snapshot(from_line))


@router.post("/val")
def api_train_val(p: ValParams):
    """对已有检测模型跑测试集验证（后台任务），输出 P/R/mAP50/mAP50-95，指标写入模型同名的 .metrics.json。"""
    if not p.model:
        raise HTTPException(400, "缺少 model 参数")
    dataset_key = _normalize_dataset_key(p.dataset)
    train_py = ROOT / "train.py"
    if not train_py.is_file():
        raise HTTPException(500, f"脚本不存在: {train_py}")
    m = Path(p.model)
    if not m.is_absolute():
        m = (ROOT / p.model).resolve()
    if not m.is_file():
        raise HTTPException(400, f"模型不存在: {m}")
    cmd = [
        PY_EXE, str(train_py),
        "--val-only",
        "--model", str(m),
        "--dataset", dataset_key,
        "--imgsz", str(p.imgsz),
        "--device", p.device,
        "--workers", str(p.workers),
    ]
    tid = task_manager.new_task(f"val:{m.name}", cmd=cmd, cwd=str(ROOT))
    return JSONResponse({"task_id": tid, "cmd": " ".join(cmd)})


@router.post("/det")
def api_train_det(p: DetParams):
    """启动 YOLO 目标检测训练（后台任务）。"""
    dataset_key = _normalize_dataset_key(p.dataset)
    train_py = ROOT / "train.py"
    if not train_py.is_file():
        raise HTTPException(500, f"脚本不存在: {train_py}")
    cmd = [
        PY_EXE, str(train_py),
        "--dataset", dataset_key,
        "--epochs", str(p.epochs),
        "--batch", str(p.batch),
        "--imgsz", str(p.imgsz),
        "--device", p.device,
        "--version", str(p.version),
    ]
    if p.model:
        # 相对路径相对 ROOT 解析成绝对，避免训练脚本 cwd 不同时找不到
        m = Path(p.model)
        if not m.is_absolute():
            m = (ROOT / p.model).resolve()
        cmd += ["--model", str(m)]
    tid = task_manager.new_task(f"train:det:{p.dataset}", cmd=cmd, cwd=str(ROOT))
    return JSONResponse({"task_id": tid, "cmd": " ".join(cmd)})


@router.post("/rec")
def api_train_rec(p: RecParams):
    """启动 PaddleOCR rec 增量训练（后台任务）。
    注意：依赖 paddle-ocr conda 环境（PaddlePaddle-GPU + PaddleOCR 包）。
    model 支持：mobile/server（自动下载预训练）或已训练 rec 模型（models/ocr/<name>/best.pdparams 继续增量）。
    """
    model_key, pretrained = _resolve_rec_pretrained(p.model)
    train_py = ROOT / "train_rec.py"
    if not train_py.is_file():
        raise HTTPException(500, f"脚本不存在: {train_py}")
    cmd = [PADDLE_PY_EXE, str(train_py)]
    if pretrained:
        cmd += ["--pretrained", pretrained]
    else:
        cmd += ["--model", model_key]
    cmd += ["--epochs", str(p.epochs), "--batch", str(p.batch)]
    if p.rec_data_dir:
        cmd += ["--data-dir", p.rec_data_dir]
    tid = task_manager.new_task(f"train:rec:{p.model}", cmd=cmd, cwd=str(ROOT))
    return JSONResponse({"task_id": tid, "cmd": " ".join(cmd), "python": PADDLE_PY_EXE, "pretrained": pretrained})


# ================= 训练中间产物扫描 / 清理 =================
def _infer_cleanup_kind(tid: str):
    """根据任务信息推断走 rec 还是 det 清理。返回 (kind, meta)，无效任务抛 404。"""
    t = task_manager.get_task(tid)
    if t is None:
        raise HTTPException(404, "任务不存在")
    cmd = ""
    for row in list(t.log):
        if row.startswith("[run] $"):
            cmd = row[len("[run] $"):].strip()
            break
    if not cmd:
        # 没日志也从创建时的 name 猜
        if t.name.startswith("train:rec:"):
            kind = "rec"
        elif t.name.startswith("train:det:"):
            kind = "det"
        else:
            kind = "none"
    elif "train_rec.py" in cmd:
        kind = "rec"
    elif "train.py" in cmd and "--val-only" not in cmd:
        kind = "det"
    else:
        kind = "none"
    return kind, t, cmd


def _scan_for_task(tid: str):
    kind, t, cmd = _infer_cleanup_kind(tid)
    from core.cleanup_artifacts import scan_rec_outputs, scan_det_outputs, CleanupReport
    if kind == "rec":
        # train_rec.py 按 model mobile/server 推断子目录名
        import re
        model_sub = None
        m = re.search(r"--model\s+(mobile|server)", cmd)
        if m:
            model_sub = f"PP-OCRv5_{m.group(1)}_rec"
        r = scan_rec_outputs(model_sub)
    elif kind == "det":
        # det 默认输出是 train-center/runs/<name>，name={fabric/text}{date}V{ver}
        project_dir = str(ROOT / "runs")
        experiment = None
        # 从 --version + --dataset 猜；若还没跑出来先扫所有 runs/* 目录的 weights/
        r = CleanupReport(category="det")
        from core.cleanup_artifacts import scan_det_outputs as _scan_det
        runs_root = Path(project_dir)
        if runs_root.is_dir():
            # 扫描所有 det 训练产物目录（每个 runs/<fabric*|text*>/weights/）
            merged: CleanupReport | None = None
            for sub in sorted(runs_root.iterdir()):
                if not sub.is_dir():
                    continue
                if not sub.name.startswith(("fabric", "text")):
                    continue
                if (sub / "weights").is_dir():
                    one = _scan_det(runs_root, sub.name)
                    if merged is None:
                        merged = one
                    else:
                        merged.scanned_dirs.extend(one.scanned_dirs)
                        merged.candidates.extend(one.candidates)
                        merged.kept.extend(one.kept)
            r = merged if merged is not None else r
    else:
        r = CleanupReport(category="none")
    return kind, t, r


@router.get("/tasks/{task_id}/cleanup_info")
def api_cleanup_info(task_id: str):
    kind, t, report = _scan_for_task(task_id)
    return JSONResponse({
        "ok": True,
        "task_id": task_id,
        "task_status": t.status,
        "kind": kind,
        "report": report.to_dict(),
    })


@router.post("/tasks/{task_id}/cleanup")
def api_cleanup_exec(task_id: str):
    kind, t, report = _scan_for_task(task_id)
    from core.cleanup_artifacts import execute_cleanup
    done = execute_cleanup(report)
    # 同时回写到 Task.meta，方便 snapshot 里也能拿到
    t.meta = t.__dict__.get("meta", {})
    t.meta["cleanup"] = done.to_dict()
    return JSONResponse({
        "ok": True,
        "task_id": task_id,
        "kind": kind,
        "deleted": done.to_dict(),
    })
