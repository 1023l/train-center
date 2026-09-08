# -*- coding: utf-8 -*-
"""
训练中间产物扫描/清理工具（被 train_routes.py 以及未来 CLI 调用）。

覆盖两类训练：
  · rec 文字识别：
        PaddleOCR/output/<MODEL_NAME>/iter_epoch_*.{pdopt,pdparams,states}
        train-center/runs/rec_output/<MODEL_NAME>/iter_epoch_*.{pdopt,pdparams,states}
    保留：best_accuracy.* / best.* / latest.* / best_model/ / config.yml / *.log / inference.*
  · det 目标检测（YOLO）：
        train-center/runs/<EXPERIMENT>/weights/last*.pt
    保留：best.pt（当前已经复制到 models/det/ 了，runs 里的 best.pt 也保留 1 份以防万一）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PADDLEOCR_REPO = ROOT / "PaddleOCR"

# ================ rec 保留集合 ================
REC_KEEP_STEMS = {"best_accuracy", "best", "latest"}
REC_KEEP_DIR_NAMES = {"best_model", "best_accuracy_model", "latest_model"}
REC_KEEP_SUFFIX_EXACT = {".yml", ".yaml", ".log"}  # 配置/日志整后缀保留

# ================ det 保留集合 ================
DET_KEEP_WEIGHT_NAMES = {"best.pt"}  # 所有 last*.pt 视作可删


@dataclass
class CleanupItem:
    path: str
    size_mb: float

    def to_dict(self):
        return {"path": self.path, "size_mb": round(self.size_mb, 1)}


@dataclass
class CleanupReport:
    category: str                        # "rec" / "det" / "mixed"
    scanned_dirs: list[str] = field(default_factory=list)
    candidates: list[CleanupItem] = field(default_factory=list)  # 被视为中间产物、可以删的
    kept: list[CleanupItem] = field(default_factory=list)        # 保留项（只给前 20 条，避免巨大）

    @property
    def total_mb(self) -> float:
        return sum(c.size_mb for c in self.candidates)

    def to_dict(self):
        return {
            "category": self.category,
            "scanned_dirs": self.scaned_dirs if hasattr(self, "scaned_dirs") else self.scanned_dirs,
            "total_mb": round(self.total_mb, 1),
            "total_gb": round(self.total_mb / 1024, 2),
            "count": len(self.candidates),
            "candidates": [c.to_dict() for c in self.candidates[:50]],
            "kept": [c.to_dict() for c in self.kept[:20]],
        }


def _mb(n: int) -> float:
    return n / 1024 / 1024


def _rec_should_delete(f: Path) -> bool:
    """rec 中间 ckpt 判断：文件名包含 iter_epoch_ 或 epoch_ 且不是 best/latest。"""
    name = f.name
    # 白名单：只要前缀命中 best*/latest* 直接留
    for ks in REC_KEEP_STEMS:
        if name.startswith(ks + ".") or name == ks:
            return False
    if f.suffix.lower() in REC_KEEP_SUFFIX_EXACT:
        return False
    if name.startswith("inference."):
        return False
    # 删除信号：iter_epoch_XXXX / epoch_XXXX（.pdopt/.pdparams/.states）
    if "iter_epoch_" in name or "epoch_" in name:
        return True
    return False


def _scan_rec_dir(target_dir: Path, report: CleanupReport) -> None:
    if not target_dir.is_dir():
        return
    for sub in sorted(target_dir.iterdir()):
        if sub.is_dir():
            if sub.name in REC_KEEP_DIR_NAMES:
                # best_model/ 这种完整推理导出目录，全部保留
                for ff in sub.rglob("*"):
                    if ff.is_file():
                        try:
                            report.kept.append(CleanupItem(str(ff), _mb(ff.stat().st_size)))
                        except OSError:
                            pass
                report.scanned_dirs.append(str(sub))
                continue
            # 其它子目录（一般没有，若有则递归扫描）
            report.scanned_dirs.append(str(sub))
            _scan_rec_dir(sub, report)
        else:
            try:
                sz = sub.stat().st_size
            except OSError:
                continue
            if _rec_should_delete(sub):
                report.candidates.append(CleanupItem(str(sub), _mb(sz)))
            else:
                report.kept.append(CleanupItem(str(sub), _mb(sz)))


def scan_rec_outputs(model_name: str | None = None) -> CleanupReport:
    """
    扫描 rec 中间产物。
    model_name：本次训练的输出目录名（如 PP-OCRv5_server_rec），None 时扫描 output 下所有子目录。
    """
    report = CleanupReport(category="rec")
    candidates = []
    # 路径 1：PaddleOCR 仓库下的 output/
    po_dir = PADDLEOCR_REPO / "output"
    # 路径 2：train-center/runs/rec_output/
    rc_dir = ROOT / "runs" / "rec_output"
    for base in (po_dir, rc_dir):
        if not base.is_dir():
            continue
        if model_name and (base / model_name).is_dir():
            report.scanned_dirs.append(str(base / model_name))
            candidates.append(base / model_name)
        elif model_name is None:
            for sub in sorted(base.iterdir()):
                if sub.is_dir() and sub.name.startswith("PP-"):  # PP-OCRv5_mobile/server_rec
                    report.scanned_dirs.append(str(sub))
                    candidates.append(sub)
    for d in candidates:
        _scan_rec_dir(d, report)
    return report


def scan_det_outputs(project_dir: Path, experiment: str) -> CleanupReport:
    """
    扫描 det 中间产物（YOLO 训练 runs/<name>/weights/last.pt）。
    project_dir：train-center/runs 或自定义 project 路径
    experiment：训练 name（如 fabric20260828V2）
    """
    report = CleanupReport(category="det")
    run_dir = Path(project_dir) / experiment
    weights = run_dir / "weights"
    if not weights.is_dir():
        # 不存在也把路径记到 scanned 里，便于前端回显"无中间产物"
        report.scanned_dirs.append(str(run_dir))
        return report
    report.scanned_dirs.append(str(weights))
    for f in sorted(weights.iterdir()):
        if not f.is_file() or f.suffix != ".pt":
            continue
        try:
            sz = f.stat().st_size
        except OSError:
            continue
        if f.name in DET_KEEP_WEIGHT_NAMES:
            report.kept.append(CleanupItem(str(f), _mb(sz)))
        else:
            report.candidates.append(CleanupItem(str(f), _mb(sz)))
    # 顺手扫一下 run_dir 本身：除了 weights/ 其他都不删（结果图 labels 等小文件可保留）
    return report


def execute_cleanup(report: CleanupReport) -> CleanupReport:
    """按 report.candidates 实际删文件。返回一份新 report（candidates=删除成功的；失败的放 kept）。"""
    deleted: list[CleanupItem] = []
    failed: list[CleanupItem] = []
    for item in report.candidates:
        p = Path(item.path)
        if not p.is_file():
            continue  # 被别的进程先删了也无所谓
        try:
            p.unlink()
            deleted.append(item)
        except OSError as e:
            failed.append(CleanupItem(f"{item.path} [{e}]", item.size_mb))
    new = CleanupReport(
        category=report.category,
        scanned_dirs=list(report.scanned_dirs),
        candidates=deleted,
        kept=report.kept + failed,
    )
    return new
