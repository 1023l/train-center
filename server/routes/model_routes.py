"""
模型管理：列出 models/ 下所有产物，下载，删除。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse, Response
from starlette.background import BackgroundTask

from core.common import ROOT

router = APIRouter(prefix="/api/models", tags=["models"])

DET_DIR = ROOT / "models" / "det"
OCR_DIR = ROOT / "models" / "ocr"


def _scan_dir(p: Path, kind: str):
    out = []
    if not p.is_dir():
        return out
    for item in sorted(p.iterdir()):
        if item.is_file() and item.suffix in {".pt", ".onnx", ".engine", ".pdparams"}:
            metrics = {}
            mj = item.with_suffix(".metrics.json")
            if mj.is_file():
                try:
                    import json
                    metrics = json.loads(mj.read_text(encoding="utf-8")) or {}
                except (OSError, ValueError):
                    metrics = {}
            out.append({
                "kind": kind,
                "name": item.name,
                "path": str(item),
                "size_kb": round(item.stat().st_size / 1024, 1),
                "updated": datetime.fromtimestamp(item.stat().st_mtime).isoformat(timespec="seconds"),
                "metrics": metrics,
            })
        elif item.is_dir():
            # 可能是 inference 目录（rec20260825V1/ 里含 3 个推理文件）
            files = list(item.iterdir())
            total = sum(f.stat().st_size for f in files if f.is_file())
            out.append({
                "kind": kind,
                "name": item.name,
                "path": str(item),
                "size_kb": round(total / 1024, 1),
                "files": len(files),
                "has_pdparams": any(f.suffix == ".pdparams" for f in files),
                "updated": datetime.fromtimestamp(item.stat().st_mtime).isoformat(timespec="seconds"),
            })
    return out


@router.get("/")
def list_models():
    return JSONResponse({
        "det": _scan_dir(DET_DIR, "det"),
        "ocr": _scan_dir(OCR_DIR, "ocr"),
    })


@router.get("/download/{kind}/{name}")
def download_model(kind: str, name: str):
    base = DET_DIR if kind == "det" else OCR_DIR
    target = base / name
    if not target.exists():
        raise HTTPException(404, "不存在")
    if target.is_file():
        return FileResponse(target, filename=name)
    # 目录：打包成 zip 流
    import zipfile
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in target.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(target).as_posix())
    buf.seek(0)
    headers = {"Content-Disposition": f"attachment; filename=\"{name}.zip\""}
    return Response(content=buf.read(), media_type="application/zip", headers=headers)


@router.get("/export_bundle")
def export_bundle():
    """一键导出 3 个最新模型（fabric.pt + text.pt + rec 权重目录）打包为 zip。

    包内结构（fabric-algo 上传接口约定的 manifest.json 格式）：
        manifest.json
        det/fabric{ver}.pt
        det/text{ver}.pt
        ocr/rec{ver}/best.pdparams
    """
    fabric = _latest_det("fabric")
    text = _latest_det("text")
    rec = _latest_rec_dir()
    missing = [n for n, p in (("fabric", fabric), ("text", text), ("rec", rec)) if p is None]
    if missing:
        raise HTTPException(404, f"缺少模型产物: {', '.join(missing)}")

    def ver_of(name: str) -> str:
        m = re.search(r"(\d{8}V\d+)", name)
        return m.group(1) if m else ""

    manifest = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "models": {
            "fabric": {"version": ver_of(fabric.name), "file": f"det/{fabric.name}"},
            "text": {"version": ver_of(text.name), "file": f"det/{text.name}"},
            "rec": {"version": ver_of(rec.name), "dir": f"ocr/{rec.name}"},
        },
    }

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()
    import zipfile
    # 权重文件压缩收益极低，STORED 打包更快
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        zf.write(fabric, f"det/{fabric.name}")
        zf.write(text, f"det/{text.name}")
        for f in sorted(rec.rglob("*")):
            # 导出 ONNX 只需要 best.pdparams；inference.yml 一并带上便于追溯
            if f.is_file() and f.name in {"best.pdparams", "inference.yml"}:
                zf.write(f, f"ocr/{rec.name}/{f.name}")

    filename = f"model_bundle_{time.strftime('%Y%m%d_%H%M')}.zip"
    return FileResponse(tmp.name, filename=filename, media_type="application/zip",
                        background=BackgroundTask(os.unlink, tmp.name))


def _latest_det(prefix: str) -> Path | None:
    """models/det 下 {prefix}YYYYMMDDVn.pt 取文件名最新的一份。"""
    pat = re.compile(rf"^{prefix}\d{{8}}V\d+\.pt$")
    cands = sorted((f for f in DET_DIR.glob(f"{prefix}*.pt") if pat.match(f.name)),
                   key=lambda f: f.name)
    return cands[-1] if cands else None


def _latest_rec_dir() -> Path | None:
    """models/ocr 下 recYYYYMMDDVn/ 目录（须含 best.pdparams）取最新。"""
    pat = re.compile(r"^rec\d{8}V\d+$")
    cands = sorted((d for d in OCR_DIR.iterdir()
                    if d.is_dir() and pat.match(d.name) and (d / "best.pdparams").is_file()),
                   key=lambda d: d.name)
    return cands[-1] if cands else None


@router.delete("/{kind}/{name}")
def delete_model(kind: str, name: str):
    base = DET_DIR if kind == "det" else OCR_DIR
    target = base / name
    if not target.exists():
        raise HTTPException(404, "不存在")
    import shutil
    deleted = []
    if target.is_file():
        target.unlink()
        deleted.append(str(target))
        # 级联清理同主名的附属文件：.metrics.json
        extras = [target.with_suffix(".metrics.json")]
        for e in extras:
            if e.is_file():
                e.unlink()
                deleted.append(str(e))
    else:
        shutil.rmtree(target)
        deleted.append(str(target))
    return JSONResponse({"ok": True, "deleted": deleted})
