"""
模型管理：列出 models/ 下所有产物，下载，删除。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse, Response

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
