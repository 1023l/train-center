"""
模型管理：列出 models/ 下所有产物，下载，删除。

rec 导出在本侧完成（paddle-ocr 环境）：一键导出与单个下载均直接产出
rec{ver}_onnx.onnx（缓存复用），算法侧不再需要 paddle 环境。
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import time
import zipfile
from collections import deque
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse, Response, StreamingResponse

from core.common import ROOT
from .train_routes import PADDLE_PY_EXE

router = APIRouter(prefix="/api/models", tags=["models"])

DET_DIR = ROOT / "models" / "det"
OCR_DIR = ROOT / "models" / "ocr"
EXPORT_SCRIPT = ROOT / "export_rec_onnx.py"


def _ensure_rec_onnx(rec_dir: Path) -> Path:
    """确保 rec 模型目录对应 ONNX 存在（export_rec_onnx.py 内有缓存，命中则秒回）。

    调 paddle-ocr 环境子进程执行导出，首次约 1 分钟；失败抛 500 带日志尾部。
    """
    out = rec_dir.with_name(f"{rec_dir.name}_onnx.onnx")
    pdparams = rec_dir / "best.pdparams"
    if out.is_file() and out.stat().st_size > 0 \
            and out.stat().st_mtime >= pdparams.stat().st_mtime:
        return out
    cmd = [PADDLE_PY_EXE, str(EXPORT_SCRIPT), "--src", str(rec_dir)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT), timeout=600)
    for line in (r.stdout or "").splitlines()[-5:] + (r.stderr or "").splitlines()[-5:]:
        sys.stderr.write(f"[rec_export] {line}\n")
    if r.returncode != 0:
        tail = "\n".join(((r.stderr or "") + (r.stdout or "")).splitlines()[-8:])
        raise HTTPException(500, f"rec ONNX 导出失败（{rec_dir.name}），日志尾部:\n{tail}")
    if not out.is_file() or out.stat().st_size == 0:
        raise HTTPException(500, f"rec ONNX 导出异常：产物缺失或 0 字节: {out}")
    return out


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
    # rec 模型目录：导出 ONNX（缓存复用）后打包为 zip（算法侧直接可转 engine）
    onnx = _ensure_rec_onnx(target)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.write(onnx, onnx.name)
    buf.seek(0)
    headers = {"Content-Disposition": f"attachment; filename=\"{name}.zip\""}
    return Response(content=buf.read(), media_type="application/zip", headers=headers)


@router.get("/export_bundle")
def export_bundle():
    """一键导出 3 个最新模型打包为 zip（rec 已在本侧导出为 ONNX）。

    流式打包：响应头立即返回（浏览器马上弹出下载提示），边打包边传输。
    rec ONNX 首次导出约 1 分钟（之后缓存秒回），发生在响应前。

    包内结构（fabric-algo 上传接口约定的 manifest.json 格式）：
        manifest.json
        det/fabric{ver}.pt
        det/text{ver}.pt
        ocr/rec{ver}_onnx.onnx
    """
    fabric = _latest_det("fabric")
    text = _latest_det("text")
    rec = _latest_rec_dir()
    missing = [n for n, p in (("fabric", fabric), ("text", text), ("rec", rec)) if p is None]
    if missing:
        raise HTTPException(404, f"缺少模型产物: {', '.join(missing)}")

    rec_onnx = _ensure_rec_onnx(rec)   # 可能触发首次导出（~1min），失败抛 500

    def ver_of(name: str) -> str:
        m = re.search(r"(\d{8}V\d+)", name)
        return m.group(1) if m else ""

    manifest = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "models": {
            "fabric": {"version": ver_of(fabric.name), "file": f"det/{fabric.name}"},
            "text": {"version": ver_of(text.name), "file": f"det/{text.name}"},
            "rec": {"version": ver_of(rec.name), "file": f"ocr/{rec_onnx.name}"},
        },
    }

    def gen():
        # 自定义 sink：zipfile 写入的数据流式转发给 HTTP 响应（chunked）
        sink = _StreamingZip()
        # 权重文件压缩收益极低，STORED 打包最快
        zf = zipfile.ZipFile(sink, "w", zipfile.ZIP_STORED)
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        yield from sink.drain()
        zf.write(fabric, f"det/{fabric.name}")
        yield from sink.drain()
        zf.write(text, f"det/{text.name}")
        yield from sink.drain()
        zf.write(rec_onnx, f"ocr/{rec_onnx.name}")
        yield from sink.drain()
        zf.close()  # central directory 在 close 时写入
        yield from sink.drain()

    filename = f"model_bundle_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    return StreamingResponse(gen(), media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{filename}"'})


class _StreamingZip(io.RawIOBase):
    """zipfile 的写入 sink：捕获写入字节，通过 drain() 流式产出。"""

    def __init__(self):
        self._chunks: deque[bytes] = deque()

    def writable(self) -> bool:
        return True

    def write(self, b) -> int:
        self._chunks.append(bytes(b))
        return len(b)

    def drain(self):
        while self._chunks:
            yield self._chunks.popleft()


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
