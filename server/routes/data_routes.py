"""
数据管理：上传 zip → 解压 → 抽帧 → 去重 → 列文件。
复用 train-center/ 根目录下的 ingest.py / extract_frames.py / dedup.py。
"""

from __future__ import annotations

import sys
import shutil
import zipfile
import tarfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List

from core.common import ROOT, IMAGE_EXTS, list_images
from services import task_manager

router = APIRouter(prefix="/api/data", tags=["data"])

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "_uploads"
# 标注页面需要直接能选到"系统内置数据集"：obj 用原始大图，ocr 用裁剪后的 det_text/images
BUILTIN_DATASETS = [
    # (显示 name, 实际路径, desc, kind)  kind: obj=Obj 标注(原始大图) / ocr=OCR 标注(裁剪图)
    ("data/det_fabric", DATA_DIR / "det_fabric", "Obj 标注原始大图，裁剪→det_text/images", "obj"),
    ("data/det_text/images", DATA_DIR / "det_text" / "images", "OCR 标注裁剪图（obj 导出产物）", "ocr"),
]

# _uploads 两分类子文件夹：fabric(数据管理上传→obj标注) / ocrtext(obj完成→ocr标注)
_UPLOAD_GROUPS = [
    ("fabric", "obj", "Obj 标注源（数据管理上传）"),
    ("ocrtext", "ocr", "OCR 标注源（obj 完成产物，裁剪图）"),
]


def _migrate_uploads():
    """把 _uploads 整理成分类子文件夹，并把旧版直接放在 _uploads 根下的目录迁移到 fabric/ 下。幂等。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for sub, _k, _d in _UPLOAD_GROUPS:
        (RAW_DIR / sub).mkdir(exist_ok=True)
    for d in list(RAW_DIR.iterdir()):
        if not d.is_dir() or d.name in ("fabric", "ocrtext"):
            continue
        tgt = RAW_DIR / "fabric" / d.name
        if not tgt.exists():
            shutil.move(str(d), str(tgt))
        else:
            for c in list(d.iterdir()):
                tc = tgt / c.name
                if not tc.exists():
                    shutil.move(str(c), str(tgt))
            try:
                d.rmdir()
            except OSError:
                pass


def _safe_name(name: str) -> str:
    """避免目录穿越。"""
    p = Path(name).name  # 只取最后一节文件名
    return p or "upload"


def _resolve_dataset(name: str) -> Path:
    """
    解析 dataset 字符串 → 绝对路径。
    - 包含 '/' 或 '\\'：视为相对 ROOT 的路径（或绝对路径），但必须在 ROOT 下。
    - 否则：视为 data/_uploads 下的子目录。
    """
    raw_name = name.strip()
    if not raw_name:
        raise HTTPException(400, "dataset 为空")
    if "/" in raw_name or "\\" in raw_name or Path(raw_name).is_absolute():
        p = Path(raw_name)
        if not p.is_absolute():
            p = ROOT / p
    else:
        p = RAW_DIR / raw_name
        if not p.is_dir():
            p2 = RAW_DIR / "fabric" / raw_name
            if p2.is_dir():
                p = p2
    p = p.resolve()
    root = ROOT.resolve()
    if root != p and root not in p.parents:
        raise HTTPException(403, f"dataset 不在项目根目录下: {name}")
    return p


@router.get("/")
def list_datasets():
    """列出所有数据集：系统内置（det_fabric / det_text/images）+ _uploads 分类（fabric/ocrtext）。
    每个数据集带 kind：obj=Obj 标注用（原始大图），ocr=OCR 标注用（裁剪图），前端按 mode 过滤。"""
    _migrate_uploads()
    datasets = []
    # 1. 内置数据集（真实存在目录才列）
    for name, path, desc, kind in BUILTIN_DATASETS:
        if path.is_dir():
            imgs = list_images(path)
            datasets.append({
                "name": name,
                "created": datetime.fromtimestamp(path.stat().st_ctime).isoformat(timespec="seconds"),
                "images": len(imgs),
                "builtin": True,
                "desc": desc,
                "kind": kind,
                "finished": (path / ".finished").is_file(),
            })
    # 2. _uploads 三分类子文件夹
    for sub, kind, desc in _UPLOAD_GROUPS:
        base = RAW_DIR / sub
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            imgs = list_images(d)
            datasets.append({
                "name": f"data/_uploads/{sub}/{d.name}",
                "created": datetime.fromtimestamp(d.stat().st_ctime).isoformat(timespec="seconds"),
                "images": len(imgs),
                "builtin": False,
                "desc": desc,
                "kind": kind,
                "finished": (d / ".finished").is_file(),
            })
    return JSONResponse({"datasets": datasets})


@router.get("/images")
@router.get("/{dataset}/images")
def list_dataset_images(
    dataset: str | None = None,
    ds: str | None = Query(None, alias="dataset"),
    page: int = Query(1, ge=1),
    size: int = Query(200, ge=1, le=20000),
):
    """分页列出数据集下的图片。dataset 可放 URL path（简单名）或 query（含斜杠路径）。"""
    name = dataset or ds
    if not name:
        raise HTTPException(400, "缺少 dataset 参数")
    base = _resolve_dataset(name)
    if not base.is_dir():
        raise HTTPException(404, f"数据集不存在: {dataset}")
    imgs = list_images(base)
    start = (page - 1) * size
    end = start + size
    items = []
    for p in imgs[start:end]:
        rel = p.relative_to(ROOT).as_posix()
        # label txt: 同级 dir 下，同名 .txt；或 data/{det_xxx}/labels 下按 split 存放（我们标注页直接写同级）
        cand = [p.with_suffix(".txt")]
        # 兼容 data/det_fabric/labels/xxx.txt 的情况
        for parent in ["images", "images/train", "images/val"]:
            if str(p.parent.as_posix()).endswith(parent):
                lab = p.parent.parent / "labels" / (p.stem + ".txt")
                cand.append(lab)
        annotated = any(c.is_file() and c.read_text(encoding="utf-8", errors="ignore").strip() != "" for c in cand)
        items.append({
            "path": str(p),
            "rel": rel,
            "name": p.name,
            "annotated": bool(annotated),
        })
    return JSONResponse({
        "dataset": dataset,
        "total": len(imgs),
        "page": page,
        "size": size,
        "items": items,
    })


@router.post("/upload")
def upload_dataset(file: UploadFile = File(...)):
    """上传 zip/rar/7z/tar 等 → 解压到 data/_uploads/fabric/<名称>/ → 返回 dataset 名。
    rar/7z 不在此自动解压，需要用户手动放。MVP 先支持 zip + tar(.gz)。"""
    _migrate_uploads()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = _safe_name(file.filename or "data")
    stem = Path(safe).stem or "dataset"
    name = f"{stem}_{ts}"

    saved_path = RAW_DIR / f"{name}.tmp"
    try:
        with saved_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:  # noqa
        if saved_path.is_file():
            saved_path.unlink()
        raise HTTPException(500, f"保存上传失败: {e}")

    out_dir = RAW_DIR / "fabric" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(safe).suffix.lower()
    try:
        if suffix in {".zip"}:
            with zipfile.ZipFile(saved_path) as zf:
                for info in zf.infolist():
                    target = (out_dir / info.filename).resolve()
                    if out_dir.resolve() not in target.parents and target != out_dir.resolve():
                        continue
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(info) as sf, open(target, "wb") as df:
                            shutil.copyfileobj(sf, df)
        elif suffix in {".tar", ".gz", ".tgz", ".bz2", ".xz"}:
            with tarfile.open(saved_path, mode="r:*") as tf:
                for mem in tf.getmembers():
                    target = (out_dir / mem.name).resolve()
                    if out_dir.resolve() not in target.parents and target != out_dir.resolve():
                        continue
                    if mem.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with tf.extractfile(mem) as sf, open(target, "wb") as df:
                            if sf:
                                shutil.copyfileobj(sf, df)
        else:
            out = out_dir / safe
            shutil.move(str(saved_path), out)
            saved_path = out
    except Exception as e:  # noqa
        raise HTTPException(500, f"解压失败: {e}")
    finally:
        if saved_path.is_file() and saved_path.suffix in {".tmp"}:
            try:
                saved_path.unlink()
            except OSError:
                pass

    n_images = len(list_images(out_dir))
    return JSONResponse({
        "dataset": f"data/_uploads/fabric/{name}",
        "dir": str(out_dir),
        "images": n_images,
        "msg": "上传解压完成。如需抽帧/去重，接着调用 extract 或 dedup 接口。",
    })


@router.post("/upload_folder")
def upload_folder(files: List[UploadFile] = File(...), root: str = Query("", description="文件夹名（可选，用作数据集命名）")):
    """
    前端 <input webkitdirectory> 选完文件夹后，会把里面所有文件（含子目录）批量发给后端。
    每个 file.filename 是原始文件名（不含路径），需要前端在 FormData 里用额外字段把相对路径传过来：
      - FormData key 约定："relpaths"（重复多次，按 files 顺序一一对应）或 "file:<i>"。
    为兼容 el-upload 只能带简单字段，我们这里同时支持：
      A) 若 FormData 里有 relpaths=xxx&relpaths=yyy...（Query param 传不了 list，前端放到 body 字段）
         → 兼容方案：前端请求用 multipart/form-data 额外 append 一个 JSON 字符串字段 "__meta__"：
             {"relpaths":["style1/a.jpg","style2/b.png",...], "folderName":"我的款式文件夹"}
    """
    import json as _json
    _migrate_uploads()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = _safe_name(root or "upload")
    ds_name = f"{folder_name}_{ts}"
    out_dir = RAW_DIR / "fabric" / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)

    relpaths: List[str] = []
    meta_bytes = bytearray()
    # 从 files 里挑 "__meta__.json" 作为 meta（前端同名 file 空 bytes 写 JSON 名）
    # 同时保留其他文件
    real_files: List[UploadFile] = []
    for f in files:
        fname = (f.filename or "").strip()
        if fname == "__meta__.json":
            meta_bytes.extend(f.file.read())
        else:
            real_files.append(f)

    if meta_bytes:
        try:
            meta = _json.loads(bytes(meta_bytes).decode("utf-8"))
            relpaths = list(meta.get("relpaths") or [])
            if meta.get("folderName"):
                # 已生成 ds_name，只在 root 为空时替换 folder_name 段
                if not root:
                    new_ds = f"{_safe_name(str(meta['folderName']))}_{ts}"
                    new_dir = RAW_DIR / "fabric" / new_ds
                    if out_dir != new_dir:
                        new_dir.mkdir(parents=True, exist_ok=True)
                        shutil.rmtree(out_dir)
                        out_dir = new_dir
                        ds_name = new_ds
        except Exception as e:  # noqa
            raise HTTPException(400, f"meta 解析失败: {e}")

    if relpaths and len(relpaths) != len(real_files):
        raise HTTPException(400, f"relpaths 长度 {len(relpaths)} 与 files 长度 {len(real_files)} 不一致")

    n_saved = 0
    for idx, f in enumerate(real_files):
        if idx < len(relpaths):
            rel = relpaths[idx]
            # 去掉最顶层的根文件夹（webkitRelativePath 格式：<rootFolder>/a/b/x.jpg）
            # 但用户要求 "子文件夹里是款式" 保留子目录结构：全部保留相对路径
            # 只是过滤掉 ../ 穿越
            parts = [p for p in Path(rel).parts if p not in ("..", ".")]
            rel_clean = str(Path(*parts)) if parts else Path(f.filename or f"file_{idx}").name
        else:
            rel_clean = _safe_name(f.filename or f"file_{idx}")

        target = (out_dir / rel_clean).resolve()
        if out_dir.resolve() not in target.parents:
            continue  # 路径穿越丢弃
        target.parent.mkdir(parents=True, exist_ok=True)
        ext = target.suffix.lower().lstrip(".")
        is_image = ext in IMAGE_EXTS
        try:
            with target.open("wb") as wf:
                shutil.copyfileobj(f.file, wf)
            if is_image:
                n_saved += 1
        except Exception:  # noqa
            pass

    # 如果用户想把所有图片平铺在一个临时数据集里（"所有图片拿出来，放到一个临时的数据集里"），
    # 同时保留结构化目录：建一个 flat 子目录？还是直接结构化目录即"数据集目录"，list_images 会递归找到所有图片。
    # 按 user 原话：把图片拿出来平铺 —— 建 flat_images/ 同步一份，便于标注页直接选（list_images 递归，已经全覆盖，无需平铺）
    # 因此直接返回结构化的 out_dir 作为数据集目录即可。

    n_images = len(list_images(out_dir))
    return JSONResponse({
        "dataset": f"data/_uploads/fabric/{ds_name}",
        "dir": str(out_dir),
        "images": n_images,
        "files_saved": n_saved,
        "msg": f"文件夹上传完成：{n_images} 张图片（{len(real_files)} 个文件）",
    })


@router.get("/extract")
@router.post("/extract")
@router.get("/{dataset}/extract")
@router.post("/{dataset}/extract")
def extract_frames(
    dataset: str | None = None,
    ds: str | None = Query(None, alias="dataset"),
    interval: int = Query(30, ge=1, description="每 N 帧抽 1 张"),
):
    """对数据集中的视频抽帧（复用 extract_frames.py）。非视频文件直接保留。"""
    name = dataset or ds
    if not name:
        raise HTTPException(400, "缺少 dataset 参数")
    src = _resolve_dataset(name)
    if not src.is_dir():
        raise HTTPException(404, f"数据集不存在: {name}")
    py = ROOT / "extract_frames.py"
    if not py.is_file():
        raise HTTPException(500, f"脚本不存在: {py}")
    cmd = [sys.executable, str(py), "--input", str(src), "--interval", str(interval)]
    tid = task_manager.new_task(f"extract:{name}", cmd=cmd, cwd=str(ROOT))
    return JSONResponse({"task_id": tid, "dataset": name})


@router.get("/dedup")
@router.post("/dedup")
@router.get("/{dataset}/dedup")
@router.post("/{dataset}/dedup")
def dedup_images(
    dataset: str | None = None,
    ds: str | None = Query(None, alias="dataset"),
    threshold: int = Query(5, ge=0, le=64, description="dHash 阈值，0最严"),
):
    """对数据集图片去重（复用 dedup.py）。"""
    name = dataset or ds
    if not name:
        raise HTTPException(400, "缺少 dataset 参数")
    src = _resolve_dataset(name)
    if not src.is_dir():
        raise HTTPException(404, f"数据集不存在: {name}")
    py = ROOT / "dedup.py"
    if not py.is_file():
        raise HTTPException(500, f"脚本不存在: {py}")
    cmd = [sys.executable, str(py), "--input", str(src), "--threshold", str(threshold)]
    tid = task_manager.new_task(f"dedup:{name}", cmd=cmd, cwd=str(ROOT))
    return JSONResponse({"task_id": tid, "dataset": name})
