"""
train-center 超算中心 Web 入口。

本地开发:
    pip install fastapi "uvicorn[standard]" python-multipart pyyaml
    python -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload

前端:
    http://localhost:8000/        (static/index.html，单文件 Vue3 CDN)
    http://localhost:8000/docs    (Swagger UI)
    http://localhost:8000/api/... (后端 API)

注意：torch/paddle/ultralytics 等重依赖不在此处 import，由训练脚本自己懒加载。
"""

from __future__ import annotations

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response

# 确保能 import train-center 根目录的脚本
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "server") not in sys.path:
    sys.path.insert(0, str(ROOT / "server"))

from core.common import ROOT as TC_ROOT  # noqa: E402

STATIC_DIR = ROOT / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时确保目录结构存在
    (TC_ROOT / "data").mkdir(parents=True, exist_ok=True)
    (TC_ROOT / "data" / "_uploads").mkdir(parents=True, exist_ok=True)
    for sub in ("fabric", "ocrtext"):
        (TC_ROOT / "data" / "_uploads" / sub).mkdir(parents=True, exist_ok=True)
    (TC_ROOT / "models" / "det").mkdir(parents=True, exist_ok=True)
    (TC_ROOT / "models" / "ocr").mkdir(parents=True, exist_ok=True)
    (TC_ROOT / "runs").mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Train Center", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 路由注册（延迟 import，避免服务启动时加载过重依赖） ----
from routes.data_routes import router as data_router  # noqa: E402
from routes.label_routes import router as label_router  # noqa: E402
from routes.prelabel_routes import router as prelabel_router  # noqa: E402
from routes.train_routes import router as train_router  # noqa: E402
from routes.model_routes import router as model_router  # noqa: E402

app.include_router(data_router)
app.include_router(label_router)
app.include_router(prelabel_router)
app.include_router(train_router)
app.include_router(model_router)


# ---- 文件服务 ----
@app.get("/files/{fpath:path}")
def serve_file(fpath: str):
    """
    服务 train-center 下的任意静态文件。
    用于:
      - /files/data/_uploads/xxx/xxx.jpg（标注预览图）
      - /files/data/rec_text/crop_img/...（rec 裁剪小图）
      - /files/models/...（模型静态列表等场景）
    禁止 .. 穿越。
    """
    if ".." in Path(fpath).parts:
        raise HTTPException(403, "路径非法")
    p = TC_ROOT / fpath
    if not p.is_file():
        raise HTTPException(404, f"文件不存在: {fpath}")
    return FileResponse(p)


@app.get("/health")
def health():
    return {"ok": True, "root": str(TC_ROOT), "static": str(STATIC_DIR)}


@app.get("/")
def index():
    if (STATIC_DIR / "index.html").is_file():
        return FileResponse(STATIC_DIR / "index.html")
    # 前端没生成也给个引导页
    return Response(
        content="<h1>Train Center</h1><p>前端未构建，请访问 <a href='/docs'>/docs</a></p>",
        media_type="text/html",
    )


# 兜底：/index.html 直接指向单文件前端
@app.get("/index.html")
def index_html():
    if (STATIC_DIR / "index.html").is_file():
        return FileResponse(STATIC_DIR / "index.html")
    return RedirectResponse("/docs")
