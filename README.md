[English](README.md) | [简体中文](README.zh-CN.md)

# train-center · One-Stop Annotation & Training Platform

A **general-purpose "object detection + OCR recognition" annotation & training platform** (fabric→text→rec three-stage pipeline) that packs the complete workflow of "data ingestion, dedup, annotation, format conversion, training, model management" into a single FastAPI + Vue 3 web app. It pairs with its sibling project **`fabric-algo` (online inference / real-time counting software)** in the same parent directory to close the "train → deploy" loop.

> Platform capabilities are industry-agnostic: object detection (YOLO), text-region detection (horizontal/vertical), OCR recognition training (PaddleOCR).
> The current **textile fabric counting / shoe-size recognition** serves as the reference implementation; dataset codes `fabric` / `text` / `rec` are legacy business naming, not industry-bound.

```
projects\
├── train-center\   ← this repo (annotation & training)
└── fabric-algo\    ← sibling project (inference & web real-time video detection)
```

---

## 1. Feature Overview

| Module | Capabilities |
|---|---|
| **Data Management** | Upload zip / folder (subdirectories = styles) → extract → video frame extraction → perceptual-hash dedup → automatic train/val split |
| **Annotation Tool** | Fabric.js + native Canvas with two modes:<br>① **Object detection mode** (object boxes/text regions, rectangle + 4-point rotated boxes; rotated boxes save coordinates and text tab-separated)<br>② **OCR mode** (toggle horizontal `text_h` / vertical `text_v` buttons; vertical boxes are auto-rotated rot90 k=3 and cropped as rec data on export) |
| **Training** | det_fabric / det_text use **YOLO26 (ultralytics)**; rec_text uses **PP-OCRv5 (PaddleOCR)**.<br>Stage order is fixed `det_fabric → det_text → rec_text`, started serially. Real-time logs/polling during training, auto model-list refresh on completion, **plus a popup prompting cleanup of intermediate checkpoints**. |
| **Model Management** | Versioned naming `[dataset]YYYYMMDDV[ver]` (V1=YOLO26n, V2=RetDR; increments on architecture change). Per model: preview, `.metrics.json`, delete (cascade-deletes the attached `.metrics.json`). **One-click export of the 3 latest models as a zip bundle**; the inference side auto-converts and goes live after upload. |
| **Logs** | All training/upload/dedup/export tasks managed centrally: real-time log, status running/success/failed, elapsed. |

---

## 1.1 Not Included in This Repo

- **PaddleOCR source**: place it yourself under `PaddleOCR/` (rec training calls `tools/train.py`).
- **Model weights** (`.pt` / `.pdparams` / inference binaries): see `models/`, not committed.
- **Raw images & annotation bulk files**: images/labels under `data/` are ignored.

---

## 2. Quick Start

```bash
cd train-center
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

> The web backend depends only on `fastapi + uvicorn` (see [requirements-server.txt](./requirements-server.txt)). Actual training spawns subprocesses in dedicated environments:
> - det_fabric / det_text: `yolo-bench` env (ultralytics + GPU)
> - rec_text: `paddle-ocr` env (paddlepaddle-gpu + PaddleOCR source `tools/train.py`)

Open in browser:

| Entry | URL |
|---|---|
| Frontend | http://127.0.0.1:8000/ |
| Health check | http://127.0.0.1:8000/health |
| Swagger docs | http://127.0.0.1:8000/docs |

Stop the service:

```cmd
netstat -ano | findstr :8000     # note the PID
taskkill /F /PID <PID>
```

---

## 3. Workflow (Standard Flow)

```
[1 Upload/Ingest]   POST /api/data/upload (zip)  or  upload_folder (subdirs = styles)
      │
      ▼
[2 Frame Extract]   extract_frames.py  (CLI / data page)
      │
      ▼
[3 Dedup]           dedup.py (pHash threshold 8)
      │
      ▼
[4 Web Annotation]  det_fabric (object detection rectangles)
                 │
                 └──► det_text (text regions, rectangle / 4-point rotated boxes, text_h / text_v classes)
                        │
                        └──► rec_text (OCR mode: write recognized text per box; auto-rotate + crop on export)
      │
      ▼
[5 Export]          Core [core/export.py](./core/export.py) handles YOLO 5-value rectangles / pure-coordinate cleaning / vertical-text rotation+crop
                 → det outputs data/{det_fabric,det_text}/data.yaml + images/train|val + labels/train|val
                 → rec outputs data/rec_text/{crop_img/train|val, dict.txt, train.txt, val.txt}
      │
      ▼
[6 Three-Stage Training · Serial]
    ├─ det_fabric  train.py  (YOLO26)   → models/det/fabricYYYYMMDDVn.pt + .metrics.json
    ├─ det_text    train.py  (YOLO26)   → models/det/textYYYYMMDDVn.pt   + .metrics.json
    └─ rec_text    train_rec.py → subprocess calls PaddleOCR/tools/train.py (PP-OCRv5 SVTR)
                                  → PaddleOCR/output/PP-OCRv5_{mobile,server}_rec/
                                  → export inference model → models/ocr/recYYYYMMDDVn/
                                        (3 files: inference.json / .pdiparams / .yml)
      │
      ▼
[7 Training Done]   • Frontend polling detects running→done and pops up: "Training produced N intermediate
                       checkpoints / X GB — clean up to save space?" (buttons "Delete, save X GB" / "Keep")
                    • Cleanup rules in [core/cleanup_artifacts.py](./core/cleanup_artifacts.py)
                          rec: delete iter_epoch_*.{pdopt,pdparams,states}, keep best*/latest*/best_model/
                          det: scan runs/<fabric*,text*>/weights/, everything except best.pt is deletable
      │
      ▼
[8 Export & Deploy] Handed to sibling project fabric-algo's tools/ (config centralized in tools/config.yaml):
                   export_onnx.py (YOLO end2end NMS opset 17 imgsz 640 / Paddle → ONNX dynamic width)
                   build_engine.py → .engine (TRT 11.2 eval build, FP32)
```

### 3.1 Version Naming (Hard Constraint)

**`[dataset]YYYYMMDDV[version]`**, e.g.:
```
fabric20260828V2    # fabric dataset · 2026-08-28 · architecture V2
text20260831V1      # text   dataset · 2026-08-31 · architecture V1
rec20260828V3       # rec    dataset · 2026-08-28 · architecture V3
```

The version number **only increments on architecture changes (e.g. YOLO26n→RetDR, PP-OCRv5 mobile→server)**; retraining the same architecture on the same day overwrites the latest `.pt`.

---

## 4. Directory Structure

```
train-center/
├── README.md                 ← this file
├── 启动说明.md                ← quick-start notes (short version, Chinese)
├── requirements-server.txt   ← web backend dependencies
├── .gitignore                ← ignores large binaries: *.pt/*.pdparams/*.engine/*.onnx / runs / data/_* / PaddleOCR/output etc.
│
├── server/                   # FastAPI backend
│   ├── main.py               # app entry; mounts static, routes, CORS
│   ├── routes/
│   │   ├── data_routes.py    # data management (upload/extract/frames/dedup/split)
│   │   ├── label_routes.py   # annotation tool (read/save labels, export datasets)
│   │   ├── train_routes.py   # training tasks (3-stage start/stop/logs/cleanup_info+cleanup)
│   │   └── model_routes.py   # model management (list/preview/delete · cascades metrics)
│   └── services/
│       └── task_manager.py   # generic tasks: subprocess popen + log (deque 10000 lines) + meta
│
├── static/                   # Vue 3 + Element Plus + Fabric.js frontend (single-page index.html)
│   └── index.html            # 5 tabs: Data / Annotation / Training / Models / Logs
│
├── core/                     # pure business library (CLI-callable)
│   ├── common.py             # ROOT, CJK font locator, cv_imread/cv_imwrite (non-ASCII path safe)
│   ├── export.py             # labels → YOLO format, vertical-text rotation, rec cropping; export core
│   └── cleanup_artifacts.py  # det/rec intermediate checkpoint scan/delete; rec keeps best/latest/inference
│
├── data/                     # datasets (images/labels/YOLO txt / crop_img)
│   ├── det_fabric/           # data.yaml + images/train|val + labels/train|val
│   ├── det_text/
│   └── rec_text/             # crop_img/train|val + dict.txt + train.txt/val.txt
│
├── models/                   # training artifacts (not committed; .gitignore blocks)
│   ├── det/<name>.pt         + <name>.metrics.json
│   └── ocr/<name>/{best.pdparams, inference.{json,pdiparams,yml}}
│
├── runs/                     # YOLO / PaddleOCR training outputs (not committed; see .gitignore)
│
├── PaddleOCR/                # PaddleOCR source (submodule or nested; used for rec training)
│   ├── tools/train.py        # train_rec.py triggers train -c configs/rec/xxx.yml via CLI args
│   ├── tools/export_model.py # exports inference models
│   └── output/               # checkpoints (huge, .gitignore blocked)
│
├── train.py                  # det_fabric / det_text unified entry (CLI or API-called)
├── train_rec.py              # rec_text unified entry (CLI or API-called)
├── ingest.py                 # upload → extract → split
├── prepare_data.py           # dataset format conversion
├── dedup.py                  # dedup (pHash)
├── extract_frames.py         # video frame extraction
└── label_tool.py             # CLI annotation helper (also usable from web)
```

---

## 5. Dependencies & Dual-Environment Strategy

**Web service env** (`yolo-bench`):

```
fastapi>=0.110
uvicorn[standard]>=0.27
python-multipart>=0.0.9
pydantic>=2.5
PyYAML>=6.0
numpy>=1.24
opencv-python-headless>=4.8
Pillow>=10.0
```

**Training envs** (two separate conda envs; the API uses the `PADDLE_PY_EXE` absolute path pointing to paddle-ocr):

| Stage | conda env | Key dependencies |
|---|---|---|
| det_fabric / det_text | `yolo-bench` | ultralytics (YOLO26), torch>=2.2, cuda |
| rec_text              | `paddle-ocr` | paddlepaddle-gpu, PaddleOCR source dir, ppocr keys |

---

## 6. Key APIs

```
Upload:     POST   /api/data/upload (zip multipart)            /api/data/upload_folder
Dedup:      POST   /api/data/dedupe/:dataset
Export:     POST   /api/data/export/:dataset
Labels:     GET    /api/labels/:dataset/:split/:img            read
            POST   /api/labels/:dataset/:split/:img            save
Tasks:      GET    /api/train/tasks                             list
            GET    /api/train/tasks/:id?from=xxx               logs + meta
            DELETE /api/train/tasks/:id                        terminate
Training:   POST   /api/train/det_fabric  det_text  rec_text   start stages
Cleanup:    GET    /api/train/tasks/:id/cleanup_info           scan intermediates (no delete)
            POST   /api/train/tasks/:id/cleanup                perform deletion (saves X GB/MB)
Models:     GET    /api/models                                  list
            DELETE /api/models/(:category)/:name              delete (cascades metrics.json)
```

---

## 7. FAQ / Pitfalls

1. **rec training never runs eval?** → Change `eval_batch_step=[0,2000]` down to `[0,200]`. With 61 images at batch=8, total iterations ≈ 1400; the original threshold only evals at step 2000 — effectively never — so best_accuracy.pdparams never gets produced.
2. **YOLO training RuntimeError: text in labels?** → Older exports didn't strip text content; `core/export.py` now guarantees det_text labels are **pure YOLO coordinates**. Text content lives only in the annotation files (coordinates & text tab-separated) and is consumed during rec export cropping.
3. **cv2.imread/imwrite failing on non-ASCII paths?** → All replaced with `utils.cv_imread / cv_imwrite` (`np.fromfile + cv2.imdecode` / `cv2.imencode + .tofile`).
4. **Why two "V1"s (rec version vs PP-OCRv5's v5)?** → Our version number is the filename suffix `V1..V99`; PaddleOCRv5 is the model family code `PP-OCRv5_*_rec`. They are unrelated.
5. **Where are the TRT engines?** → This repo **only produces .pt and PaddleOCR inference models**. Actual ONNX/TRT engines are generated in the **fabric-algo** project via `tools/export_onnx.py` / `tools/build_engine.py` (or auto-converted after uploading a model bundle on the inference side's "Model Management" page).
6. **Intermediate checkpoints piling up?** → After training completes, the web frontend auto-pops a one-click cleanup dialog. Rules are unified in [core/cleanup_artifacts.py](./core/cleanup_artifacts.py).

---

## 8. Data / Model File Storage (Git LFS or Separate Repo)

Large binaries (`.pt / .pdparams / .onnx / .engine / runs export images`) are all `.gitignore`d. Recommended:
- Upload artifacts to object storage / network shares;
- Or use Git LFS `git lfs track "*.pt" "*.pdparams" "*.onnx"` (requires running after git init; this repo defaults to `.gitignore` exclusion).

---

## 9. Integration with the Inference Side (fabric-algo)

```
train-center                         fabric-algo
├─ models/det/fabric*.pt   ──► _export_det_onnx.py ──► .onnx ──► _trt11_onnx.py ──► models/fabric/*.engine
├─ models/det/text*.pt     ──► same                                models/text/*.engine
└─ models/ocr/rec*         ──► _export_rec_onnx.py ──► .onnx ──► _trt11_rec_onnx.py ─► models/rec/*.engine
                                                                       │
                                                          counter.py / infer_business.py / web_server.py
                                                          (real-time detection / mp4 export / counting-line drag / mode switch)
```
