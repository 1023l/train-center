# 超算中心 · 布料 OCR 标注训练一体化平台

一套面向**鞋布片文字识别（fabric→text→rec 三阶段）**业务的端到端平台，把「数据上卷、去重、标注、格式转换、训练、模型管理」的完整流水线装进一个 FastAPI + Vue 3 Web 应用里。与同级目录的 **`fabric-algo`（在线推理/实时计数软件）**配合形成「训练→部署」闭环。

```
c:\projects\
├── train-center\   ← 本仓库（标注训练）
└── fabric-algo\    ← 姐妹项目（推理与 Web 实时视频检测）
```

---

## 1. 功能速览

| 模块 | 能力 |
|---|---|
| **数据管理** | 上传压缩包 / 单文件夹（子目录代表款式）→ 解压 → 视频抽帧 → 基于感知哈希去重 → 数据集自动划分 train/val |
| **标注工具** | Fabric.js + 原生 Canvas 实现两种模式：<br>① **目标检测模式**（布片框/文字区域，支持矩形框+4点旋转框；旋转框保存 Tab 分隔的坐标与文本）<br>② **OCR 模式**（切换横文 `text_h` / 竖文 `text_v` 两个按钮；竖文框导出后会自动 rot90 k=3 再裁剪作为 rec 数据） |
| **训练** | det_fabric / det_text 用 **YOLO26（ultralytics）**；rec_text 用 **PP-OCRv5（PaddleOCR）**。<br>三阶段顺序固定 `det_fabric → det_text → rec_text`，串行启动。训练过程实时日志/轮询、完成后自动刷新模型列表、**并弹窗提示清理中间 checkpoint 体积**。 |
| **模型管理** | 版本化命名 `[dataset]YYYYMMDDV[ver]`（V1=YOLO26n、V2=RetDR 等架构变化时递增）。每个模型：预览、`.metrics.json`、删除（级联删除附属 `.metrics.json`）。 |
| **日志** | 所有训练/上传/去重/导出任务统一管理：实时 log、状态 running/success/failed、elapsed。 |

---

## 1.1 本仓库不包含的内容

- **PaddleOCR 源码**：请自行放到 `PaddleOCR/`（rec 训练调用 `tools/train.py`）。
- **模型权重**（`.pt` / `.pdparams` / inference 二进制）：见 `models/`，不入库。
- **原始图片与标注大文件**：`data/` 下 images/labels 已忽略。

---

## 2. 一键启动

```bash
cd c:\projects\train-center
C:\Users\Administrator\miniconda3\envs\yolo-bench\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

> Web 后端只依赖 `fastapi + uvicorn`（见 [requirements-server.txt](./requirements-server.txt)）。真正训练用子进程切环境：
> - det_fabric / det_text：`yolo-bench` 环境（ultralytics + GPU）
> - rec_text：`paddle-ocr` 环境（paddlepaddle-gpu + PaddleOCR 源码 `tools/train.py`）

打开浏览器：

| 入口 | URL |
|---|---|
| 前端页面 | http://127.0.0.1:8000/ |
| 健康检查 | http://127.0.0.1:8000/health |
| Swagger 文档 | http://127.0.0.1:8000/docs |

停止服务：

```cmd
netstat -ano | findstr :8000     # 记 PID
taskkill /F /PID <PID>
```

---

## 3. 工作流（标准流程）

```
[1 上传/上卷]   POST /api/data/upload (zip)  或  upload_folder (子目录=款式)
      │
      ▼
[2 视频抽帧]   extract_frames.py  (CLI / 数据页)
      │
      ▼
[3 去重]       dedup.py (pHash 阈值 8)
      │
      ▼
[4 Web 标注]    det_fabric (目标检测矩形框)
                 │
                 └──► det_text (文字区域，矩形 / 4 点旋转框，text_h / text_v 类)
                        │
                        └──► rec_text (OCR 模式：每个框写识别文本；导出时自动旋转+裁剪)
      │
      ▼
[5 导出]       核心 [core/export.py](./core/export.py) 统一做 YOLO 5 值矩形 / 纯坐标清洗 / 竖文旋转裁剪
                 → det 输出 data/{det_fabric,det_text}/data.yaml + images/train|val + labels/train|val
                 → rec 输出 data/rec_text/{crop_img/train|val, dict.txt, train.txt, val.txt}
      │
      ▼
[6 训练三阶段 · 串行]
    ├─ det_fabric  train.py  (YOLO26)   → models/det/fabricYYYYMMDDVn.pt + .metrics.json
    ├─ det_text    train.py  (YOLO26)   → models/det/textYYYYMMDDVn.pt   + .metrics.json
    └─ rec_text    train_rec.py → 子进程调用 PaddleOCR/tools/train.py (PP-OCRv5 SVTR)
                                  → PaddleOCR/output/PP-OCRv5_{mobile,server}_rec/
                                  → 导出 inference 模型 → models/ocr/recYYYYMMDDVn/
                                        （3 件套：inference.json / .pdiparams / .yml）
      │
      ▼
[7 训练完成]    • 前端轮询自动识别 running→完成，弹窗：「训练产生中间 checkpoint 共 N 个 / X GB
                   是否清理节省空间？」（按钮"删除节省 X GB" / "保留"）
                • 清理规则详见 [core/cleanup_artifacts.py](./core/cleanup_artifacts.py)
                      rec: 删 iter_epoch_*.{pdopt,pdparams,states}，保留 best*/latest*/best_model/
                      det: 扫 runs/<fabric*,text*>/weights/，除 best.pt 之外全部可删
      │
      ▼
[8 导出 & 部署]   交给姐妹项目 fabric-algo：
                   _export_det_onnx.py（YOLO end2end NMS opset 17 imgsz 640）
                   _export_rec_onnx.py（Paddle → ONNX 动态宽）
                   _trt11_onnx.py / _trt11_rec_onnx.py → .engine（TRT 11.2 eval 版，FP32）
```

### 3.1 版本命名（硬约束）

**`[dataset]YYYYMMDDV[版本号]`**，例如：
```
fabric20260828V2    # fabric 数据集 · 2026-08-28 · 架构 V2
text20260831V1      # text   数据集 · 2026-08-31 · 架构 V1
rec20260828V3       # rec    数据集 · 2026-08-28 · 架构 V3
```

版本号**仅在架构变化（如 YOLO26n→RetDR，PP-OCRv5 mobile→server）**时递增；同一天同一架构重复训练会覆盖最新 `.pt`。

---

## 4. 目录结构

```
train-center/
├── README.md                 ← 本文件
├── 启动说明.md                ← 快速启动（简版）
├── requirements-server.txt   ← Web 后端依赖
├── .gitignore                ← 已忽略大二进制：*.pt/*.pdparams/*.engine/*.onnx / runs / data/_* / PaddleOCR/output 等
│
├── server/                   # FastAPI 后端
│   ├── main.py               # app 入口；挂载 static、routes、CORS
│   ├── routes/
│   │   ├── data_routes.py    # 数据管理（上传/解压/抽帧/去重/划分）
│   │   ├── label_routes.py   # 标注工具（读取保存标签、导出数据集）
│   │   ├── train_routes.py   # 训练任务（3 阶段启动/停止/日志/cleanup_info+cleanup）
│   │   └── model_routes.py   # 模型管理（列表/预览/删除·级联删 metrics）
│   └── services/
│       └── task_manager.py   # 通用任务：子进程 popen + log(deque 10000 行) + meta
│
├── static/                   # Vue 3 + Element Plus + Fabric.js 前端（单页 index.html）
│   └── index.html            # 5 个 Tab：数据管理 / 标注 / 训练 / 模型 / 日志
│
├── core/                     # 纯业务库（可 CLI 直接调用）
│   ├── common.py             # ROOT、中英文字体定位、cv_imread/cv_imwrite (中文路径安全)
│   ├── export.py             # 标签 → YOLO 格式、竖文旋转、rec 裁剪；导出核心
│   └── cleanup_artifacts.py  # det/rec 中间 checkpoint 扫描 / 删除；rec 保留 best/latest/inference
│
├── data/                     # 数据集 (images/labels/YOLO txt / crop_img)
│   ├── det_fabric/           # 数据.yaml + images/train|val + labels/train|val
│   ├── det_text/
│   └── rec_text/             # crop_img/train|val + dict.txt + train.txt/val.txt
│
├── models/                   # 训练产物（模型不入库，.gitignore 已屏蔽）
│   ├── det/<name>.pt         + <name>.metrics.json
│   └── ocr/<name>/{best.pdparams, inference.{json,pdiparams,yml}}
│
├── runs/                     # YOLO / PaddleOCR 训练输出（不入库；见 .gitignore）
│
├── PaddleOCR/                # PaddleOCR 源码（submodule 或直接嵌套均可；训练 rec 用）
│   ├── tools/train.py        # train_rec.py 通过命令行参数触发 train -c configs/rec/xxx.yml
│   ├── tools/export_model.py # 导出 inference 模型
│   └── output/               # checkpoint（巨大量，.gitignore 已屏蔽）
│
├── train.py                  # det_fabric / det_text 统一入口（CLI 或被 API 调）
├── train_rec.py              # rec_text 统一入口（CLI 或被 API 调）
├── ingest.py                 # 上传→解压→划分
├── prepare_data.py           # 数据集格式转换
├── dedup.py                  # 去重（pHash）
├── extract_frames.py         # 视频抽帧
└── label_tool.py             # 命令行版标注辅助工具（Web 上也可用）
```

---

## 5. 依赖 & 双环境策略

**Web 服务环境**（`yolo-bench`）：

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

**训练环境**（两份独立 conda env，API 端用 `PADDLE_PY_EXE` 绝对路径指向 paddle-ocr）：

| 阶段 | conda env | 主要依赖 |
|---|---|---|
| det_fabric / det_text | `yolo-bench` | ultralytics (YOLO26), torch>=2.2, cuda |
| rec_text              | `paddle-ocr` | paddlepaddle-gpu, PaddleOCR 源码目录, ppocr keys |

---

## 6. 关键 API

```
上传:     POST   /api/data/upload (zip multipart)            /api/data/upload_folder
去重:     POST   /api/data/dedupe/:dataset
导出:     POST   /api/data/export/:dataset
标注:     GET    /api/labels/:dataset/:split/:img            读取
          POST   /api/labels/:dataset/:split/:img            保存
任务:     GET    /api/train/tasks                             列表
          GET    /api/train/tasks/:id?from=xxx               日志 + meta
          DELETE /api/train/tasks/:id                        终止
训练:     POST   /api/train/det_fabric  det_text  rec_text   启动三阶段
清理:     GET    /api/train/tasks/:id/cleanup_info           扫描中间产物（不删）
          POST   /api/train/tasks/:id/cleanup                执行删除（节省 X GB/MB）
模型:     GET    /api/models                                  列表
          DELETE /api/models/(:category)/:name              删除（级联 metrics.json）
```

---

## 7. FAQ / 踩坑

1. **rec 训练全程不跑 eval？** → 把 `eval_batch_step=[0,2000]` 改小到 `[0,200]`。我们 61 张图 batch=8 时总迭代约 1400，原阈值只会到 2000 才跑一次 eval，等于没 eval，也就不会产出 best_accuracy.pdparams。
2. **YOLO 训练报 RuntimeError：label 里有文本？** → 旧版本导出未 strip 文本内容，现在 `core/export.py` 已保证 det_text label 是**纯 YOLO 坐标**，text 内容只保存在标注文件里（`\t` 分隔坐标与文本），会被 rec 导出裁剪时消费。
3. **中文路径下 cv2.imread/imwrite 失败？** → 全部改用 `utils.cv_imread / cv_imwrite`（`np.fromfile + cv2.imdecode` / `cv2.imencode + .tofile`）。
4. **为什么有两个 "V1"（rec 的版本和 PP-OCRv5 的 v5）？** → 我们自己的版本号是文件名后缀 `V1..V99`，PaddleOCRv5 是模型家族代号 `PP-OCRv5_*_rec`，两者没关系。
5. **TRT engine 去哪了？** → 本仓库**只产出 .pt 和 PaddleOCR inference**，真正的 ONNX/TRT engine 在 **fabric-algo** 项目中由 `_export_det_onnx.py / _trt11_onnx.py` 系列脚本生成。
6. **中间 checkpoint 越积越大？** → 训练完成后 Web 前端会自动弹窗，一键清理。规则统一写在 [core/cleanup_artifacts.py](./core/cleanup_artifacts.py)。

---

## 8. 数据 / 模型文件仓库（Git LFS 或别仓库）

大二进制（`.pt / .pdparams / .onnx / .engine / runs 导出图`）已全部 `.gitignore`。推荐：
- 把这些产物上传到对象存储 / 网络共享盘；
- 或用 Git LFS `git lfs track "*.pt" "*.pdparams" "*.onnx"`（若要启用会需要你在 git 初始化后执行，本仓库默认开启 `.gitignore` 排除）。

---

## 9. 与推理端 fabric-algo 对接

```
train-center                         fabric-algo
├─ models/det/fabric*.pt   ──► _export_det_onnx.py ──► .onnx ──► _trt11_onnx.py ──► models/fabric/*.engine
├─ models/det/text*.pt     ──► 同上                                models/text/*.engine
└─ models/ocr/rec*         ──► _export_rec_onnx.py ──► .onnx ──► _trt11_rec_onnx.py ─► models/rec/*.engine
                                                                       │
                                                          counter.py / infer_business.py / web_server.py
                                                          （实时检测 / 导出 mp4 / 计数线拖拽 / 模式切换）
```
