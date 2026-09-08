"""
鞋布片（fabric）检测模型训练脚本。

所有路径基于 fabric-algo 目录相对定位，可在任意 cwd 下运行。

用法示例:
    python train.py                              # 默认: souzhidata 数据 + yolo26n 全量训练
    python train.py --dataset bairundata         # 换用百润数据
    python train.py --model runs/xxx/weights/best.pt --epochs 100   # 增量微调
    python train.py --data custom.yaml           # 完全自定义数据集配置
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent          # train-center/
PROJECTS = ROOT.parent                          # c:\projects\

# 数据集注册表：键名 -> (数据集目录, 类别字典)
DATASETS = {
    "det_fabric": (ROOT / "data" / "det_fabric", {0: "fabric"}),
    "det_text": (ROOT / "data" / "det_text", {0: "text_h", 1: "text_v"}),
    "rec_text": (ROOT / "data" / "rec_text", {}),
}


def write_data_cfg(dataset: Path, tag: str, class_names: dict) -> Path:
    """生成数据集 yaml 配置到 runs/dataset_cfg/，返回 yaml 路径（ultralytics 8.4+ 需文件路径）。"""
    cfg = {
        "path": str(dataset),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }
    if (dataset / "images" / "test").is_dir():
        cfg["test"] = "images/test"
    out_dir = ROOT / "runs" / "dataset_cfg"
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = out_dir / f"{tag}.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return yaml_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="fabric 检测训练")
    p.add_argument("--dataset", choices=list(DATASETS), default="det_fabric",
                   help="数据集（det_fabric=鞋型布检测, det_text=文字检测, rec_text=文字识别）")
    p.add_argument("--data", type=str, default=None,
                   help="自定义 data.yaml 路径（优先级高于 --dataset）")
    p.add_argument("--model", default=str(ROOT / "models" / "det" / "yolo26n.pt"),
                   help="预训练权重路径（默认 models/det/yolo26n.pt）")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=24,
                   help="batch size（8GB 显存建议 <=24）")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--device", default="0", help="GPU 编号，如 0 / cpu")
    p.add_argument("--workers", type=int, default=2,
                   help="Windows 建议 <=2，避免多进程问题")
    p.add_argument("--name", default=None, help="实验名（默认自动生成：{prefix}{date}V{ver}）")
    p.add_argument("--project", default=None, help="输出根目录（默认 train-center/runs）")
    p.add_argument("--version", type=int, default=1, help="版本号（1=yolo26n, 2=retdtr, ...）")
    p.add_argument("--prefix", default=None, help="模型名前缀（默认按数据集自动：det_fabric→fabric, det_text→text）")
    p.add_argument("--val-only", action="store_true", help="只对已有模型跑测试集验证（不训练），输出 mAP 等指标")
    return p.parse_args()


def _round4(v) -> float:
    try:
        return round(float(v), 4) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _extract_metrics(results_dict: dict) -> dict:
    """从 ultralytics 指标 dict 里提取 P/R/mAP50/mAP50-95（兼容列名变体）。"""
    def pick(*keys):
        for k in keys:
            if k in results_dict:
                return _round4(results_dict[k])
        return 0.0
    return {
        "precision": pick("metrics/precision(B)", "metrics/precision"),
        "recall": pick("metrics/recall(B)", "metrics/recall"),
        "mAP50": pick("metrics/mAP50(B)", "metrics/mAP50"),
        "mAP50-95": pick("metrics/mAP50-95(B)", "metrics/mAP50-95"),
    }


def _load_last_results_metrics(results_csv: Path) -> dict:
    """从 ultralytics results.csv 读最后一行指标（训练末尾自动 val 的输出）。"""
    if not results_csv.is_file():
        return {}
    try:
        with open(results_csv, encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            rows = [ln.strip().split(",") for ln in f if ln.strip()]
        if not rows:
            return {}
        d = dict(zip(header, rows[-1]))
        return _extract_metrics(d)
    except OSError:
        return {}


def write_metrics(model_pt: Path, metrics: dict) -> None:
    """把 val 指标写到模型同名的 .metrics.json（模型管理页据此展示）。"""
    try:
        mj = Path(model_pt).with_suffix(".metrics.json")
        mj.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[train] 指标已保存: {mj}")
    except OSError:
        pass


def main() -> None:
    args = parse_args()

    # 关键：chdir 到 ROOT，让所有相对路径（包括 ultralytics 下载 yolo26n.pt 的缓存位置）
    # 都落在 train-center 根目录，避免不同 cwd 下重复下载
    os.chdir(ROOT)

    # 数据集配置（训练与验证共用）：优先 --data 指定的 yaml，否则按 --dataset 生成
    if args.data:
        data_cfg = str(Path(args.data))
        if not Path(args.data).is_absolute():
            data_cfg = str(ROOT / args.data)
    else:
        dataset_dir, class_names = DATASETS[args.dataset]
        if not (dataset_dir / "images" / "train").is_dir():
            raise FileNotFoundError(
                f"数据集图片不存在: {dataset_dir / 'images' / 'train'}\n"
                f"请确认数据已就位，或使用 --data 指定其他数据集。"
            )
        data_cfg = str(write_data_cfg(dataset_dir, args.dataset, class_names))
        print(f"[train] 数据集: {args.dataset} -> {dataset_dir}")
        print(f"[train] 类别: {class_names}")

    project = args.project or str(ROOT / "runs")
    # 模型名：{prefix}{date}V{version}，如 fabric20260825V1
    if args.name:
        name = args.name
    else:
        prefix = args.prefix
        if prefix is None:
            prefix = "fabric" if args.dataset == "det_fabric" else "text"
        date_str = datetime.now().strftime("%Y%m%d")
        name = f"{prefix}{date_str}V{args.version}"

    from ultralytics import YOLO

    # 强制模型路径 resolve：相对 ROOT 解析；不存在就报错
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (ROOT / args.model).resolve()
    else:
        model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"预训练权重不存在: {model_path}\n"
            f"（不能依赖 ultralytics 联网下载；请把权重放到 models/det/ 下然后传绝对路径）"
        )

    # ================= 验证模式：只跑测试集，不训练 =================
    if args.val_only:
        print(f"[val] 验证模型: {model_path} on dataset {args.dataset}")
        metrics_obj = YOLO(str(model_path)).val(
            data=data_cfg,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            project=project,
            name=name,
            exist_ok=True,
        )
        rd = getattr(metrics_obj, "results_dict", None) or {}
        m = _extract_metrics(rd)
        print(f"\n[val] 测试集指标  P={m['precision']}  R={m['recall']}  mAP50={m['mAP50']}  mAP50-95={m['mAP50-95']}")
        write_metrics(model_path, m)
        return

    model = YOLO(str(model_path))
    print(f"[train] 模型: {model_path} | imgsz={args.imgsz} batch={args.batch} epochs={args.epochs}")
    print(f"[train] 输出: {project}/{name}")

    model.train(
        data=data_cfg,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        project=project,
        name=name,
        exist_ok=True,
    )
    best_pt = Path(project) / name / "weights" / "best.pt"
    print(f"\n[train] 完成。最佳权重: {best_pt}")

    # 复制到 models/det/ 便于管理（det_fabric/det_text 都是 YOLO 检测模型；
    # rec 文字识别模型由 train_rec.py 导出到 models/ocr/）
    if best_pt.is_file():
        model_dir = ROOT / "models" / "det"
        model_dir.mkdir(parents=True, exist_ok=True)
        dest = model_dir / f"{name}.pt"
        shutil.copy2(best_pt, dest)
        print(f"[train] 已复制到: {dest}")
        # 训练末尾 ultralytics 已自动跑过 val：把指标存成 {name}.metrics.json，模型管理页直接展示
        metrics = _load_last_results_metrics(Path(project) / name / "results.csv")
        if metrics:
            write_metrics(dest, metrics)
            print(f"[train] 训练集 val 指标: {metrics}")


if __name__ == "__main__":
    main()
