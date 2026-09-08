"""
构建 YOLO 训练数据目录：去重后的图片 → 分训练/测试集 → YOLO 目录结构。

流程:
    图片目录 -> 按比例分 train/val -> 复制到 data/yolo/ -> 生成 data.yaml

输出结构:
    data/yolo/
        images/
            train/   ← 训练集图片
            val/     ← 验证集图片
        labels/
            train/   ← 训练集标注（空 .txt，待标注）
            val/     ← 验证集标注（空 .txt，待标注）
        data.yaml    ← YOLO 配置

用法:
    python prepare_data.py --input data/dedup/
    python prepare_data.py --input data/dedup/ --ratio 0.85      # 85% 训练
    python prepare_data.py --input data/dedup/ --out data/yolo
    python prepare_data.py --input data/dedup/ --classes fabric,damage  # 多类别
"""

import argparse
import random
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建 YOLO 训练数据目录")
    p.add_argument("--input", required=True, help="去重后的图片目录")
    p.add_argument("--out", default=str(ROOT / "data" / "yolo"),
                   help="输出目录（默认 data/yolo）")
    p.add_argument("--ratio", type=float, default=0.8,
                   help="训练集比例 0~1（默认 0.8，即 80% 训练 20% 验证）")
    p.add_argument("--classes", default="fabric",
                   help="类别名列表，逗号分隔（默认 fabric）")
    p.add_argument("--seed", type=int, default=42, help="随机种子（保证可复现）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    out_dir = Path(args.out)

    if not src.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {src}")

    images = sorted([f for f in src.rglob("*") if f.suffix.lower() in IMAGE_EXTS])
    if not images:
        print(f"[prepare] 未找到图片: {src}")
        return

    # 随机打散分集合
    random.seed(args.seed)
    random.shuffle(images)
    split_idx = int(len(images) * args.ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    # 创建目录结构
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 复制图片 + 创建空标注文件
    def copy_split(imgs: list[Path], split: str) -> int:
        n = 0
        for img in imgs:
            dst_img = out_dir / "images" / split / img.name
            shutil.copy2(img, dst_img)
            # 创建同名空 .txt（YOLO 标注占位）
            lbl_path = out_dir / "labels" / split / f"{img.stem}.txt"
            if not lbl_path.exists():
                lbl_path.write_text("", encoding="utf-8")
            n += 1
        return n

    n_train = copy_split(train_imgs, "train")
    n_val = copy_split(val_imgs, "val")

    # 生成 data.yaml
    class_names = [c.strip() for c in args.classes.split(",")]
    cfg = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    print(f"[prepare] 数据集构建完成:")
    print(f"  训练集: {n_train} 张 -> {out_dir / 'images' / 'train'}")
    print(f"  验证集: {n_val} 张 -> {out_dir / 'images' / 'val'}")
    print(f"  配置: {yaml_path}")
    print(f"  类别: {class_names}")

    print(f"\n[prepare] 下一步:")
    print(f"  1. 用 LabelImg 或 LabelMe 标注图片（标注存到 labels/train, labels/val）")
    print(f"  2. python train.py --data {yaml_path}")
    print(f"     或: python train.py --dataset yolo  （自动找 data/yolo/data.yaml）")


if __name__ == "__main__":
    main()
