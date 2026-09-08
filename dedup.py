"""
图片去重：基于感知哈希（dHash）去除重复/高度相似的图片。

原理:
    dHash: 将图片转为 9x8 灰度图，比较相邻像素亮度差，生成 64bit 哈希。
    两图 Hamming Distance 越小越相似，阈值内判定为重复。

流程:
    图片目录 -> 计算每张图 dHash -> 两两比较 -> 保留首张，重复的移走

用法:
    python dedup.py --input data/frames/                        # 默认阈值 5
    python dedup.py --input data/frames/ --threshold 8          # 更宽松
    python dedup.py --input data/frames/ --out data/dedup/      # 指定输出目录
    python dedup.py --input data/frames/ --move data/dup/       # 重复图移到单独目录（不删除）
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="图片去重（dHash 感知哈希）")
    p.add_argument("--input", required=True, help="图片目录")
    p.add_argument("--out", default=None,
                   help="去重后输出目录（默认 data/dedup，复制去重后的图片）")
    p.add_argument("--threshold", type=int, default=5,
                   help="Hamming 距离阈值，小于等于则判为重复（默认 5）")
    p.add_argument("--move", default=None,
                   help="重复图片移动到此目录（不指定则仅列表，不移动）")
    return p.parse_args()


def dhash(img_path: Path, size: int = 9) -> np.ndarray | None:
    """计算图片 dHash（64bit），返回 64 维 0/1 数组。"""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # 缩放到 (size+1) x size，即 9x8
    resized = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    # 比较相邻像素：左 > 右 → 1
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()


def hamming(h1: np.ndarray, h2: np.ndarray) -> int:
    """计算两个哈希的 Hamming 距离。"""
    return int(np.count_nonzero(h1 != h2))


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    out_dir = Path(args.out) if args.out else ROOT / "data" / "dedup"
    move_dir = Path(args.move) if args.move else None

    if not src.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {src}")

    images = sorted([f for f in src.rglob("*") if f.suffix.lower() in IMAGE_EXTS])
    if not images:
        print(f"[dedup] 未找到图片: {src}")
        return

    print(f"[dedup] 共 {len(images)} 张图片，开始计算哈希...")

    # 计算所有图片的哈希
    hashes: list[tuple[Path, np.ndarray]] = []
    failed = 0
    for img_path in images:
        h = dhash(img_path)
        if h is not None:
            hashes.append((img_path, h))
        else:
            failed += 1
            print(f"[dedup] 无法读取: {img_path.name}")

    print(f"[dedup] 哈希计算完成: {len(hashes)} 张有效，{failed} 张失败")

    # 去重：保留首张，后续重复的标记
    unique: list[Path] = []
    duplicates: list[Path] = []

    for i, (path_i, hash_i) in enumerate(hashes):
        is_dup = False
        for path_u, hash_u in [(p, h) for p, h in hashes if p in unique]:
            if hamming(hash_i, hash_u) <= args.threshold:
                is_dup = True
                break
        if is_dup:
            duplicates.append(path_i)
        else:
            unique.append(path_i)

    print(f"\n[dedup] 结果: 唯一 {len(unique)} 张 | 重复 {len(duplicates)} 张")

    # 输出唯一图片
    out_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    for img_path in unique:
        dst = out_dir / img_path.name
        # 避免重名覆盖
        if dst.exists():
            dst = out_dir / f"{img_path.stem}_{hash(img_path)}{img_path.suffix}"
        shutil.copy2(img_path, dst)

    print(f"[dedup] 唯一图片已复制到: {out_dir}")

    # 移动重复图片（可选）
    if move_dir:
        move_dir.mkdir(parents=True, exist_ok=True)
        for img_path in duplicates:
            shutil.move(str(img_path), str(move_dir / img_path.name))
        print(f"[dedup] 重复图片已移动到: {move_dir}")
    elif duplicates:
        print(f"[dedup] 重复图片（未移动）:")
        for d in duplicates[:20]:
            print(f"  {d.name}")
        if len(duplicates) > 20:
            print(f"  ...共 {len(duplicates)} 张")

    print(f"\n[dedup] 下一步: python prepare_data.py --input {out_dir}")


if __name__ == "__main__":
    main()
