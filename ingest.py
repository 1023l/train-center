"""
数据接入：上传压缩包 → 解压 → 按后缀分类（视频/图片）。

支持格式:
    压缩包: .zip .tar .tar.gz .tgz .7z .rar
    直接传入目录也可以（跳过解压，直接扫描分类）

流程:
    1. 解压压缩包到 data/raw/
    2. 递归扫描所有文件，按后缀分类
    3. 输出统计：视频 N 个、图片 M 个、其他 K 个

用法:
    python ingest.py --input data.zip
    python ingest.py --input /path/to/dir
    python ingest.py --input data.zip --out data/raw
"""

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
ARCHIVE_EXTS = {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="数据接入：解压压缩包 + 分类")
    p.add_argument("--input", required=True, help="压缩包路径或目录")
    p.add_argument("--out", default=str(ROOT / "data" / "raw"),
                   help="解压输出目录（默认 data/raw）")
    return p.parse_args()


def extract_archive(archive: Path, out_dir: Path) -> Path:
    """解压压缩包到 out_dir，返回解压目录。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()

    if name.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(out_dir)
    elif name.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(out_dir)
    elif name.endswith(".7z"):
        try:
            import py7zr
            with py7zr.SevenZipFile(archive, mode="r") as sz:
                sz.extractall(out_dir)
        except ImportError:
            raise RuntimeError("解压 .7z 需要安装 py7zr: pip install py7zr")
    elif name.endswith(".rar"):
        try:
            import rarfile
            with rarfile.RarFile(archive, "r") as rf:
                rf.extractall(out_dir)
        except ImportError:
            raise RuntimeError("解压 .rar 需要安装 rarfile: pip install rarfile")
    else:
        raise ValueError(f"不支持的压缩格式: {archive.name}")

    print(f"[ingest] 解压完成: {archive.name} -> {out_dir}")
    return out_dir


def scan_files(directory: Path) -> dict[str, list[Path]]:
    """递归扫描目录，按文件类型分类。"""
    videos, images, others = [], [], []
    for f in directory.rglob("*"):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext in VIDEO_EXTS:
            videos.append(f)
        elif ext in IMAGE_EXTS:
            images.append(f)
        elif ext in ARCHIVE_EXTS:
            # 嵌套压缩包，跳过（可手动再次处理）
            others.append(f)
        else:
            others.append(f)
    return {"videos": videos, "images": images, "others": others}


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    out_dir = Path(args.out)

    if not src.exists():
        raise FileNotFoundError(f"输入不存在: {src}")

    # 判断是压缩包还是目录
    if src.is_file() and src.suffix.lower() in ARCHIVE_EXTS:
        extract_archive(src, out_dir)
        scan_dir = out_dir
    elif src.is_dir():
        # 直接扫描目录，复制到 out_dir
        if src.resolve() != out_dir.resolve():
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in src.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(src)
                    dst = out_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst)
            print(f"[ingest] 复制完成: {src} -> {out_dir}")
        scan_dir = out_dir
    else:
        raise ValueError(f"不支持的输入: {src}（需为压缩包或目录）")

    # 扫描分类
    result = scan_files(scan_dir)
    videos = sorted(result["videos"])
    images = sorted(result["images"])
    others = sorted(result["others"])

    print(f"\n[ingest] 扫描结果 ({scan_dir}):")
    print(f"  视频: {len(videos)} 个")
    for v in videos:
        print(f"    {v.relative_to(scan_dir)}")
    print(f"  图片: {len(images)} 个")
    print(f"  其他: {len(others)} 个")

    # 写入文件清单
    manifest = out_dir.parent / "file_list.txt"
    with manifest.open("w", encoding="utf-8") as f:
        f.write("# videos\n")
        for v in videos:
            f.write(f"{v}\n")
        f.write("\n# images\n")
        for img in images:
            f.write(f"{img}\n")
    print(f"\n[ingest] 文件清单: {manifest}")

    next_step = "extract_frames.py" if videos else "dedup.py"
    print(f"[ingest] 下一步: python {next_step} --input {out_dir}")


if __name__ == "__main__":
    main()
