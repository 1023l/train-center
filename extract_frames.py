"""
视频抽帧：从视频文件中按间隔提取帧为图片。

流程:
    视频 -> 按帧间隔/时间间隔抽帧 -> 输出到 data/frames/

用法:
    python extract_frames.py --input data/raw/                    # 处理目录下所有视频
    python extract_frames.py --input video.avi                    # 单个视频
    python extract_frames.py --input data/raw/ --interval 30     # 每 30 帧抽一帧
    python extract_frames.py --input data/raw/ --fps 1            # 每秒 1 帧
    python extract_frames.py --input data/raw/ --out data/frames
"""

import argparse
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="视频抽帧")
    p.add_argument("--input", required=True, help="视频文件或目录")
    p.add_argument("--out", default=str(ROOT / "data" / "frames"),
                   help="输出目录（默认 data/frames）")
    p.add_argument("--interval", type=int, default=None,
                   help="每隔 N 帧抽一帧（与 --fps 二选一，默认 30）")
    p.add_argument("--fps", type=float, default=None,
                   help="按帧率抽帧（如 1 = 每秒 1 帧，0.5 = 每 2 秒 1 帧）")
    p.add_argument("--quality", type=int, default=95,
                   help="JPEG 保存质量 1-100（默认 95）")
    return p.parse_args()


def extract_video(video_path: Path, out_dir: Path, interval: int, quality: int) -> int:
    """从单个视频抽帧，返回抽帧数量。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[frames] 无法打开视频: {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if interval is None:
        interval = 30

    out_subdir = out_dir / video_path.stem
    out_subdir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            name = f"{video_path.stem}_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_subdir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"[frames] {video_path.name}: 总帧 {frame_idx} | FPS {fps:.1f} | 抽帧 {saved} -> {out_subdir}")
    return saved


def extract_video_by_fps(video_path: Path, out_dir: Path, target_fps: float, quality: int) -> int:
    """按目标帧率抽帧（每 1/target_fps 秒抽一帧），返回抽帧数量。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[frames] 无法打开视频: {video_path}")
        return 0

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, round(source_fps / target_fps))
    return extract_video_intern(video_path, cap, out_dir, interval, quality)


def extract_video_intern(video_path: Path, cap, out_dir: Path, interval: int, quality: int) -> int:
    """内部抽帧逻辑（cap 已打开）。"""
    out_subdir = out_dir / video_path.stem
    out_subdir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            name = f"{video_path.stem}_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_subdir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"[frames] {video_path.name}: 抽帧 {saved} -> {out_subdir}")
    return saved


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 收集视频文件
    if src.is_file():
        videos = [src]
    elif src.is_dir():
        videos = sorted([f for f in src.rglob("*") if f.suffix.lower() in VIDEO_EXTS])
    else:
        raise FileNotFoundError(f"输入不存在: {src}")

    if not videos:
        print(f"[frames] 未找到视频文件: {src}")
        return

    print(f"[frames] 共 {len(videos)} 个视频，输出到 {out_dir}")

    # 确定抽帧方式
    if args.fps is not None:
        # 按帧率抽
        total = 0
        for v in videos:
            total += extract_video_by_fps(v, out_dir, args.fps, args.quality)
    else:
        interval = args.interval or 30
        total = 0
        for v in videos:
            total += extract_video(v, out_dir, interval, args.quality)

    print(f"\n[frames] 完成！共抽帧 {total} 张 -> {out_dir}")
    print(f"[frames] 下一步: python dedup.py --input {out_dir}")


if __name__ == "__main__":
    main()
