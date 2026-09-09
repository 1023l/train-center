# -*- coding: utf-8 -*-
"""rec 模型导出 ONNX（训练平台侧，paddle-ocr 环境运行）。

把 train-center/models/ocr/rec{ver}/best.pdparams 导出为动态宽 ONNX
（1x3x48x-1, opset 12），产物放到同目录 rec{ver}_onnx.onnx。

设计动机：算法侧（fabric-algo / Docker 容器）不再装 paddle 环境，
rec 的 ONNX 导出统一收敛到训练平台——模型包与单个下载直接携带 ONNX，
算法侧只做 ONNX -> TRT engine。

用法（paddle-ocr 环境）:
    python export_rec_onnx.py --src models/ocr/rec20260828V3 [--arch server] [--force]

缓存：产物已存在、非 0 字节且比 best.pdparams 新时跳过（--force 强制重导）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description="PaddleOCR rec 权重 -> ONNX（训练平台侧）")
    p.add_argument("--src", required=True, help="rec 模型目录（须含 best.pdparams），如 models/ocr/rec20260828V3")
    p.add_argument("--arch", default="server", choices=["server", "mobile"], help="训练配置架构")
    p.add_argument("--force", action="store_true", help="忽略缓存强制重导")
    a = p.parse_args()

    import yaml  # paddle-ocr 环境自带

    src = Path(a.src)
    if not src.is_absolute():
        src = ROOT / src
    pdparams = src / "best.pdparams"
    if not pdparams.is_file():
        raise SystemExit(f"[fatal] 找不到 {pdparams}")
    out = src.with_name(f"{src.name}_onnx")
    out_onnx = Path(f"{out}.onnx")

    # ---- 缓存检查 ----
    if not a.force and out_onnx.is_file() and out_onnx.stat().st_size > 0 \
            and out_onnx.stat().st_mtime >= pdparams.stat().st_mtime:
        print(f"[export] 缓存命中，跳过: {out_onnx} "
              f"({out_onnx.stat().st_size / 1e6:.1f} MB)")
        return

    t0 = time.perf_counter()
    sys.path.insert(0, str(ROOT / "PaddleOCR"))
    import paddle  # noqa: E402
    from ppocr.modeling.architectures import build_model  # noqa: E402

    yml = ROOT / "PaddleOCR" / ("configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml" if a.arch == "server"
                                else "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml")
    with open(yml, encoding="utf-8") as fp:
        cfg_ = yaml.safe_load(fp)
    arch_cfg = cfg_["Architecture"]
    print(f"[export] arch: {arch_cfg.get('name')}  yml: {yml.name}")

    # char_num = blank + dict + space（CTCLabelDecode；训练权重通道数必须一致）
    dict_path = ROOT / "data" / "rec_text" / "dict.txt"
    chars = dict_path.read_text(encoding="utf-8").splitlines()
    char_num = len(chars) + 2
    arch_cfg["Head"]["out_channels_list"] = {
        "CTCLabelDecode": char_num,
        "NRTRLabelDecode": char_num + 3,
    }
    print(f"[export] dict={len(chars)} chars, char_num={char_num}")

    model = build_model(arch_cfg)
    state = paddle.load(str(pdparams))
    model.set_state_dict(state)
    model.eval()

    spec = [paddle.static.InputSpec(shape=[1, 3, 48, -1], dtype="float32", name="x")]
    paddle.onnx.export(model, input_spec=spec, path=str(out), opset_version=12)

    # 产物校验：历史上出现过导出"成功"但产物 0 字节的残留文件
    if not out_onnx.is_file() or out_onnx.stat().st_size == 0:
        raise SystemExit(f"[fatal] ONNX 导出失败：产物缺失或为 0 字节: {out_onnx}")

    # paddle.onnx.export 会在输出目录旁留下推理模型中间产物（~72MB），用完即清
    tmp_dir = out.parent / "paddle_model_temp_dir"
    if tmp_dir.is_dir():
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[export] OK {time.perf_counter() - t0:.0f}s -> {out_onnx} "
          f"({out_onnx.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
