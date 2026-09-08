"""
PaddleOCR rec（文字识别）增量训练脚本。

自动处理：
1. 克隆/更新 PaddleOCR 仓库（含训练工具 tools/train.py）
2. 下载 PP-OCRv5_mobile_rec 预训练模型
3. 从训练数据生成字典 dict.txt
4. 生成训练配置 yaml（覆盖数据路径）
5. 启动训练
6. 导出推理模型到 models/ocr/

用法:
    python train_rec.py                              # 默认 50 epochs
    python train_rec.py --epochs 200                 # 训练 200 epochs
    python train_rec.py --model server               # 用 server 大模型
"""

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PADDLEOCR_REPO = ROOT / "PaddleOCR"
DATA_DIR = ROOT / "data" / "rec_text"
MODEL_DIR = ROOT / "models" / "ocr"
PRETRAIN_DIR = PADDLEOCR_REPO / "pretrain_models"

# PP-OCRv5 rec 预训练模型
MODELS = {
    "mobile": {
        "train_url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams",
        "name": "PP-OCRv5_mobile_rec",
        "config": "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml",
    },
    "server": {
        "train_url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams",
        "name": "PP-OCRv5_server_rec",
        "config": "configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml",
    },
}


def run(cmd, cwd=None):
    print(f"[run] {cmd}")
    env = os.environ.copy()
    # paddle 3.0 + cuDNN 9 在 Windows 上 backward 卷积偶发 CUDNN 4003（host 分配失败），限制 workspace 规避
    env.setdefault("FLAGS_cudnn_workspace_size_limit", "512")
    subprocess.run(cmd, shell=True, cwd=cwd, check=True, env=env)


def download(url, dest):
    """用 Python 下载文件。"""
    print(f"[rec] 下载: {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"[rec] 完成: {dest} ({dest.stat().st_size} bytes)")


def ensure_repo():
    """确保 PaddleOCR 仓库存在。"""
    if PADDLEOCR_REPO.is_dir() and (PADDLEOCR_REPO / "tools" / "train.py").is_file():
        print(f"[rec] PaddleOCR 仓库已存在: {PADDLEOCR_REPO}")
        return
    print("[rec] 克隆 PaddleOCR 仓库（Gitee 镜像）...")
    run(f'git clone --depth 1 https://gitee.com/paddlepaddle/PaddleOCR.git "{PADDLEOCR_REPO}"')


def ensure_pretrain(model_key):
    """下载预训练模型（训练用 .pdparams）。"""
    info = MODELS[model_key]
    pdparams = PRETRAIN_DIR / f"{info['name']}_pretrained.pdparams"
    if pdparams.is_file():
        print(f"[rec] 预训练模型已存在: {pdparams}")
        return pdparams
    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    download(info["train_url"], pdparams)
    return pdparams


def build_dict():
    """从训练数据生成字典 dict.txt。"""
    dict_path = DATA_DIR / "dict.txt"
    chars = set()
    for txt in [DATA_DIR / "train.txt", DATA_DIR / "val.txt"]:
        if not txt.is_file():
            continue
        for line in txt.read_text(encoding="utf-8").strip().splitlines():
            if "\t" in line:
                _, text = line.split("\t", 1)
                chars.update(text)
    # PaddleOCR dict 格式：每行一个字符，最后一行是空格
    with dict_path.open("w", encoding="utf-8") as f:
        for c in sorted(chars):
            f.write(f"{c}\n")
        f.write(" \n")  # 空格放最后
    print(f"[rec] 字典生成: {dict_path} ({len(chars)} 字符)")
    return dict_path


def infer_model_from_pretrained(pdparams: Path) -> str:
    """从预训练权重所在目录的 inference.yml 推断模型类型（mobile/server）。"""
    cfg = pdparams.parent / "inference.yml"
    if cfg.is_file():
        txt = cfg.read_text(encoding="utf-8")
        if "mobile" in txt.lower():
            return "mobile"
        if "server" in txt.lower():
            return "server"
    return "mobile"  # 兜底


def build_config(model_key):
    """返回配置文件路径。"""
    info = MODELS[model_key]
    config_path = PADDLEOCR_REPO / info["config"]
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR rec 增量训练")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model", choices=["mobile", "server"], default=None,
                        help="模型类型（不传时若指定 --pretrained 则从权重目录 inference.yml 自动推断，否则默认 mobile）")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--eval-step", type=int, default=50,
                        help="每隔多少训练步评估一次（当前数据 ~2 步/epoch；默认 50 步≈25 epoch 评估一次，太大会导致整个训练不评估）")
    parser.add_argument("--pretrained", default=None,
                        help="已训练 rec 模型的 pdparams 权重路径（如 models/ocr/rec20260826V1/best.pdparams）；"
                             "不传则按 --model 下载 PP-OCRv5 预训练权重")
    parser.add_argument("--data-dir", default=None, help="rec 训练数据目录（默认 data/rec_text）")
    args = parser.parse_args()

    # 数据目录可覆盖（默认 data/rec_text）
    global DATA_DIR
    if args.data_dir:
        data_dir_arg = Path(args.data_dir)
        if not data_dir_arg.is_absolute():
            data_dir_arg = (ROOT / args.data_dir).resolve()
        DATA_DIR = data_dir_arg

    # 1. 检查数据
    if not (DATA_DIR / "train.txt").is_file():
        print(f"[rec] 错误: 训练数据不存在 {DATA_DIR / 'train.txt'}")
        print("[rec] 请先用 label_tool 标注并导出 ocr 数据")
        return

    # 2. 预训练权重：优先 --pretrained（已训练模型继续增量），否则按 --model 下载
    ensure_repo()
    if args.pretrained:
        pdparams = Path(args.pretrained)
        if not pdparams.is_absolute():
            pdparams = (ROOT / args.pretrained).resolve()
        if not pdparams.is_file():
            print(f"[rec] 错误: 找不到预训练权重 {pdparams}")
            return
        print(f"[rec] 使用已训练权重继续增量: {pdparams}")
    else:
        pdparams = ensure_pretrain(args.model or "mobile")

    # 2.5 模型类型：显式指定优先；否则从预训练权重目录的 inference.yml 推断
    if args.model:
        model_key = args.model
    elif args.pretrained:
        model_key = infer_model_from_pretrained(pdparams)
    else:
        model_key = "mobile"
    print(f"[rec] 模型类型: {model_key}（配置 {MODELS[model_key]['name']}）")

    # 3. 生成字典
    dict_path = build_dict()

    # 4. 配置文件
    config_path = build_config(model_key)

    # 5. 启动训练（用 -o 覆盖路径）
    output_dir = ROOT / "runs" / "rec_output"
    print(f"\n[rec] 开始训练: {args.epochs} epochs, batch={args.batch}")
    # 路径转换为正斜杠，避免 yaml 解析问题
    pdparams_str = str(pdparams).replace("\\", "/")
    data_dir_str = str(DATA_DIR).replace("\\", "/")
    dict_str = str(dict_path).replace("\\", "/")
    output_str = str(output_dir).replace("\\", "/")
    train_txt = str(DATA_DIR / "train.txt").replace("\\", "/")
    val_txt = str(DATA_DIR / "val.txt").replace("\\", "/")

    cmd = (
        f'"{sys.executable}" tools/train.py '
        f'-c {config_path} '
        f'-o Global.pretrained_model={pdparams_str} '
        f'Global.epoch_num={args.epochs} '
        f'Global.output_dir={output_str} '
        f'Global.character_dict_path={dict_str} '
        f'Global.use_gpu=true '
        f'Global.eval_batch_step=[0,{args.eval_step}] '
        f'Train.dataset.data_dir={data_dir_str} '
        f'Train.dataset.label_file_list=["{train_txt}"] '
        f'Train.loader.batch_size_per_card={args.batch} '
        f'Eval.dataset.data_dir={data_dir_str} '
        f'Eval.dataset.label_file_list=["{val_txt}"] '
        f'Eval.loader.batch_size_per_card={args.batch}'
    )
    run(cmd, cwd=str(PADDLEOCR_REPO))

    # 6. 导出推理模型（优先 best，其次 latest）
    model_output_dir = output_dir / MODELS[model_key]["name"]
    best_model = model_output_dir / "best_accuracy.pdparams"
    latest_model = model_output_dir / "latest.pdparams"
    # PaddleOCR 可能保存到 repo 下的 output/
    repo_output = PADDLEOCR_REPO / "output" / MODELS[model_key]["name"]
    if not best_model.is_file():
        best_model = repo_output / "best_accuracy.pdparams"
    if not best_model.is_file():
        best_model = repo_output / "latest.pdparams"
    if not best_model.is_file():
        best_model = latest_model

    if best_model.is_file():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d")
        # 版本递增：rec{date}V{n}，同日期已存在则 n+1（避免同一天多次训练互相覆盖）
        v = 1
        while (MODEL_DIR / f"rec{date_str}V{v}").exists():
            v += 1
        export_name = f"rec{date_str}V{v}"
        export_dir = str(MODEL_DIR / export_name).replace("\\", "/")
        best_str = str(best_model).replace("\\", "/")
        export_cmd = (
            f'"{sys.executable}" tools/export_model.py '
            f'-c {config_path} '
            f'-o Global.pretrained_model={best_str} '
            f'Global.save_inference_dir={export_dir} '
            f'Global.character_dict_path={dict_str}'
        )
        run(export_cmd, cwd=str(PADDLEOCR_REPO))
        # 把训练权重复制为 best.pdparams，供下次增量训练（--pretrained）
        if best_model.is_file() and best_model.suffix == ".pdparams":
            shutil.copy2(best_model, MODEL_DIR / export_name / "best.pdparams")
        print(f"\n[rec] 导出完成: {MODEL_DIR / export_name}")
        print(f"[rec] 使用模型: {best_model}")
    else:
        print(f"[rec] 未找到模型文件: {best_model}")
        print(f"[rec] 尝试查找: {model_output_dir} 和 {repo_output}")


if __name__ == "__main__":
    main()
