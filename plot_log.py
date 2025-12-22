import os
import glob
import argparse
import re

import matplotlib.pyplot as plt


def find_log_files(log_dir, eval_metric, seed, nma, modality):
    seed_pattern = f"seed{seed}" if seed is not None else "seed*"
    mode_dirname = 'nma' if nma else 'default'
    modality_pattern = modality if modality else '*'
    pattern = os.path.join(log_dir, seed_pattern, mode_dirname, modality_pattern, f"logging_{eval_metric}_fold*.log")
    candidates = glob.glob(pattern)
    files = []
    for path in candidates:
        if seed is not None and f"_seed{seed}" not in path:
            continue
        if nma and not path.endswith("_nma.log"):
            continue
        if not nma and path.endswith("_nma.log"):
            continue
        files.append(path)
    return sorted(files)


def parse_log_file(log_path):
    epoch_rows = []
    regex = re.compile(
        r"Ep\s+(\d+):.*?Train\s+\[NLL\s+([-\d\.eE]+)\s+F1\s+([-\d\.eE]+)\s+Acc\s+([-\d\.eE]+)\]\s+\|\s+"
        r"Val\s+\[NLL\s+([-\d\.eE]+)\s+F1\s+([-\d\.eE]+)\s+Acc\s+([-\d\.eE]+)\]\s+\|\s+"
        r"Test\s+\[NLL\s+([-\d\.eE]+)\s+F1\s+([-\d\.eE]+)\s+Acc\s+([-\d\.eE]+)\]",
        re.IGNORECASE,
    )
    with open(log_path, 'r') as log_file:
        for line in log_file:
            match = regex.search(line)
            if not match:
                continue
            values = list(map(float, match.groups()[1:]))
            epoch_rows.append(
                {
                    "epoch": int(match.group(1)),
                    "train": {"nll": values[0], "f1": values[1], "acc": values[2]},
                    "val": {"nll": values[3], "f1": values[4], "acc": values[5]},
                    "test": {"nll": values[6], "f1": values[7], "acc": values[8]},
                }
            )
    return epoch_rows


def plot_metrics(rows, log_path, output_dir=None):
    epochs = [row["epoch"] for row in rows]
    metrics = [("NLL", "nll"), ("F1", "f1"), ("Acc", "acc")]

    if output_dir:
        target_dir = output_dir if os.path.isabs(output_dir) else os.path.join(output_dir)
        os.makedirs(target_dir, exist_ok=True)
    else:
        target_dir = os.path.dirname(log_path)

    base_name = os.path.splitext(os.path.basename(log_path))[0]
    plot_path = os.path.join(target_dir, f"{base_name}.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (title, key) in enumerate(metrics):
        axes[idx].plot(epochs, [row["train"][key] for row in rows], label="Train")
        axes[idx].plot(epochs, [row["val"][key] for row in rows], label="Val")
        axes[idx].plot(epochs, [row["test"][key] for row in rows], label="Test")
        axes[idx].set_title(title)
        axes[idx].set_xlabel("Epoch")
        axes[idx].grid(True, linestyle="--", alpha=0.3)
        axes[idx].legend()
    fig.suptitle(base_name, fontsize=12)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"プロットを保存しました: {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='saved_models', type=str)
    parser.add_argument('--eval_metric', default='loss', type=str)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--nma', action='store_true')
    parser.add_argument('--output_dir', default=None, type=str)
    parser.add_argument('--modality', default=None, type=str, help='特定モダリティのログだけ可視化')
    args = parser.parse_args()

    log_files = find_log_files(args.log_dir, args.eval_metric, args.seed, args.nma, args.modality)
    if not log_files:
        print(f"対象ログが見つかりません: {os.path.join(args.log_dir, 'seed*/default|nma/*')}")
        return

    out_dir = args.output_dir
    if out_dir and not os.path.isabs(out_dir):
        out_dir = os.path.join(args.log_dir, out_dir)

    for log_path in log_files:
        rows = parse_log_file(log_path)
        if not rows:
            print(f"エポック情報を解析できませんでした: {log_path}")
            continue
        plot_metrics(rows, log_path, out_dir)


if __name__ == '__main__':
    main()