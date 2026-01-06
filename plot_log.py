import os
import glob
import argparse
import re
import matplotlib.pyplot as plt

def find_log_files(log_dir, eval_metric, seed, nma, modality, simple_nn, standard_gnn, edl_r2):
    # --- 1. アーキテクチャ名の決定 ---
    if edl_r2:
        arch_name = "edl_r2"
    elif simple_nn:
        arch_name = "simple_nn"
    elif standard_gnn:
        arch_name = "standard_gnn"
    else:
        arch_name = "dag_erc"

    # --- 2. ディレクトリパスの構築 ---
    seed_pattern = f"seed{seed}" if seed is not None else "seed*"
    mode_dirname = 'nma' if nma else 'default'
    
    base_dir_pattern = os.path.join(log_dir, seed_pattern, mode_dirname)

    if edl_r2:
        target_dir_pattern = os.path.join(base_dir_pattern, 'edl_r2')
    elif modality == 'multimodal' and arch_name != 'dag_erc':
        target_dir_pattern = os.path.join(base_dir_pattern, arch_name)
    else:
        mod_pat = modality if modality else '*'
        target_dir_pattern = os.path.join(base_dir_pattern, mod_pat)

    file_pattern = f"logging_{eval_metric}_fold*.log"
    search_path = os.path.join(target_dir_pattern, file_pattern)
    
    candidates = glob.glob(search_path)
    
    # --- 3. フィルタリング ---
    files = []
    for path in candidates:
        if seed is not None and f"seed{seed}" not in path and f"seed{seed}" not in os.path.dirname(path):
            continue
            
        if nma:
            if "_nma.log" not in path and "_nma_" not in path:
                continue
        else:
            if "_nma.log" in path or "_nma_" in path:
                continue
                
        files.append(path)
        
    return sorted(files)


def parse_log_file(log_path):
    epoch_rows = []
    
    # --- Regex修正のポイント ---
    # 1. "Train" だけでなく "Tr" に対応
    # 2. "Test" だけでなく "MA" に対応 (これがTestとして扱われます)
    # 3. F1スコアが省略されている場合 (?: ...)? に対応 (なければ0.0)
    regex = re.compile(
        r"Ep\s+(\d+):.*?"
        # Train part: F1は省略可能とする
        r"(?:Train|Tr)\s+\[NLL\s+([-\d\.eE]+)(?:\s+F1\s+([-\d\.eE]+))?\s+Acc\s+([-\d\.eE]+)\]\s+\|\s+"
        # Val part: F1は省略可能とする
        r"Val\s+\[NLL\s+([-\d\.eE]+)(?:\s+F1\s+([-\d\.eE]+))?\s+Acc\s+([-\d\.eE]+)\]\s+\|\s+"
        # Test/MA part: "MA" または "Test"
        r"(?:Test|MA)\s+\[NLL\s+([-\d\.eE]+)\s+F1\s+([-\d\.eE]+)\s+Acc\s+([-\d\.eE]+)\]",
        re.IGNORECASE,
    )
    
    with open(log_path, 'r') as log_file:
        for line in log_file:
            match = regex.search(line)
            if not match:
                continue
            
            # グループの取得 (Noneの場合は0.0にする)
            epoch = int(match.group(1))
            
            # Train
            t_nll = float(match.group(2))
            t_f1  = float(match.group(3)) if match.group(3) else 0.0 # Missing F1 -> 0.0
            t_acc = float(match.group(4))
            
            # Val
            v_nll = float(match.group(5))
            v_f1  = float(match.group(6)) if match.group(6) else 0.0 # Missing F1 -> 0.0
            v_acc = float(match.group(7))
            
            # Test (MA)
            test_nll = float(match.group(8))
            test_f1  = float(match.group(9))
            test_acc = float(match.group(10))

            epoch_rows.append(
                {
                    "epoch": epoch,
                    "train": {"nll": t_nll, "f1": t_f1, "acc": t_acc},
                    "val":   {"nll": v_nll, "f1": v_f1, "acc": v_acc},
                    "test":  {"nll": test_nll, "f1": test_f1, "acc": test_acc},
                }
            )
            
    if not epoch_rows:
        print(f"警告: 有効なログ行が見つかりませんでした: {log_path}")
        
    return epoch_rows


def plot_metrics(rows, log_path, output_dir=None):
    if not rows:
        return

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
        axes[idx].plot(epochs, [row["test"][key] for row in rows], label="Test (MA)") # ラベル変更
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
    
    parser.add_argument('--modality', default='multimodal', type=str)
    
    # アブレーション & EDL用フラグ
    parser.add_argument('--simple_nn', action='store_true')
    parser.add_argument('--standard_gnn', action='store_true')
    parser.add_argument('--edl_r2', action='store_true')

    args = parser.parse_args()

    # ログファイルの検索
    log_files = find_log_files(
        args.log_dir, args.eval_metric, args.seed, args.nma, args.modality,
        args.simple_nn, args.standard_gnn, args.edl_r2
    )
    
    if not log_files:
        print(f"対象ログが見つかりません。")
        return

    out_dir = args.output_dir
    if out_dir and not os.path.isabs(out_dir):
        out_dir = os.path.join(args.log_dir, out_dir)

    for log_path in log_files:
        print(f"Processing: {log_path}")
        rows = parse_log_file(log_path)
        plot_metrics(rows, log_path, out_dir)


if __name__ == '__main__':
    main()