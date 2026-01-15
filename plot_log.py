import os
import glob
import argparse
import re
import matplotlib.pyplot as plt

def find_log_files(log_dir, eval_metric, seed, nma, modality, simple_nn, standard_gnn, edl_r2, train_ma_test_nma):
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
    
    if train_ma_test_nma:
        mode_dirname = 'ma2nma'
    elif nma:
        mode_dirname = 'nma'
    else:
        mode_dirname = 'default'
    
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
            
        if train_ma_test_nma:
            if 'ma2nma' not in path: continue
        elif nma:
            if "_nma.log" not in path and "_nma_" not in path:
                continue
        else:
            if "_nma.log" in path or "_nma_" in path:
                continue
            if 'ma2nma' in path:
                continue
                
        files.append(path)
        
    return sorted(files)

def parse_metrics_inner(content_str):
    """
    文字列 "NLL 1.23 KL 0.45 F1 88.5" をパースして辞書にする
    """
    data = {}
    parts = content_str.strip().split()
    # キーと値のペアを取り出す
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            key = parts[i]
            val = parts[i+1]
            try:
                # 'Acc'などのキーもそのまま使う
                data[key] = float(val)
            except ValueError:
                pass
    return data

def parse_log_file(log_path):
    epoch_rows = []
    # 柔軟な正規表現: []の中身を非貪欲に取得する
    # 例: Ep 1: Train [NLL 1.2 KL 0.3] | Val [NLL 1.5 KL 0.4] ...
    regex = re.compile(
        r"Ep\s+(\d+):.*?Train\s+\[(.*?)\]\s+\|\s+Val\s+\[(.*?)\]\s+\|\s+Test\s+\[(.*?)\]",
        re.IGNORECASE,
    )
    
    with open(log_path, 'r') as log_file:
        for line in log_file:
            match = regex.search(line)
            if not match:
                continue
            
            epoch = int(match.group(1))
            train_str = match.group(2)
            val_str = match.group(3)
            test_str = match.group(4)
            
            epoch_rows.append(
                {
                    "epoch": epoch,
                    "train": parse_metrics_inner(train_str),
                    "val": parse_metrics_inner(val_str),
                    "test": parse_metrics_inner(test_str),
                }
            )
    return epoch_rows


def plot_metrics(rows, log_path, output_dir=None):
    if not rows:
        return

    epochs = [row["epoch"] for row in rows]
    
    # 最初の行に含まれるキー（指標名）をすべて取得してプロット対象にする
    first_data = rows[0]["train"]
    keys = list(first_data.keys()) # 例: ['NLL', 'KL', 'F1', 'Acc']
    
    # 表示順序を整える（見やすさのため）
    priority_order = ['NLL', 'KL', 'F1', 'Acc']
    metrics = sorted(keys, key=lambda k: priority_order.index(k) if k in priority_order else 999)

    num_metrics = len(metrics)
    if num_metrics == 0:
        print(f"プロットすべき指標が見つかりませんでした: {log_path}")
        return

    if output_dir:
        target_dir = output_dir if os.path.isabs(output_dir) else os.path.join(output_dir)
        os.makedirs(target_dir, exist_ok=True)
    else:
        target_dir = os.path.dirname(log_path)

    base_name = os.path.splitext(os.path.basename(log_path))[0]
    plot_path = os.path.join(target_dir, f"{base_name}.png")

    # 指標の数だけグラフを並べる
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))
    if num_metrics == 1:
        axes = [axes] # 配列化してループ処理を共通化

    for idx, key in enumerate(metrics):
        ax = axes[idx]
        ax.plot(epochs, [row["train"].get(key, None) for row in rows], label="Train")
        ax.plot(epochs, [row["val"].get(key, None) for row in rows], label="Val")
        ax.plot(epochs, [row["test"].get(key, None) for row in rows], label="Test")
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
    
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
    parser.add_argument('--modality', default='multimodal', type=str, help='特定モダリティのログだけ可視化')
    
    # アブレーション & EDL用フラグ
    parser.add_argument('--simple_nn', action='store_true')
    parser.add_argument('--standard_gnn', action='store_true')
    parser.add_argument('--edl_r2', action='store_true')

    parser.add_argument('--train_ma_test_nma', action='store_true')

    args = parser.parse_args()

    # ログファイルの検索
    log_files = find_log_files(
        args.log_dir, args.eval_metric, args.seed, args.nma, args.modality,
        args.simple_nn, args.standard_gnn, args.edl_r2,
        args.train_ma_test_nma
    )
    
    if not log_files:
        print(f"対象ログが見つかりません。")
        print(f"条件: Seed={args.seed}, NMA={args.nma}, MA2NMA={args.train_ma_test_nma}, Modality={args.modality}")
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