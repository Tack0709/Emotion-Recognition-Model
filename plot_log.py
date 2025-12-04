import os
import glob
import re
import matplotlib.pyplot as plt
import argparse

def plot_validation_nll(log_dir='saved_models', eval_metric='loss', output_file='nll_learning_curve.png', seed=None, nma=False):
    """
    ログファイルを読み込み、検証データ(Val)のNLLの推移をグラフ化して保存する。
    ベストモデル（NLL最小）の地点に〇印をつける。
    """
    # --- ファイル検索 ---
    pattern = os.path.join(log_dir, f"logging_{eval_metric}_fold*.log")
    all_log_files = glob.glob(pattern)
    
    if not all_log_files:
        print(f"エラー: ログファイルが見つかりません: {pattern}")
        return

    # フィルタリング
    log_files = []
    for f in all_log_files:
        if seed is not None and f"_seed{seed}" not in f: continue
        if nma:
            if not f.endswith("_nma.log"): continue
        else:
            if f.endswith("_nma.log"): continue
        log_files.append(f)

    print(f"Found {len(log_files)} log files for Metric:{eval_metric}, Seed:{seed}, NMA:{nma}")
    if not log_files: return

    # --- データ読み込み ---
    data = {}
    log_pattern = re.compile(r"Ep\s+(\d+):.*?Val\s+\[NLL\s+(\d+\.\d+)")
    fold_pattern = re.compile(r"fold(\d+)")

    for log_path in sorted(log_files):
        fold_match = fold_pattern.search(log_path)
        fold_name = f"Fold {fold_match.group(1)}" if fold_match else os.path.basename(log_path)
        
        epochs = []
        nlls = []
        
        with open(log_path, 'r') as f:
            for line in f:
                match = log_pattern.search(line)
                if match:
                    epochs.append(int(match.group(1)))
                    nlls.append(float(match.group(2)))
        
        if epochs:
            data[fold_name] = (epochs, nlls)
            print(f"  Loaded {len(epochs)} epochs from {fold_name}")

    # --- グラフの描画 ---
    plt.figure(figsize=(12, 8)) # サイズを少し大きく
    
    for label, (epochs, nlls) in data.items():
        # 1. 通常の線をプロット
        p = plt.plot(epochs, nlls, label=label, marker='.', markersize=4, linewidth=1)
        line_color = p[0].get_color()

        # 2. ベスト地点（NLL最小）を探す
        min_nll = min(nlls)
        min_index = nlls.index(min_nll)
        best_epoch = epochs[min_index]

        # 3. ベスト地点に強調用の〇をつける
        # (黒い枠線の大きな丸)
        plt.plot(best_epoch, min_nll, 'o', 
                 markersize=12, 
                 markerfacecolor='none', 
                 markeredgecolor=line_color, 
                 markeredgewidth=2)
        
        # 4. スコアをテキストで表示
        plt.annotate(f"{min_nll:.4f}", 
                     xy=(best_epoch, min_nll), 
                     xytext=(0, -15), 
                     textcoords='offset points', 
                     ha='center', 
                     fontsize=9, 
                     color=line_color,
                     fontweight='bold')

    # タイトル・ラベル設定
    title_str = f'Validation NLL Learning Curve\n(Metric: {eval_metric}'
    if seed is not None: title_str += f', Seed: {seed}'
    if nma: title_str += ', NMA Mode'
    title_str += ')'
    
    plt.title(title_str)
    plt.xlabel('Epoch')
    plt.ylabel('NLL (Negative Log-Likelihood)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # ファイル名設定
    if output_file == 'nll_learning_curve.png':
        seed_suffix = f"_seed{seed}" if seed is not None else ""
        nma_suffix = "_nma" if nma else ""
        output_file = f"nll_learning_curve_{eval_metric}{seed_suffix}{nma_suffix}.png"

    plt.savefig(output_file, dpi=300)
    print(f"\nグラフを保存しました: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='saved_models', type=str)
    parser.add_argument('--eval_metric', default='loss', type=str)
    parser.add_argument('--output_file', default='nll_learning_curve.png', type=str)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--nma', action='store_true')
    
    args = parser.parse_args()
    plot_validation_nll(args.log_dir, args.eval_metric, args.output_file, args.seed, args.nma)