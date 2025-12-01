import os
import glob
import numpy as np
import argparse
import re

def calculate_average(log_dir='saved_models', eval_metric='f1'):
    """
    指定されたディレクトリ内のログファイルを読み込み、
    交差検証の平均スコア（F1, Loss）を計算して表示する。
    """
    # ファイル検索パターン: logging_{metric}_fold*_seed*.log
    pattern = os.path.join(log_dir, f"logging_{eval_metric}_fold*_seed*.log")
    log_files = glob.glob(pattern)

    if not log_files:
        print(f"エラー: ログファイルが見つかりません: {pattern}")
        return

    test_f1s = []
    test_losses = []

    print("-" * 40)
    print(f"Found {len(log_files)} log files for metric '{eval_metric}':")

    for log_file in sorted(log_files):
        with open(log_file, 'r') as f:
            content = f.read()
            
            # ログファイルからスコアを抽出 (正規表現)
            # "Test F1 at Best Val: 65.43" のような行を探す
            f1_match = re.search(r"Test F1 at Best Val: (\d+\.\d+)", content)
            loss_match = re.search(r"Test Loss at Best Val: (\d+\.\d+)", content)

            fold_num = "Unknown"
            fold_match = re.search(r"fold(\d+)", log_file)
            if fold_match:
                fold_num = fold_match.group(1)

            if f1_match and loss_match:
                f1 = float(f1_match.group(1))
                loss = float(loss_match.group(1))
                test_f1s.append(f1)
                test_losses.append(loss)
                print(f"  Fold {fold_num}: F1 = {f1:.2f}, Loss = {loss:.4f}")
            else:
                print(f"  Fold {fold_num}: Warning - Score not found in log.")

    print("-" * 40)
    if test_f1s:
        avg_f1 = np.mean(test_f1s)
        std_f1 = np.std(test_f1s)
        avg_loss = np.mean(test_losses)
        std_loss = np.std(test_losses)

        print(f"Average Test F1  : {avg_f1:.2f} (+/- {std_f1:.2f})")
        print(f"Average Test Loss: {avg_loss:.4f} (+/- {std_loss:.4f})")
    else:
        print("有効なスコアが見つかりませんでした。")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='saved_models', type=str)
    parser.add_argument('--eval_metric', default='f1', type=str)
    args = parser.parse_args()
    
    calculate_average(args.log_dir, args.eval_metric)