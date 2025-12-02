import os
import glob
import numpy as np
import argparse
import re

def calculate_average(log_dir='saved_models', eval_metric='loss', output_file=None):
    """
    指定されたディレクトリ内のログファイルを読み込み、
    交差検証の平均スコア（F1, NLL）を計算して表示・保存する。
    """
    # デフォルトが loss なので logging_loss_*.log を探す
    pattern = os.path.join(log_dir, f"logging_{eval_metric}_fold*_seed*.log")
    log_files = glob.glob(pattern)

    if not log_files:
        print(f"エラー: ログファイルが見つかりません: {pattern}")
        return

    test_f1s = []
    test_nlls = []

    result_lines = []
    result_lines.append("-" * 40)
    result_lines.append(f"Found {len(log_files)} log files for metric '{eval_metric}':")

    for log_file in sorted(log_files):
        with open(log_file, 'r') as f:
            content = f.read()
            
            f1_match = re.search(r"Test F1 at Best Val: (\d+\.\d+)", content)
            nll_match = re.search(r"Test NLL at Best Val: (\d+\.\d+)", content) # NLLを取得

            fold_match = re.search(r"fold(\d+)", log_file)
            fold_num = fold_match.group(1) if fold_match else "Unknown"

            if f1_match and nll_match:
                f1 = float(f1_match.group(1))
                nll = float(nll_match.group(1))
                test_f1s.append(f1)
                test_nlls.append(nll)
                result_lines.append(f"  Fold {fold_num}: F1 = {f1:.2f}, NLL = {nll:.4f}")
            else:
                result_lines.append(f"  Fold {fold_num}: Warning - Score not found in log.")

    result_lines.append("-" * 40)
    
    if test_f1s:
        avg_f1 = np.mean(test_f1s)
        std_f1 = np.std(test_f1s)
        avg_nll = np.mean(test_nlls)
        std_nll = np.std(test_nlls)

        result_lines.append(f"Average Test F1  : {avg_f1:.2f} (+/- {std_f1:.2f})")
        result_lines.append(f"Average Test NLL : {avg_nll:.4f} (+/- {std_nll:.4f})")
    else:
        result_lines.append("有効なスコアが見つかりませんでした。")
    
    result_lines.append("-" * 40)

    output_text = "\n".join(result_lines)
    print(output_text)

    if output_file:
        out_path = os.path.join(log_dir, output_file)
    else:
        out_path = os.path.join(log_dir, f"average_result_{eval_metric}.txt")
    
    with open(out_path, 'w') as f:
        f.write(output_text)
    print(f"結果を保存しました: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='saved_models', type=str)
    # <--- 変更: デフォルトを 'loss' に
    parser.add_argument('--eval_metric', default='loss', type=str)
    parser.add_argument('--output_file', default=None, type=str, help='Name of the output file')
    args = parser.parse_args()
    
    calculate_average(args.log_dir, args.eval_metric, args.output_file)