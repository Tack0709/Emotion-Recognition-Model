import os
import glob
import numpy as np
import argparse
import re

def calculate_average(log_dir='saved_models', eval_metric='loss', output_file=None, seed=None):
    """
    指定されたディレクトリ内のログファイルを読み込み、
    交差検証の平均スコア（F1, NLL）を計算して表示・保存する。
    特定のシード値のみを対象にすることも可能。
    """
    # シード指定の有無で検索パターンを変える
    if seed is not None:
        seed_pattern = f"seed{seed}"
    else:
        seed_pattern = "seed*"

    # ファイル検索パターン: logging_{metric}_fold*_{seed}.log
    pattern = os.path.join(log_dir, f"logging_{eval_metric}_fold*_{seed_pattern}.log")
    log_files = glob.glob(pattern)

    if not log_files:
        print(f"エラー: ログファイルが見つかりません: {pattern}")
        return

    test_f1s = []
    test_nlls = []

    result_lines = []
    result_lines.append("-" * 40)
    result_lines.append(f"Metric: {eval_metric}")
    if seed is not None:
        result_lines.append(f"Seed  : {seed}")
    result_lines.append(f"Found : {len(log_files)} log files")
    result_lines.append("-" * 40)

    # ログファイルを読み込んでスコアを抽出
    for log_file in sorted(log_files):
        with open(log_file, 'r') as f:
            content = f.read()
            
            f1_match = re.search(r"Test F1 at Best Val: (\d+\.\d+)", content)
            # NLL または Loss に対応
            nll_match = re.search(r"Test (NLL|Loss) at Best Val: (\d+\.\d+)", content)

            # ファイル名から Fold番号 と Seed番号 を抽出
            fold_match = re.search(r"fold(\d+)", log_file)
            fold_num = fold_match.group(1) if fold_match else "?"
            
            seed_match = re.search(r"seed(\d+)", log_file)
            seed_num = seed_match.group(1) if seed_match else "?"

            if f1_match and nll_match:
                f1 = float(f1_match.group(1))
                nll = float(nll_match.group(2))
                test_f1s.append(f1)
                test_nlls.append(nll)
                
                # シード指定なしの場合は、どのシードの結果かも表示
                prefix = f"Fold {fold_num}"
                if seed is None:
                    prefix += f" (Seed {seed_num})"
                
                result_lines.append(f"  {prefix}: F1 = {f1:.2f}, NLL = {nll:.4f}")
            else:
                result_lines.append(f"  Fold {fold_num} (Seed {seed_num}): Warning - Score not found.")

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

    # ファイル保存
    if output_file:
        out_path = os.path.join(log_dir, output_file)
    else:
        # ファイル名を指定しない場合は自動生成
        # 例: average_result_loss_seed100.txt
        seed_suffix = f"_seed{seed}" if seed is not None else ""
        out_path = os.path.join(log_dir, f"average_result_{eval_metric}{seed_suffix}.txt")
    
    with open(out_path, 'w') as f:
        f.write(output_text)
    print(f"結果を保存しました: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='saved_models', type=str)
    parser.add_argument('--eval_metric', default='loss', type=str)
    parser.add_argument('--output_file', default=None, type=str)
    
    # <--- 追加: シード指定引数
    parser.add_argument('--seed', default=None, type=int, help='Filter logs by specific seed value')
    
    args = parser.parse_args()
    
    calculate_average(args.log_dir, args.eval_metric, args.output_file, args.seed)