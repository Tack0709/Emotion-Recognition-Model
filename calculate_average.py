import os
import glob
import numpy as np
import argparse
import re

def calculate_average(log_dir='saved_models', eval_metric='loss', output_file=None, seed=None, nma=False):
    """
    指定されたディレクトリ内のログファイルを読み込み、
    交差検証の平均スコア（F1, NLL）を計算して表示・保存する。
    特定のシード値やNMAフラグの有無でフィルタリング可能。
    """
    # まずは広めにログファイルを検索
    pattern = os.path.join(log_dir, f"logging_{eval_metric}_fold*.log")
    all_log_files = glob.glob(pattern)

    if not all_log_files:
        print(f"エラー: ログファイルが見つかりません: {pattern}")
        return

    # フィルタリング処理
    log_files = []
    for f in all_log_files:
        # 1. シード値のチェック
        if seed is not None:
            if f"_seed{seed}" not in f:
                continue
        
        # 2. NMAフラグのチェック
        # nma=True ならファイル名末尾が "_nma.log" であるべき
        # nma=False ならファイル名末尾が "_nma.log" であってはいけない
        if nma:
            if not f.endswith("_nma.log"):
                continue
        else:
            if f.endswith("_nma.log"):
                continue
        
        log_files.append(f)

    if not log_files:
        print(f"条件に一致するログファイルが見つかりませんでした。")
        print(f"Metric: {eval_metric}, Seed: {seed}, NMA: {nma}")
        return

    test_f1s = []
    test_nlls = []

    result_lines = []
    result_lines.append("-" * 40)
    result_lines.append(f"Metric: {eval_metric}")
    if seed is not None:
        result_lines.append(f"Seed  : {seed}")
    result_lines.append(f"NMA   : {nma}")
    result_lines.append(f"Found : {len(log_files)} log files")
    result_lines.append("-" * 40)

    for log_file in sorted(log_files):
        with open(log_file, 'r') as f:
            content = f.read()
            
            f1_match = re.search(r"Test F1 at Best Val: (\d+\.\d+)", content)
            nll_match = re.search(r"Test (NLL|Loss) at Best Val: (\d+\.\d+)", content)

            fold_match = re.search(r"fold(\d+)", log_file)
            fold_num = fold_match.group(1) if fold_match else "?"
            
            seed_match = re.search(r"seed(\d+)", log_file)
            seed_num = seed_match.group(1) if seed_match else "?"

            if f1_match and nll_match:
                f1 = float(f1_match.group(1))
                nll = float(nll_match.group(2))
                test_f1s.append(f1)
                test_nlls.append(nll)
                
                prefix = f"Fold {fold_num}"
                if seed is None:
                    prefix += f" (Seed {seed_num})"
                
                result_lines.append(f"  {prefix}: F1 = {f1:.2f}, NLL = {nll:.4f}")
            else:
                result_lines.append(f"  Fold {fold_num}: Warning - Score not found.")

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
        # 自動ファイル名生成
        seed_suffix = f"_seed{seed}" if seed is not None else ""
        nma_suffix = "_nma" if nma else ""
        out_path = os.path.join(log_dir, f"average_result_{eval_metric}{seed_suffix}{nma_suffix}.txt")
    
    with open(out_path, 'w') as f:
        f.write(output_text)
    print(f"結果を保存しました: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='saved_models', type=str)
    parser.add_argument('--eval_metric', default='loss', type=str)
    parser.add_argument('--output_file', default=None, type=str)
    parser.add_argument('--seed', default=None, type=int, help='Filter logs by specific seed value')
    # <--- 追加: NMAフラグ
    parser.add_argument('--nma', action='store_true', help='Filter logs for NMA mode')
    
    args = parser.parse_args()
    
    calculate_average(args.log_dir, args.eval_metric, args.output_file, args.seed, args.nma)