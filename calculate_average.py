import os
import glob
import numpy as np
import argparse
import re

def calculate_average(log_dir='saved_models', eval_metric='loss', output_file=None, seed=None, nma=False, modality=None):
    """
    指定されたディレクトリ内のログファイルを読み込み、
    交差検証の平均スコア（F1, NLL, Acc）を計算して表示・保存する。
    """
    seed_pattern = f"seed{seed}" if seed is not None else "seed*"
    mode_dirname = 'nma' if nma else 'default'
    modality_pattern = modality if modality else '*'
    # シード / NMA / モダリティを反映した探索パターンを構築
    pattern = os.path.join(log_dir, seed_pattern, mode_dirname, modality_pattern, f"logging_{eval_metric}_fold*.log")
    all_log_files = glob.glob(pattern)

    if not all_log_files:
        print(f"エラー: ログファイルが見つかりません: {pattern}")
        return

    # フィルタリング処理
    log_files = []
    for f in all_log_files:
        # 条件に一致するログのみ残す
        # シード値のチェック
        if seed is not None:
            if f"_seed{seed}" not in f:
                continue
        
        # NMAフラグのチェック
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
    test_accs = []

    result_lines = []
    result_lines.append("-" * 40)
    result_lines.append(f"Metric: {eval_metric}")
    if seed is not None:
        result_lines.append(f"Seed  : {seed}")
    result_lines.append(f"NMA   : {nma}")
    result_lines.append(f"Modality: {modality or 'all'}")
    result_lines.append(f"Found : {len(log_files)} log files")
    result_lines.append("-" * 40)

    for log_file in sorted(log_files):
        # 各ログから評価結果を抽出
        with open(log_file, 'r') as f:
            content = f.read()
            
            f1_match = re.search(r"Test F1 at Best Val: (\d+\.\d+)", content)
            nll_match = re.search(r"Test (NLL|Loss) at Best Val: (\d+\.\d+)", content)
            acc_match = re.search(r"Test Acc at Best Val: (\d+\.\d+)", content)

            fold_match = re.search(r"fold(\d+)", log_file)
            fold_num = fold_match.group(1) if fold_match else "?"
            
            seed_match = re.search(r"seed(\d+)", log_file)
            seed_num = seed_match.group(1) if seed_match else "?"

            if f1_match and nll_match:
                f1 = float(f1_match.group(1))
                nll = float(nll_match.group(2))
                acc = float(acc_match.group(1)) if acc_match else 0.0
                
                test_f1s.append(f1)
                test_nlls.append(nll)
                if acc_match: test_accs.append(acc)
                
                prefix = f"Fold {fold_num}"
                if seed is None:
                    prefix += f" (Seed {seed_num})"
                
                acc_str = f", Acc = {acc:.2f}" if acc_match else ""
                result_lines.append(f"  {prefix}: F1 = {f1:.2f}, NLL = {nll:.4f}{acc_str}")
            else:
                result_lines.append(f"  Fold {fold_num}: Warning - Score not found.")

    result_lines.append("-" * 40)
    
    if test_f1s:
        # Fold 全体の平均と分散を計算
        avg_f1 = np.mean(test_f1s)
        std_f1 = np.std(test_f1s)
        avg_nll = np.mean(test_nlls)
        std_nll = np.std(test_nlls)

        result_lines.append(f"Average Test F1  : {avg_f1:.2f} (+/- {std_f1:.2f})")
        result_lines.append(f"Average Test NLL : {avg_nll:.4f} (+/- {std_nll:.4f})")
        
        if test_accs:
            avg_acc = np.mean(test_accs)
            std_acc = np.std(test_accs)
            result_lines.append(f"Average Test Acc : {avg_acc:.2f} (+/- {std_acc:.2f})")
    else:
        result_lines.append("有効なスコアが見つかりませんでした。")
    
    result_lines.append("-" * 40)

    output_text = "\n".join(result_lines)
    print(output_text)

    # 保存先ファイルパスを決定
    if output_file:
        out_path = output_file if os.path.isabs(output_file) else os.path.join(log_dir, output_file)
    else:
        if seed is not None:
            target_dir = os.path.join(log_dir, f"seed{seed}", mode_dirname)
            if modality:
                target_dir = os.path.join(target_dir, modality)
            os.makedirs(target_dir, exist_ok=True)
            suffix = modality or mode_dirname
            out_path = os.path.join(target_dir, f"average_result_{eval_metric}_{suffix}.txt")
        else:
            suffix = f"{mode_dirname}_{modality}" if modality else f"{mode_dirname}_all_modalities"
            out_path = os.path.join(log_dir, f"average_result_{eval_metric}_seed_all_{suffix}.txt")

    with open(out_path, 'w') as f:
        f.write(output_text)
    print(f"結果を保存しました: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='saved_models', type=str)
    parser.add_argument('--eval_metric', default='loss', type=str)
    parser.add_argument('--output_file', default=None, type=str)
    parser.add_argument('--seed', default=None, type=int, help='Filter logs by specific seed value')
    parser.add_argument('--nma', action='store_true', help='Filter logs for NMA mode')
    parser.add_argument('--modality', default=None, type=str, help='特定モダリティのログに限定')
    
    args = parser.parse_args()
    
    calculate_average(args.log_dir, args.eval_metric, args.output_file, args.seed, args.nma, args.modality)