import os
import glob
import numpy as np
import argparse
import re

def calculate_average(log_dir='saved_models', eval_metric='loss', output_file=None, 
                      seed=None, nma=False, modality='multimodal', 
                      simple_nn=False, standard_gnn=False, edl_r2=False,
                      train_ma_test_nma=False, train_nma_test_ma=False): # ★追加: 引数
    """
    指定されたディレクトリ内のログファイルを読み込み、
    交差検証の平均スコア（F1, NLL, Acc）を計算して表示・保存する。
    """
    
    # --- 1. アーキテクチャ名の決定 ---
    if edl_r2:
        arch_name = "edl_r2"
    elif simple_nn:
        arch_name = "simple_nn"
    elif standard_gnn:
        arch_name = "standard_gnn"
    else:
        arch_name = "dag_erc"

    # --- 2. ターゲットディレクトリの特定 ---
    seed_pattern = f"seed{seed}" if seed is not None else "seed*"
    
    # ★修正: ディレクトリ選択ロジック
    if train_ma_test_nma:
        mode_dirname = 'ma2nma'
    elif train_nma_test_ma:      # ★追加: nma2maモード
        mode_dirname = 'nma2ma'
    elif nma:
        mode_dirname = 'nma'
    else:
        mode_dirname = 'default'
    
    # ベースパス: saved_models/seedXXX/[mode_dirname]/
    base_dir_pattern = os.path.join(log_dir, seed_pattern, mode_dirname)

    # フォルダ分けルールの適用
    if edl_r2:
        target_dir_pattern = os.path.join(base_dir_pattern, 'edl_r2')
    elif modality == 'multimodal' and arch_name != 'dag_erc':
        target_dir_pattern = os.path.join(base_dir_pattern, arch_name)
    else:
        target_dir_pattern = os.path.join(base_dir_pattern, modality)

    # ログファイルの検索パターン構築
    log_filename_pattern = f"logging_{eval_metric}_fold*.log"
    search_path = os.path.join(target_dir_pattern, log_filename_pattern)
    
    all_log_files = glob.glob(search_path)

    if not all_log_files:
        print(f"エラー: ログファイルが見つかりません: {search_path}")
        print(f"検索条件 -> Arch: {arch_name}, Modality: {modality}, Seed: {seed}, NMA: {nma}, MA2NMA: {train_ma_test_nma}, NMA2MA: {train_nma_test_ma}")
        return

    # --- 3. フィルタリング処理 ---
    log_files = []
    for f in all_log_files:
        # ディレクトリ構造で既に絞り込まれているはずだが、念のためファイル名のチェック
        if train_ma_test_nma:
             if 'ma2nma' not in f: continue
        elif train_nma_test_ma: # ★追加
             if 'nma2ma' not in f: continue
        elif nma:
            if "_nma.log" not in f and "_nma_" not in f: 
                 continue
        else:
            if "_nma.log" in f or "_nma_" in f:
                continue
        
        log_files.append(f)

    if not log_files:
        print(f"条件に一致するログファイルがフィルタリング後に残りませんでした。")
        return

    test_f1s = []
    test_nlls = []
    test_accs = []

    result_lines = []
    result_lines.append("=" * 40)
    result_lines.append(f"Evaluation Result")
    result_lines.append("=" * 40)
    result_lines.append(f"Architecture: {arch_name}")
    result_lines.append(f"Metric      : {eval_metric}")
    result_lines.append(f"Modality    : {modality}")
    result_lines.append(f"Mode        : {mode_dirname}")
    result_lines.append(f"Seed        : {seed if seed is not None else 'All'}")
    result_lines.append(f"Found Logs  : {len(log_files)}")
    result_lines.append("-" * 40)

    # ログ読み込み処理
    for log_file in sorted(log_files):
        with open(log_file, 'r') as f:
            content = f.read()
            
            f1_match = re.search(r"Test F1 at Best Val: (\d+\.\d+)", content)
            nll_match = re.search(r"Test (NLL|Loss) at Best Val: (\d+\.\d+)", content)
            acc_match = re.search(r"Test Acc at Best Val: (\d+\.\d+)", content)

            fold_match = re.search(r"fold(\d+)", log_file)
            fold_num = fold_match.group(1) if fold_match else "?"
            
            seed_match = re.search(r"seed(\d+)", log_file)
            if not seed_match:
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
                
                acc_str = f", Acc = {acc:.4f}" if acc_match else ""
                result_lines.append(f"  {prefix}: F1 = {f1:.4f}, NLL = {nll:.4f}{acc_str}")
            else:
                result_lines.append(f"  Fold {fold_num}: Warning - Score not found in log.")

    result_lines.append("-" * 40)
    
    if test_f1s:
        avg_f1 = np.mean(test_f1s)
        std_f1 = np.std(test_f1s)
        avg_nll = np.mean(test_nlls)
        std_nll = np.std(test_nlls)

        result_lines.append(f"Average Test F1  : {avg_f1:.4f} (+/- {std_f1:.4f})")
        result_lines.append(f"Average Test NLL : {avg_nll:.4f} (+/- {std_nll:.4f})")
        
        if test_accs:
            avg_acc = np.mean(test_accs)
            std_acc = np.std(test_accs)
            result_lines.append(f"Average Test Acc : {avg_acc:.4f} (+/- {std_acc:.4f})")
    else:
        result_lines.append("有効なスコアが見つかりませんでした。")
    
    result_lines.append("=" * 40)

    output_text = "\n".join(result_lines)
    print(output_text)

    # --- 4. 結果の保存 ---
    if output_file:
        out_path = output_file
    else:
        if log_files:
            save_dir = os.path.dirname(log_files[0])
        else:
            save_dir = log_dir

        suffix = f"{arch_name}_{modality}"
        out_path = os.path.join(save_dir, f"average_result_{eval_metric}_{suffix}.txt")

    try:
        with open(out_path, 'w') as f:
            f.write(output_text)
        print(f"結果を保存しました: {out_path}")
    except Exception as e:
        print(f"保存に失敗しました: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='saved_models', type=str)
    parser.add_argument('--eval_metric', default='loss', type=str)
    parser.add_argument('--output_file', default=None, type=str)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--nma', action='store_true')
    parser.add_argument('--modality', default='multimodal', type=str)
    
    # アブレーション & EDL用フラグ
    parser.add_argument('--simple_nn', action='store_true')
    parser.add_argument('--standard_gnn', action='store_true')
    parser.add_argument('--edl_r2', action='store_true')
    
    # ★追加
    parser.add_argument('--train_ma_test_nma', action='store_true')
    parser.add_argument('--train_nma_test_ma', action='store_true') # ★追加
    
    args = parser.parse_args()
    
    calculate_average(
        log_dir=args.log_dir, 
        eval_metric=args.eval_metric, 
        output_file=args.output_file, 
        seed=args.seed, 
        nma=args.nma, 
        modality=args.modality,
        simple_nn=args.simple_nn,
        standard_gnn=args.standard_gnn,
        edl_r2=args.edl_r2,
        train_ma_test_nma=args.train_ma_test_nma,
        train_nma_test_ma=args.train_nma_test_ma # ★追加
    )