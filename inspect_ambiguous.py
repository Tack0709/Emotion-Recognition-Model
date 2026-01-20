import numpy as np
import argparse
import os
import glob

# 感情ラベルの定義
EMOTIONS = ['Neutral', 'Happy', 'Sad', 'Anger', 'Other']

def analyze_all_folds(target_dir):
    # ファイルを検索 (test_results_fold*.npy)
    search_pattern = os.path.join(target_dir, "test_results_fold*.npy")
    files = sorted(glob.glob(search_pattern))

    if not files:
        print(f"エラー: 指定されたディレクトリ '{target_dir}' に 'test_results_fold*.npy' が見つかりませんでした。")
        return

    print(f"=== Found {len(files)} files in '{target_dir}' ===\n")

    # 全体集計用変数
    grand_total_samples = 0
    grand_total_ambiguous = 0

    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"==========================================")
        print(f" Processing: {filename}")
        print(f"==========================================")

        try:
            data = np.load(file_path, allow_pickle=True).item()
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        true_softs = data['true_softs']
        pred_probs = data['pred_probs']
        texts = data['texts']
        
        n_samples = len(true_softs)
        n_ambiguous = 0
        
        grand_total_samples += n_samples

        for i, (soft_label, pred_prob) in enumerate(zip(true_softs, pred_probs)):
            # 正規化
            total_votes = np.sum(soft_label)
            if total_votes == 0: continue
            soft_prob = soft_label / total_votes

            # 最大得票率を取得
            max_vote_ratio = np.max(soft_prob)

            # 過半数割れ判定 (50%以下)
            # ※ 厳密に「過半数なし(Majority Vote成立せず)」を見るなら <= 0.5
            if max_vote_ratio <= 0.50001:
                n_ambiguous += 1
                
                # --- 詳細表示 (必要ならコメントアウトしてください) ---
                print(f" [Fold:{filename[-5]}] Index: {i}")
                print(f"  Text: {texts[i]}")
                
                # 正解分布の表示
                votes_str = ", ".join([f"{EMOTIONS[k]}:{soft_label[k]:.0f}({soft_prob[k]*100:.0f}%)" 
                                       for k in range(len(EMOTIONS)) if soft_label[k] > 0])
                print(f"  [正解分布] 票割れ (Max: {max_vote_ratio*100:.1f}%) -> {votes_str}")
                
                # モデル予測の表示
                pred_class = np.argmax(pred_prob)
                print(f"  [モデル予測] -> {EMOTIONS[pred_class]} ({pred_prob[pred_class]*100:.1f}%)")
                print("-" * 30)
        
        grand_total_ambiguous += n_ambiguous
        print(f"\n -> {filename} Result: {n_ambiguous} / {n_samples} ({n_ambiguous/n_samples*100:.1f}%) cases were ambiguous.\n")

    # 最終集計
    print("==========================================")
    print("           FINAL SUMMARY")
    print("==========================================")
    print(f"Target Directory: {target_dir}")
    print(f"Total Files Processed: {len(files)}")
    print(f"Total Samples: {grand_total_samples}")
    print(f"Total No-Majority Cases: {grand_total_ambiguous}")
    if grand_total_samples > 0:
        print(f"Overall Percentage: {grand_total_ambiguous / grand_total_samples * 100:.2f}%")
    print("==========================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ディレクトリパスを指定するように変更
    parser.add_argument('--dir', type=str, required=True, help='Directory containing test_results_fold*.npy files')
    args = parser.parse_args()
    
    analyze_all_folds(args.dir)