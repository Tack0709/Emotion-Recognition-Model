import numpy as np
import argparse
import os
import glob

# クラス定義
EMOTION_NAMES = {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Anger', 4: 'Other'}

def analyze_errors(target_path, output_txt=None):
    """
    target_path: .npyファイルパス、またはディレクトリパス
    """
    # ディレクトリなら中の *_fold*.npy を全部探す
    if os.path.isdir(target_path):
        file_list = sorted(glob.glob(os.path.join(target_path, "*_fold*.npy")))
        if not file_list:
            print(f"Error: No result files found in directory: {target_path}")
            return
        print(f"Found {len(file_list)} result files in directory.")
    else:
        file_list = [target_path]

    all_errors = []
    total_samples = 0
    total_correct = 0

    print("Loading results...")
    
    # 全ファイルをループしてデータを統合
    for npy_file in file_list:
        try:
            data = np.load(npy_file, allow_pickle=True).item()
            fold_num = data.get('fold', '?')
            
            true_ids = data['true_ids']
            pred_ids = data['pred_ids']
            pred_probs = data['pred_probs']
            true_softs = data['true_softs']
            texts = data['texts']
            
            n_samples = len(true_ids)
            total_samples += n_samples
            
            # エラー抽出
            for i in range(n_samples):
                if true_ids[i] == pred_ids[i]:
                    total_correct += 1
                else:
                    all_errors.append({
                        'fold': fold_num, # どのFoldの間違いか
                        'text': texts[i],
                        'true_id': true_ids[i],
                        'pred_id': pred_ids[i],
                        'true_soft': true_softs[i],
                        'pred_prob': pred_probs[i]
                    })
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")

    # 出力先設定
    if output_txt is None:
        if os.path.isdir(target_path):
            output_txt = os.path.join(target_path, 'cross_validation_error_report.txt')
        else:
            output_txt = target_path.replace('.npy', '_error_report.txt')

    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    
    print(f"Total Samples: {total_samples}")
    print(f"Total Errors : {len(all_errors)}")
    print(f"Overall Acc  : {accuracy:.2f}%")
    print(f"Writing report to {output_txt} ...")

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(f"Cross Validation Error Analysis Report\n")
        f.write(f"Source: {target_path}\n")
        f.write(f"Files Processed: {len(file_list)}\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}% ({total_correct}/{total_samples})\n")
        f.write("="*60 + "\n\n")

        for idx, err in enumerate(all_errors):
            t_name = EMOTION_NAMES.get(err['true_id'], str(err['true_id']))
            p_name = EMOTION_NAMES.get(err['pred_id'], str(err['pred_id']))
            
            f.write(f"[Error #{idx+1}] (Fold {err['fold']})\n")
            f.write(f"Text : {err['text']}\n")
            f.write(f"True : {t_name:<7} (ID:{err['true_id']}) | Dist: {np.round(err['true_soft'], 2)}\n")
            f.write(f"Pred : {p_name:<7} (ID:{err['pred_id']}) | Prob: {np.round(err['pred_prob'], 2)}\n")
            
            # 惜しい間違い判定
            sorted_indices = np.argsort(err['pred_prob'])[::-1]
            if sorted_indices[1] == err['true_id']:
                 f.write(f"Note : Top-2 prediction was correct.\n")
            
            f.write("-" * 50 + "\n")

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ファイル単体でも、ディレクトリ指定でもOKにする
    parser.add_argument('target', type=str, help='Path to npy file OR directory containing npy files')
    args = parser.parse_args()
    
    analyze_errors(args.target)