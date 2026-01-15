import numpy as np
import argparse
import os
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# クラス定義
EMOTION_NAMES = {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Anger', 4: 'Other', 5: 'XXX'}

def safe_calculate_entropy(probs):
    """エントロピー計算 (エラー時はNoneを返す)"""
    try:
        if probs is None or len(probs) == 0:
            return None
        
        probs = np.array(probs, dtype=float)
        
        # 合計が0またはNaNが含まれる場合は計算不可
        if np.sum(probs) == 0 or np.isnan(probs).any():
            return None
            
        # 正規化
        probs = probs / np.sum(probs)
        
        epsilon = 1e-10
        return -np.sum(probs * np.log(probs + epsilon))
    except Exception:
        return None

def safe_calculate_kl_divergence(p, q):
    """KLダイバージェンス KL(P || Q) (エラー時はNoneを返す)"""
    try:
        if p is None or q is None or len(p) == 0 or len(q) == 0:
            return None
        
        p = np.array(p, dtype=float)
        q = np.array(q, dtype=float)
        
        # サイズ不一致チェック
        if p.shape != q.shape:
            return None

        # 合計0やNaNチェック
        if np.sum(p) == 0 or np.sum(q) == 0 or np.isnan(p).any() or np.isnan(q).any():
            return None

        # 正規化
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        epsilon = 1e-10
        return np.sum(p * np.log((p + epsilon) / (q + epsilon)))
    except Exception:
        return None

def analyze_errors(target_path, output_dir=None):
    """
    target_path: .npyファイルパス、またはディレクトリパス
    """
    if not os.path.exists(target_path):
        print(f"Error: 指定されたパスが見つかりません: {target_path}")
        return

    if os.path.isdir(target_path):
        file_list = sorted(glob.glob(os.path.join(target_path, "*_fold*.npy")))
        if not file_list:
            print(f"Error: No result files found in directory: {target_path}")
            return
        print(f"Found {len(file_list)} result files in directory.")
        if output_dir is None:
            output_dir = target_path
    else:
        file_list = [target_path]
        if output_dir is None:
            output_dir = os.path.dirname(target_path)

    all_true_ids = []
    all_pred_ids = []
    all_results = [] 
    
    total_samples = 0
    total_correct = 0
    top2_correct = 0

    print("Loading results...")
    
    for npy_file in file_list:
        try:
            data = np.load(npy_file, allow_pickle=True).item()
            fold_num = data.get('fold', '?')
            
            true_ids = data['true_ids']
            pred_ids = data['pred_ids']
            
            # キーが存在しない場合の対策
            pred_probs = data.get('pred_probs', [None]*len(true_ids))
            true_softs = data.get('true_softs', [None]*len(true_ids))
            texts = data.get('texts', [""]*len(true_ids))
            
            if isinstance(true_ids, list):
                all_true_ids.extend(true_ids)
                all_pred_ids.extend(pred_ids)
            else:
                all_true_ids.extend(true_ids.tolist())
                all_pred_ids.extend(pred_ids.tolist())

            n_samples = len(true_ids)
            total_samples += n_samples
            
            for i in range(n_samples):
                t_id = true_ids[i]
                p_id = pred_ids[i]
                is_correct = (t_id == p_id)
                
                if is_correct:
                    total_correct += 1
                
                # 分布情報の取得 (Noneチェック)
                t_soft = true_softs[i] if (true_softs is not None and i < len(true_softs)) else None
                p_prob = pred_probs[i] if (pred_probs is not None and i < len(pred_probs)) else None
                
                # KL / Entropy 計算 (安全な関数を使用)
                kl_div = safe_calculate_kl_divergence(t_soft, p_prob)
                t_entropy = safe_calculate_entropy(t_soft)
                p_entropy = safe_calculate_entropy(p_prob)

                # Top-2 判定 & ランク確認
                is_top2_flip = False
                true_rank = -1
                
                if p_prob is not None and len(p_prob) > 0:
                    try:
                        p_prob_arr = np.array(p_prob)
                        if not np.isnan(p_prob_arr).any():
                            sorted_indices = np.argsort(p_prob_arr)[::-1]
                            ranks = np.where(sorted_indices == t_id)[0]
                            if len(ranks) > 0:
                                true_rank = ranks[0] + 1
                            
                            if true_rank <= 2:
                                top2_correct += 1
                            
                            if not is_correct and true_rank == 2:
                                is_top2_flip = True
                    except:
                        pass # 計算エラー時は何もしない

                all_results.append({
                    'fold': fold_num,
                    'text': texts[i],
                    'true_id': t_id,
                    'pred_id': p_id,
                    'is_correct': is_correct,
                    'is_top2_flip': is_top2_flip,
                    'true_rank': true_rank,
                    'true_soft': t_soft,
                    'pred_prob': p_prob,
                    'kl_div': kl_div,
                    't_entropy': t_entropy,
                    'p_entropy': p_entropy
                })

        except Exception as e:
            print(f"Error loading or processing {npy_file}: {e}")
            # エラーが起きても続行する

    if total_samples == 0:
        print("Error: 有効なデータが読み込めませんでした。")
        return

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # --- 1. 全件レポートの作成 ---
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    top2_acc = (top2_correct / total_samples) * 100 if total_samples > 0 else 0
    
    report_path = os.path.join(output_dir, 'full_analysis_report.txt')
    
    print(f"Total Samples: {total_samples}")
    print(f"Overall Acc  : {accuracy:.2f}%")
    print(f"Top-2 Acc    : {top2_acc:.2f}%")
    print(f"Writing full report to {report_path} ...")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Full Analysis Report (All Utterances)\n")
        f.write(f"Source: {target_path}\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}% ({total_correct}/{total_samples})\n")
        f.write(f"Top-2 Accuracy  : {top2_acc:.2f}% ({top2_correct}/{total_samples})\n")
        f.write("="*80 + "\n")
        f.write("Legend:\n")
        f.write("  [CORRECT]      : Prediction matched Ground Truth\n")
        f.write("  [ERROR]        : Wrong prediction\n")
        f.write("  (Top-2 Flip)   : Wrong, but Ground Truth was the 2nd highest probability\n")
        f.write("="*80 + "\n\n")

        for idx, res in enumerate(all_results):
            t_name = EMOTION_NAMES.get(res['true_id'], str(res['true_id']))
            p_name = EMOTION_NAMES.get(res['pred_id'], str(res['pred_id']))
            
            # ステータス表示
            if res['is_correct']:
                status = "[CORRECT]"
            else:
                if res['is_top2_flip']:
                    status = ">> [ERROR] (Top-2 Flip) <<"
                else:
                    rank_str = str(res['true_rank']) if res['true_rank'] != -1 else "?"
                    status = f">> [ERROR] (True Rank: {rank_str}) <<"
            
            f.write(f"#{idx+1} (Fold {res['fold']}) {status}\n")
            f.write(f"Text : {res['text']}\n")
            
            # 確信度の比較
            p_true_val_str = "N/A"
            p_pred_val_str = "N/A"
            
            if res['pred_prob'] is not None and len(res['pred_prob']) > max(res['true_id'], res['pred_id']):
                try:
                    p_true_val_str = f"{res['pred_prob'][res['true_id']]:.4f}"
                    p_pred_val_str = f"{res['pred_prob'][res['pred_id']]:.4f}"
                except:
                    pass

            f.write(f"True : {t_name:<7} (ID:{res['true_id']}) | Model Prob: {p_true_val_str}\n")
            f.write(f"Pred : {p_name:<7} (ID:{res['pred_id']}) | Model Prob: {p_pred_val_str}\n")
            
            # KL / Entropy 表示 (計算できた場合のみ)
            metrics_str = []
            if res['kl_div'] is not None: metrics_str.append(f"KL Div: {res['kl_div']:.4f}")
            if res['t_entropy'] is not None: metrics_str.append(f"T-Entropy: {res['t_entropy']:.4f}")
            if res['p_entropy'] is not None: metrics_str.append(f"P-Entropy: {res['p_entropy']:.4f}")
            
            if metrics_str:
                f.write(f"Metrics -> {' | '.join(metrics_str)}\n")

            # 分布の比較テーブル
            if res['true_soft'] is not None and res['pred_prob'] is not None:
                try:
                    f.write(f"Dist Table:\n")
                    f.write(f"  {'Label':<8} {'True(GT)':<9} {'Pred(Model)':<11} {'Diff(Gap)':<10}\n")
                    f.write(f"  {'-'*42}\n")
                    
                    t_arr = np.array(res['true_soft'], dtype=float)
                    p_arr = np.array(res['pred_prob'], dtype=float)
                    
                    # 正規化 (表示用)
                    if np.sum(t_arr) > 1.0 + 1e-5: t_arr /= np.sum(t_arr)
                    
                    for li in range(len(t_arr)):
                        l_name = EMOTION_NAMES.get(li, str(li))
                        t_val = t_arr[li]
                        p_val = p_arr[li] if li < len(p_arr) else 0.0
                        diff = p_val - t_val 
                        
                        f.write(f"  {l_name:<8} {t_val:.4f}    {p_val:.4f}      {diff:+.4f}")
                        
                        mark = ""
                        if li == res['true_id']: mark += "  <- True"
                        if li == res['pred_id']: mark += "  <- Pred"
                        f.write(f"{mark}\n")
                except Exception as e:
                    f.write(f"  (Distribution data error: {e})\n")

            f.write("-" * 50 + "\n")

    # --- 2. 混同行列の作成 ---
    print("Generating Confusion Matrix...")
    unique_labels = sorted(list(set(all_true_ids) | set(all_pred_ids)))
    display_labels = []
    for uid in unique_labels:
        if uid in EMOTION_NAMES:
            display_labels.append(EMOTION_NAMES[uid])
        else:
            display_labels.append(str(uid))

    cm = confusion_matrix(all_true_ids, all_pred_ids, labels=unique_labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title(f"Confusion Matrix (Acc: {accuracy:.1f}%)")
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    cm_norm = confusion_matrix(all_true_ids, all_pred_ids, labels=unique_labels, normalize='true')
    fig, ax = plt.subplots(figsize=(8, 6))
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=display_labels)
    disp_norm.plot(cmap=plt.cm.Blues, ax=ax, values_format='.2f')
    plt.title(f"Normalized Confusion Matrix (Recall)")
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_norm.png'))
    plt.close()

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=str, help='Path to npy file OR directory containing npy files')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save results')
    args = parser.parse_args()
    
    analyze_errors(args.target, args.output_dir)