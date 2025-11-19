import os
import numpy as np
import sys
from collections import defaultdict

def create_label_files(iemo_root='IEMOCAP_full_release/', output_dir='output_data'):
    """
    IEMOCAPのEmoEvaluationファイル を解析し、
    ハードラベルとソフトラベルの辞書を作成して.npyファイルとして保存します。
    
    (修正版：C-E, C-F, C-M のすべての行を読み込むように修正)
    """
    
    # --- 1. ディレクトリ設定 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_roots = [
        os.path.join(iemo_root, f"Session{i}/dialog/EmoEvaluation") 
        for i in range(1, 6)
    ]
    
    # print(f"Processing directories: {file_roots}")

    file_paths = []
    for file_dir in file_roots:
        if not os.path.exists(file_dir):
            print(f"Warning: ディレクトリが見つかりません: {file_dir}")
            continue
        for files in os.listdir(file_dir):
            # ⬇️ 修正: '._' で始まるファイルは無視する
            if files.startswith('._'):
                continue
            if files.endswith('.txt'):
                file_paths.append(os.path.join(file_dir, files))
    
    print(f"合計 {len(file_paths)} 個の EmoEvaluation ファイルを処理します。")
    if not file_paths:
        print(f"エラー: {iemo_root} にデータが見つかりません。")
        return

    # --- 2. ラベルマッピング定義 (ERC-SLT22 準拠) ---
    
    mapping = {
        'Neutral':   [1, 0, 0, 0, 0],
        'Happiness': [0, 1, 0, 0, 0],
        'Excited':   [0, 1, 0, 0, 0], # Happinessに統合
        'Sadness':   [0, 0, 1, 0, 0],
        'Anger':     [0, 0, 0, 1, 0]
    }
    OTHER_VECTOR = [0, 0, 0, 0, 1]
    
    emo_dic = {
        'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3, 'oth': 4, 'xxx': 5
    }

    # --- 3. ファイル解析とデータ抽出 ---
    hard_label_dic = {} 
    soft_label_dic = {} 

    current_utt_id = None
    current_hard_label = None
    current_soft_labels = []

    for label_path in file_paths:
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    if not line:
                        continue

                    if line.startswith('['):
                        # --- 以前の発話IDのデータを保存 ---
                        if current_utt_id:
                            if current_hard_label == 'exc':
                                current_hard_label = 'hap'
                            if current_hard_label not in emo_dic:
                                current_hard_label = 'oth'
                            
                            hard_label_dic[current_utt_id] = emo_dic[current_hard_label]
                            
                            if current_soft_labels:
                                soft_label_dic[current_utt_id] = np.array(current_soft_labels).sum(axis=0)
                            else:
                                soft_label_dic[current_utt_id] = np.array([0, 0, 0, 0, 0])

                        # --- 新しい発話データを初期化 ---
                        parts = line.split()
                        if len(parts) > 4:
                            current_utt_id = parts[3]       # 例: Ses01F_impro01_F000
                            current_hard_label = parts[4].lower() # 例: neu
                            current_soft_labels = []        
                        else:
                            current_utt_id = None 
                    
                    # #############################################################
                    # ⬇️ 修正点: 'C-E' だけでなく 'C-F' や 'C-M' も対象にする
                    # #############################################################
                    elif (line.startswith('C-E') or line.startswith('C-F') or line.startswith('C-M')) and current_utt_id:
                        parts = line.split()
                        # C-F1: Neutral; Anger; () のような行から 'Neutral', 'Anger' を取得
                        emotion_tags = parts[1].split(';')
                        
                        for tag in emotion_tags:
                            if tag and tag != '()': # 空白や()を除外
                                current_soft_labels.append(mapping.get(tag, OTHER_VECTOR))
                    # #############################################################

        except Exception as e:
            print(f"エラー: ファイル {label_path} の処理中に問題が発生しました: {e}")
            sys.exit(1)


    # --- 最後の発話データを保存 ---
    if current_utt_id:
        if current_hard_label == 'exc':
            current_hard_label = 'hap'
        if current_hard_label not in emo_dic:
            current_hard_label = 'oth'
        hard_label_dic[current_utt_id] = emo_dic[current_hard_label]
        
        if current_soft_labels:
            soft_label_dic[current_utt_id] = np.array(current_soft_labels).sum(axis=0)
        else:
            soft_label_dic[current_utt_id] = np.array([0, 0, 0, 0, 0])

    # --- 4. ファイル保存 ---
    hard_label_path = os.path.join(output_dir, 'IEMOCAP-hardlabel.npy')
    soft_label_path = os.path.join(output_dir, 'IEMOCAP-softlabel-sum.npy')

    try:
        np.save(hard_label_path, hard_label_dic)
        np.save(soft_label_path, soft_label_dic)
    except Exception as e:
        print(f"エラー: ファイルの保存中に問題が発生しました: {e}")
        sys.exit(1)

    print(f"処理完了。総発話数: {len(hard_label_dic)}")
    print(f"  ➡️  {hard_label_path} (ハードラベル辞書)")
    print(f"  ➡️  {soft_label_path} (ソフトラベル合計値辞書)")
    
    if len(hard_label_dic) != 10039 or len(soft_label_dic) != 10039:
        print(f"Warning: 期待される発話数(10039)と一致しません。 検出数: {len(hard_label_dic)}")
    else:
        print("発話数が期待値(10039)と一致しました。")


# --- スクリプト実行 ---
if __name__ == "__main__":
    create_label_files(
        iemo_root=r'/home/datasets/mizuno/IEMOCAP_full_release/IEMOCAP_full_release', 
        output_dir='output_data'
    )