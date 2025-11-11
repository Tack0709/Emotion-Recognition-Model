import os
import numpy as np

def create_label_files(iemo_root='IEMOCAP_full_release', output_dir='output_data'):
    """
    IEMOCAPのEmoEvaluationファイル を解析し、
    ハードラベルとソフトラベルの辞書を作成して.npyファイルとして保存します。
    
    処理は tack0709/erc-slt22/.../data_prep_process_label.py に準拠しています。
    """
    
    # --- 1. ディレクトリ設定 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Session1から5までの EmoEvaluation ディレクトリパスを生成
    file_roots = [
        os.path.join(iemo_root, f"Session{i}/dialog/EmoEvaluation") 
        for i in range(1, 6)
    ]

    file_paths = []
    for file_dir in file_roots:
        if not os.path.exists(file_dir):
            print(f"Warning: ディレクトリが見つかりません: {file_dir}")
            continue
        for files in os.listdir(file_dir):
            if files.endswith('.txt'):
                file_paths.append(os.path.join(file_dir, files))
    
    print(f"合計 {len(file_paths)} 個の EmoEvaluation ファイルを処理します。")
    if not file_paths:
        print(f"エラー: {iemo_root} にデータが見つかりません。")
        return

    # --- 2. ラベルマッピング定義 (ERC-SLT22 準拠) ---
    
    # ソフトラベル用マッピング (5クラス: neu, hap/exc, sad, ang, oth)
    # 'exc' (excited) は 'hap' (happiness) に統合されます。
    mapping = {
        'Neutral':   [1, 0, 0, 0, 0],
        'Happiness': [0, 1, 0, 0, 0],
        'Excited':   [0, 1, 0, 0, 0], # Happinessに統合
        'Sadness':   [0, 0, 1, 0, 0],
        'Anger':     [0, 0, 0, 1, 0]
        # 上記以外は 'Other' [0, 0, 0, 0, 1]として処理されます
    }
    
    # ハードラベル用マッピング (6クラス: xxx, oth を含む)
    emo_dic = {
        'neu': 0, # Neutral
        'hap': 1, # Happiness / Excited
        'sad': 2, # Sadness
        'ang': 3, # Anger
        'oth': 4, # Other
        'xxx': 5  # Garbage/No agreement
    }

    # --- 3. ファイル解析とデータ抽出 ---
    hard_label_dic = {} # {発話ID: ラベルIndex}
    soft_label_dic = {} # {発話ID: [n, h, s, a, o] の合計値配列}

    current_utt_id = None
    current_hard_label = None
    current_soft_labels = []

    for label_path in file_paths:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    continue

                # 画像 の [START_TIME - END_TIME] TURN_NAME EMOTION ... の行
                if line.startswith('['):
                    # 以前の発話IDのデータを保存
                    if current_utt_id:
                        # ハードラベルの正規化 ('exc' -> 'hap', それ以外 -> 'oth')
                        if current_hard_label == 'exc':
                            current_hard_label = 'hap'
                        if current_hard_label not in emo_dic:
                            current_hard_label = 'oth'
                        
                        # ハードラベル辞書に保存
                        hard_label_dic[current_utt_id] = emo_dic[current_hard_label]
                        
                        # ソフトラベル辞書に保存 (合計値)
                        if current_soft_labels:
                            soft_label_dic[current_utt_id] = np.array(current_soft_labels).sum(axis=0)
                        else:
                            # C-E行が全くなかった場合は、[0,0,0,0,0] を設定
                            soft_label_dic[current_utt_id] = np.array([0, 0, 0, 0, 0])

                    # 新しい発話データを初期化
                    parts = line.split()
                    current_utt_id = parts[3]       # 例: Ses01F_impro01_F000
                    current_hard_label = parts[4] # 例: neu
                    current_soft_labels = []        # 評価者ごとのラベルリストをリセット

                # 画像 の C-E: ... の行
                elif line.startswith('C-E'):
                    parts = line.split()
                    # C-E: Neutral; () のような行から 'Neutral' のみ取得
                    emotion_tags = parts[1].split(';')
                    
                    for tag in emotion_tags:
                        if not tag:
                            continue
                        # 'mapping' にあればそのベクトルを、なければ 'Other' のベクトルを追加
                        current_soft_labels.append(mapping.get(tag, [0, 0, 0, 0, 1]))

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

    np.save(hard_label_path, hard_label_dic)
    np.save(soft_label_path, soft_label_dic)

    print(f"処理完了。発話数: {len(hard_label_dic)}")
    print(f"ハードラベル辞書を保存しました: {hard_label_path}")
    print(f"ソフトラベル辞書を保存しました: {soft_label_path}")

# --- スクリプト実行 ---
if __name__ == "__main__":
    # IEMOCAP_full_release ディレクトリがこのスクリプトと同じ階層にあると仮定
    # 出力先は 'output_data' ディレクトリ
    create_label_files(
        iemo_root='IEMOCAP_full_release', 
        output_dir='output_data'
    )