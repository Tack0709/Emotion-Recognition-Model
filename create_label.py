import os
import argparse
import numpy as np
import sys
from collections import defaultdict

def create_label_files(iemo_root='IEMOCAP_full_release/', output_dir='output_data'):
    """
    IEMOCAPのEmoEvaluationファイル を解析し、
    ハードラベルとソフトラベルの辞書を作成して.npyファイルとして保存します。
    
    【修正版仕様 v4】
    - タグ解析のバグ修正: 複数のタグがある場合(Frustration; Disgust; ...)も全て取得して集計する
    - カウント方法: 正規化せず、タグの数だけ票を加算する (例: 3タグなら3票)
    - ハードラベル: 集計後のソフトラベルに基づいて「最多得票（相対多数）」で決定する
    """
    
    # --- 1. ディレクトリ設定 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_roots = [
        os.path.join(iemo_root, f"Session{i}/dialog/EmoEvaluation") 
        for i in range(1, 6)
    ]
    
    # すべての EmoEvaluation ファイルパスを収集
    file_paths = []
    for file_dir in file_roots:
        if not os.path.exists(file_dir):
            print(f"Warning: ディレクトリが見つかりません: {file_dir}")
            continue
        for files in os.listdir(file_dir):
            if files.startswith('._'):
                continue
            if files.endswith('.txt'):
                file_paths.append(os.path.join(file_dir, files))
    
    print(f"合計 {len(file_paths)} 個の EmoEvaluation ファイルを処理します。")
    if not file_paths:
        print(f"エラー: {iemo_root} にデータが見つかりません。")
        return

    # --- 2. ラベルマッピング定義 (ERC-SLT22 準拠) ---
    
    # 個別評価の集約用
    mapping = {
        'Neutral':   [1, 0, 0, 0, 0],
        'Happiness': [0, 1, 0, 0, 0],
        'Excited':   [0, 1, 0, 0, 0], # Happinessに統合
        'Sadness':   [0, 0, 1, 0, 0],
        'Anger':     [0, 0, 0, 1, 0]
    }
    OTHER_VECTOR = [0, 0, 0, 0, 1] # 上記以外はすべてOtherへ
    
    # 最終的なクラスID
    # 0: neu, 1: hap, 2: sad, 3: ang, 4: oth, 5: xxx

    # --- 3. ファイル解析とデータ抽出 ---
    soft_label_dic = {}
    # 元の評価タグごとの投票数と、そのタグが1回以上付いた発話数を集計する。
    # 同じ発話に複数評価者が同じタグを付けた場合、投票数には複数回、発話数には1回計上。
    raw_tag_vote_counts = defaultdict(int)
    raw_tag_utterances = defaultdict(set)
    
    current_utt_id = None
    current_soft_labels = []

    # すべての EmoEvaluation ファイルをループ
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
                            # ソフトラベルの集計と保存
                            if current_soft_labels:
                                soft_label_dic[current_utt_id] = np.array(current_soft_labels).sum(axis=0)
                            else:
                                soft_label_dic[current_utt_id] = np.array([0, 0, 0, 0, 0])

                        # --- 新しい発話データを初期化 ---
                        parts = line.split()
                        if len(parts) > 4:
                            current_utt_id = parts[3]       # 例: Ses01F_impro01_F000
                            current_soft_labels = []        
                        else:
                            current_utt_id = None 
                    
                    # --- C-E行（第三者評価）の解析 ---
                    elif line.startswith('C-E') and current_utt_id:
                        # ★修正箇所: split()による分割ミスを防ぐため、記号位置で切り出す
                        start_idx = line.find(':')
                        end_idx = line.rfind('(')
                        
                        if start_idx != -1 and end_idx != -1:
                            # コロンの後ろ〜カッコの前までを取得 (例: " Frustration; Disgust; Anger; ")
                            tag_part = line[start_idx+1 : end_idx].strip()
                            
                            # セミコロン区切りでタグリスト化
                            emotion_tags = tag_part.split(';')
                            
                            for tag in emotion_tags:
                                tag = tag.strip()
                                if tag:
                                    raw_tag_vote_counts[tag] += 1
                                    raw_tag_utterances[tag].add(current_utt_id)
                                    # Otherへの集約と追加 (1タグにつき1ベクトル追加)
                                    vec = mapping.get(tag, OTHER_VECTOR)
                                    current_soft_labels.append(vec)

        except Exception as e:
            print(f"エラー: ファイル {label_path} の処理中に問題が発生しました: {e}")
            sys.exit(1)


    # --- 最後の発話データを保存 ---
    if current_utt_id:
        if current_soft_labels:
            soft_label_dic[current_utt_id] = np.array(current_soft_labels).sum(axis=0)
        else:
            soft_label_dic[current_utt_id] = np.array([0, 0, 0, 0, 0])

    # =============================================================================
    # 集約済みソフトラベルからハードラベルを再計算する (相対多数決・同率1位はXXX)
    # =============================================================================
    print("ハードラベルを再計算しています（タグ全加算 + 相対多数決）...")
    
    final_hard_label_dic = {}
    recalc_count = 0
    xxx_count = 0

    for utt_id, vote_vector in soft_label_dic.items():
        # vote_vector: [neu, hap, sad, ang, oth] の投票数 (例: [0, 0, 0, 1, 4])
        
        total_votes = np.sum(vote_vector)
        
        if total_votes == 0:
            final_hard_label = 5 # XXX
        else:
            max_votes = np.max(vote_vector)
            # 最大得票のインデックス一覧
            winners = np.where(vote_vector == max_votes)[0] 
            
            # 条件: 単独1位であれば採用 (例: Otherが4票で1位ならOther採用)
            if len(winners) == 1:
                final_hard_label = int(winners[0]) # 0~4
            else:
                # 同数1位がいる場合
                final_hard_label = 5 
        
        final_hard_label_dic[utt_id] = final_hard_label
        
        if final_hard_label == 5:
            xxx_count += 1
        recalc_count += 1

    # =============================================================================

    # --- 4. ファイル保存 ---
    hard_label_path = os.path.join(output_dir, 'IEMOCAP-hardlabel.npy')
    soft_label_path = os.path.join(output_dir, 'IEMOCAP-softlabel-sum.npy')

    try:
        np.save(hard_label_path, final_hard_label_dic)
        np.save(soft_label_path, soft_label_dic)
    except Exception as e:
        print(f"エラー: ファイルの保存中に問題が発生しました: {e}")
        sys.exit(1)

    print(f"処理完了。総発話数: {len(final_hard_label_dic)}")
    print(f"  うち XXX (評価不一致) 数: {xxx_count} ({(xxx_count/len(final_hard_label_dic))*100:.2f}%)")

    # --- 5. クラス別の発話数・割合と元タグ別の内訳を表示 ---
    class_names = ['Neutral', 'Happiness (+ Excited)', 'Sadness', 'Anger', 'Other', 'XXX']
    class_counts = np.bincount(list(final_hard_label_dic.values()), minlength=6)
    total_utterances = len(final_hard_label_dic)
    print("\n[ハードラベル別 発話数]")
    for class_id, class_name in enumerate(class_names):
        count = int(class_counts[class_id])
        ratio = count / total_utterances * 100 if total_utterances else 0.0
        print(f"  {class_name}: {count} 発話 ({ratio:.2f}%)")

    total_tag_votes = sum(raw_tag_vote_counts.values())
    other_tag_names = [tag for tag in raw_tag_vote_counts if tag not in mapping]
    other_tag_votes = sum(raw_tag_vote_counts[tag] for tag in other_tag_names)
    print("\n[元の評価タグ別 内訳]")
    print("  ※投票数は評価タグの出現回数、発話数はそのタグが付いたユニークな発話数")
    for tag in sorted(raw_tag_vote_counts):
        count = raw_tag_vote_counts[tag]
        vote_ratio = count / total_tag_votes * 100 if total_tag_votes else 0.0
        utterance_count = len(raw_tag_utterances[tag])
        utterance_ratio = utterance_count / total_utterances * 100 if total_utterances else 0.0
        group = 'Other' if tag in other_tag_names else ('Excited (Happinessへ統合)' if tag == 'Excited' else tag)
        print(f"  {tag} [{group}]: {count} 票 ({vote_ratio:.2f}%), "
              f"{utterance_count} 発話 ({utterance_ratio:.2f}%)")
    other_ratio = other_tag_votes / total_tag_votes * 100 if total_tag_votes else 0.0
    print(f"  Otherに集約される元タグ合計: {other_tag_votes} 票 ({other_ratio:.2f}%)")

    # ソフトラベルの分布エントロピー（自然対数、単位: nats）を帯ごとに集計。
    # 5クラスの最大値は ln(5) ≈ 1.609 nats。投票がない発話はエントロピーを定義できないため別計上。
    entropy_band_labels = ['0.0–<0.4', '0.4–<0.8', '0.8–<1.2', '1.2–ln(5)']
    entropy_band_counts = [0, 0, 0, 0]
    valid_entropy_count = 0
    for vote_vector in soft_label_dic.values():
        total_votes = float(np.sum(vote_vector))
        if total_votes <= 0:
            continue
        probs = np.asarray(vote_vector, dtype=float) / total_votes
        entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
        band_index = 0 if entropy < 0.4 else 1 if entropy < 0.8 else 2 if entropy < 1.2 else 3
        entropy_band_counts[band_index] += 1
        valid_entropy_count += 1

    no_vote_count = total_utterances - valid_entropy_count
    print("\n[ソフトラベルのエントロピー帯別 発話数]")
    print("  ※エントロピーはnats。割合の分母は投票がありエントロピーを計算できる発話数")
    for band_label, count in zip(entropy_band_labels, entropy_band_counts):
        ratio = count / valid_entropy_count * 100 if valid_entropy_count else 0.0
        print(f"  {band_label}: {count} 発話 ({ratio:.2f}%)")
    no_vote_ratio = no_vote_count / total_utterances * 100 if total_utterances else 0.0
    print(f"  投票なし（計算対象外）: {no_vote_count} 発話 ({no_vote_ratio:.2f}% of all utterances)")

    print(f"  ➡️  {hard_label_path}")
    print(f"  ➡️  {soft_label_path}")
    
    if len(final_hard_label_dic) != 10039:
        print(f"Warning: 期待される発話数(10039)と一致しません。 検出数: {len(final_hard_label_dic)}")

# --- スクリプト実行 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create IEMOCAP hard and soft labels.')
    parser.add_argument('--iemo_root', default='IEMOCAP_full_release', help='IEMOCAP_full_release directory')
    parser.add_argument('--output_dir', default='output_data')
    args = parser.parse_args()
    create_label_files(iemo_root=args.iemo_root, output_dir=args.output_dir)
