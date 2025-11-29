import os
import re
import pickle

def create_metadata_files(iemo_root='IEMOCAP_full_release', output_dir='output_data'):
    """
    IEMOCAPのtranscriptionsファイル を解析し、
    order.pkl, text_dict.pkl, speaker_vocab.pkl を作成します。
    
    処理は tack0709/erc-slt22/.../data_prep_diag_order.py に準拠し、
    テキスト抽出機能を追加しています。
    """
    
    # --- 1. ディレクトリ設定 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Session1から5までの transcriptions ディレクトリパスを生成
    trans_dir = [
        os.path.join(iemo_root, f"Session{i}/dialog/transcriptions/") 
        for i in range(1, 6)
    ]

    order_dic = {} # order.pkl 用
    text_dic = {}  # text_dict.pkl 用
    speakers = set() # speaker_vocab.pkl 用

    total_files = 0
    total_utterances = 0

    # 正規表現で行をパース
    #オブジェクト化してこれとマッチする形を抽出
    # 例: Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me.
    #\s+は空白文字の1回以上の繰り返し
    line_regex = re.compile(r'^(Ses\d{2}[FM]_[\w\d]+_[FM]\d{3})\s+\[.+?\]:\s+(.+)$')

    # --- 2. ファイル解析とデータ抽出 ---
    # transcriptions ディレクトリには発話の順番通りのテキストファイルがある
    for path in trans_dir:
        if not os.path.exists(path):
            print(f"Warning: ディレクトリが見つかりません: {path}")
            continue
            
        #listdir関数ででディレクトリ内のファイルを取得
        for file in os.listdir(path):
            # 修正: '._' で始まるファイルは無視する
            if file.startswith('._'):
                continue
            if not file.endswith('.txt'):
                continue
                
            # セッションID (ファイル名から .txt を除く)
            # 例: Ses01F_impro01
            ses_id = file.split('.')[0]
            order_dic[ses_id] = []
            total_files += 1

            # ファイルを開いて行ごと(各発話)に処理
            #path=ルートディレクトリまでのパス+file=ファイル名
            file_path = os.path.join(path, file)
            with open(file_path, 'r') as f:
                for line in f:
                    match = line_regex.match(line.strip())
                    
                    if match:
                        #マッチした時のグループを取得グループは括弧で囲まれた部分 始まりは1
                        utt_id = match.group(1)   # 例: Ses01F_impro01_F000
                        text = match.group(2)     # 例: Excuse me.
                        
                        # 発話IDに 'XX' が含まれるものは除外 (ERC-SLT22 準拠)
                        if 'XX' in utt_id:
                            continue
                        
                        # 1. order.pkl用データに追加
                        # 発話IDリストに追加
                        # 例: order_dic['Ses01F_impro01'(ファイル名)] = ['Ses01F_impro01_F000', 'Ses01F_impro01_F001', ...]
                        order_dic[ses_id].append(utt_id)
                        
                        # 2. text_dict.pkl用データに追加
                        text_dic[utt_id] = text
                        
                        # 3. speaker_vocab.pkl用データに追加 (性別 'F' or 'M')
                        speaker_char = utt_id.split('_')[-1][0] # 'F' or 'M'
                        #speakersはset型なので重複しない
                        #F,M以外の文字が入る
                        speakers.add(speaker_char)
                        
                        total_utterances += 1

    # --- 3. speaker_vocab.pkl の作成 ---
    # 'X' (不明/その他) も追加しておく
    speakers.add('X') 
    # ソートしてインデックスを割り当て ローマ字順にソートして再現性の確保
    speaker_vocab = {
        'stoi': {char: i for i, char in enumerate(sorted(list(speakers)))},
        'itos': {i: char for i, char in enumerate(sorted(list(speakers)))}
    }

    # --- 4. ファイル保存 ---
    order_path = os.path.join(output_dir, 'order.pkl')
    text_path = os.path.join(output_dir, 'text_dict.pkl')
    speaker_path = os.path.join(output_dir, 'speaker_vocab.pkl')

    with open(order_path, 'wb') as f:
        pickle.dump(order_dic, f)
        
    with open(text_path, 'wb') as f:
        pickle.dump(text_dic, f)
        
    with open(speaker_path, 'wb') as f:
        pickle.dump(speaker_vocab, f)

    print(f"処理完了。{total_files} ファイル、{total_utterances} 発話を処理しました。")
    print(f"発話順序辞書を保存しました: {order_path}")
    print(f"発話テキスト辞書を保存しました: {text_path}")
    print(f"話者語彙辞書を保存しました: {speaker_path}")
    print(f"検出された話者 (性別): {sorted(list(speakers))}")
    
    #order.pkl: {セッションID: [発話IDリスト]} 発話順
    #text_dict.pkl: {発話ID: テキスト} 発話テキスト 順序も同じ
    #speaker_vocab.pkl: {'stoi': {'F': 0, 'M': 1, 'X': 2}, 'itos': {0: 'F', 1: 'M', 2: 'X'}}

# --- スクリプト実行 ---
if __name__ == "__main__":
    # IEMOCAP_full_release ディレクトリがこのスクリプトと同じ階層にあると仮定
    # 出力先は 'output_data' ディレクトリ
    create_metadata_files(
        iemo_root=r'/home/datasets/mizuno/IEMOCAP_full_release/IEMOCAP_full_release', 
        output_dir='output_data'
    )