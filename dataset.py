import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import json
import numpy as np
import random
import os

class MultimodalDAGDataset(Dataset):
    """
    固定分割 (Ses1-4: Train/Dev, Ses5: Test) を行うデータセット。
    .scp ファイル を使用せず、order.pkl のキー（セッションID）に基づいて動的に分割する。
    """
    def __init__(self, split='train', speaker_vocab=None, args=None, data_dir='output_data', dev_ratio=0.1):
        self.speaker_vocab = speaker_vocab
        self.args = args
        datapath = data_dir
        
        # 修正: データディレクトリの存在確認を追加
        if not os.path.exists(data_dir): # <--- 追加
            raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

        print(f"Loading {split} data from {data_dir}...")

        # 1. IDをキーとする辞書を先に読み込む
        # すべて各発話IDをキーにした特徴量・ラベル辞書
        self.bert_features = np.load(os.path.join(datapath, 'bert-base-diag.npy'), allow_pickle=True).item()
        self.w2v2_features = np.load(os.path.join(datapath, 'w2v2-ft-diag.npy'), allow_pickle=True).item()
        self.soft_labels = np.load(os.path.join(datapath, 'IEMOCAP-softlabel-sum.npy'), allow_pickle=True).item()
        self.hard_labels = np.load(os.path.join(datapath, 'IEMOCAP-hardlabel.npy'), allow_pickle=True).item()

        # 発話IDの順序 (セッションのスクリプトIDがキーの辞書)
        order_path = os.path.join(datapath, 'order.pkl') 
        with open(order_path, 'rb') as f: 
            self.order = pickle.load(f) # .pklからロード
            
        # 発話テキスト (各発話IDがキーの辞書) 一応発話順になっている
        text_dict_path = os.path.join(datapath, 'text_dict.pkl')
        if os.path.exists(text_dict_path):
             with open(text_dict_path, 'rb') as f:
                self.text_dict = pickle.load(f)
        else:
            self.text_dict = {}

        # #############################################################
        # ⬇️ 変更点: .scp 読み込みを削除し、固定分割ロジックを実装
        # #############################################################
        
        # 2. self.order のキー（セッションのスクリプトID）に基づいてデータを分割
        all_session_ids = sorted(list(self.order.keys()))
        train_sessions = []
        dev_sessions = []
        test_sessions = []

        # 各スクリプトIDを確認し、セッション5はtest、1-4はtrain/devに振り分け
        for ses_id in all_session_ids:
            if ses_id.startswith('Ses05'): # Session 5 は Test
                test_sessions.append(ses_id)
            elif ses_id.startswith(('Ses01', 'Ses02', 'Ses03', 'Ses04')): # Session 1-4 は Train/Dev
                train_sessions.append(ses_id)
        
        # Session 1-4 から dev_ratio (例: 1割) をランダムに dev に振り分け
        # (注: 元の ERC-SLT22 は 'cv' (検証) 用に 1/5 を固定で分けていた)
        # 訓練データをシャッフルしてから分割
        # スクリプトごとだから会話途中の発話が抜けるとかはない
        random.shuffle(train_sessions)
        dev_size = int(len(train_sessions) * dev_ratio)
        dev_sessions = train_sessions[:dev_size]
        train_sessions = train_sessions[dev_size:]

        if split == 'train':
            self.session_list = train_sessions
        elif split == 'dev':
            self.session_list = dev_sessions
        elif split == 'test':
            self.session_list = test_sessions
        else:
            raise ValueError(f"Unknown split name: {split}")
        
        self.len = len(self.session_list)
        print(f"Loaded {self.len} dialogues for {split}.")
        # #############################################################


    def __getitem__(self, index):
        # 1. セッションIDを取得
        # (変更点: .scp ではなく、内部リスト self.session_list から取得)
        # session_listはスクリプトのIDリスト
        # train/dev/test に応じたスクリプトIDが入っている
        ses_id = self.session_list[index]

        # 2. スクリプト内の各発話IDのリストを取得
        # (変更点: start=0, end=max でセッション全体を取得)
        utt_ids = self.order[ses_id] 

        features_text_list = []
        features_audio_list = []
        labels_soft_list = []
        speakers_list = []
        utterances_list = []

        # 3. 各発話IDをキーにしてデータを収集 (処理は同一)
        for utt_id in utt_ids:
            
            # 3a. 'xxx' (hard_label==5) を除外 
            hard_label = self.hard_labels.get(utt_id, -1)
            if hard_label == 5:
                continue

            # 3b. 特徴量
            bert_feat = self.bert_features.get(utt_id)
            w2v2_feat = self.w2v2_features.get(utt_id)
            
            if bert_feat is None or w2v2_feat is None:
                continue

            features_text_list.append(bert_feat)
            features_audio_list.append(w2v2_feat)

            # 3c. ソフトラベル (分布に正規化)
            soft_label_vec = self.soft_labels.get(utt_id)
            if soft_label_vec is None or hard_label == -1: 
                labels_soft_list.append(np.array([-1.0] * 5)) 
            else:
                label_sum = np.sum(soft_label_vec)
                if label_sum > 0:
                    labels_soft_list.append(soft_label_vec / label_sum)
                else: 
                    labels_soft_list.append(np.array([-1.0] * 5)) 
            
            # 3d. 話者ID/性別 (発話ID '..._F000' や '..._M000' から抽出)
            speaker_char = utt_id.split('_')[-1][0] # 'F' or 'M'
            # 話者語彙に基づいてインデックス化
            speakers_list.append(self.speaker_vocab['stoi'].get(speaker_char, 0)) 

            # 3e. 発話テキスト
            # 辞書内にその発話IDがなければ空文字列を追加
            utterances_list.append(self.text_dict.get(utt_id, ""))

        if not speakers_list:
             return torch.FloatTensor([]), torch.FloatTensor([]), torch.FloatTensor([]), [], 0, []

        return torch.FloatTensor(np.array(features_text_list)), \
               torch.FloatTensor(np.array(features_audio_list)), \
               torch.FloatTensor(np.array(labels_soft_list)), \
               speakers_list, \
               len(speakers_list), \
               utterances_list

    def __len__(self):
        return self.len

    # --- グラフ構築関数 (DAG-ERC と同一) ---
    
    def get_adj_v1(self, speakers, max_dialog_len):
        # max_dialog_len: バッチ内(各スクリプトの発話数の)の最大発話数
        adj = []
        # バッチサイズが16ならspeakersは16個のリストを持つ（16個のスクリプト分の話者IDリスト）
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker): # i:現在の発話番号, s:現在の話者
                cnt = 0
                # 過去に遡るループ (j = i-1, i-2, ... 0)
                for j in range(i - 1, -1, -1): 
                    # 【接続】過去の発話jと現在の発話iをつなぐ
                    a[i,j] = 1
                    if speaker[j] == s: # もし過去の発話jが「現在の発話と同じ話者」だったら
                        cnt += 1
                        if cnt==self.args.windowp: # 同一話者の過去発話はwindowp個まで接続
                            break
            #adj: (B, N, N) のテンソルにする (B:バッチサイズ, N:最大発話数)
            #各バッチサイズごとに (N, N) の隣接行列を持つ
            adj.append(a)
        return torch.stack(adj)

    def get_s_mask(self, speakers, max_dialog_len):
        # つながったエッジが「自分自身との関係（Intra）」なのか「他者との関係（Inter）」なのかを区別するためのラベル（マスク）を作る
        s_mask = []
        s_mask_onehot = []
        # バッチサイズが16ならspeakersは16個のリストを持つ（16個のスクリプト分の話者IDリスト）
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype = torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            # 発話iと発話jの話者が同じなら s[i,j]=1, 異なるなら0
            # 全ての発話ペア (i, j) について総当たり
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i,j] = 1
                        s_onehot[i,j,1] = 1
                    else:
                        s_onehot[i,j,0] = 1
            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)


    def collate_fn(self, batch):
        # (DAG-ERC の collate_fn とほぼ同一)
        # これ呼び出されるとデータローダーがバッチを作成する
        # batch: __getitem__ から返された「1会話分のデータ」のリスト
        # train, dev, test 用に分割されたスクリプトIDごとのデータ
        # [ (feat_text, feat_audio, label, speakers, len, text), ... ]
        # ここでミニバッチ内（一つのスクリプトID）の各会話データをパディングしてテンソル化する 
        # 発話数が0の会話を除外 d[4]：発話数分の特徴量
        batch = [d for d in batch if d[4] > 0]
        if not batch:
            return None, None, None, None, None, None, None, None, None

        # 2. 最大長の取得
        # このバッチの中で「一番長い会話の発話数」を調べます
        max_dialog_len = max([d[4] for d in batch])
        
        # 3. パディング処理 (pad_sequence)
        # リスト状のデータを、max_dialog_len に合わせてパディングし、Tensorに変換します
        # batch_first=True: (バッチサイズ, シーケンス長, 特徴量次元) の形にする
        
        # テキスト特徴量 (パディング: 0)
        features_text = pad_sequence([d[0] for d in batch], batch_first = True) 
        # 音声特徴量 (パディング: 0)
        features_audio = pad_sequence([d[1] for d in batch], batch_first = True) 
        # ソフトラベル (パディング: -1.0)
        # ★重要: 損失計算の際、この -1.0 を見て「ここは計算しない」と判断します
        labels_soft = pad_sequence([d[2] for d in batch], batch_first = True, padding_value = -1.0) 

        # 4. グラフ構造の構築 (ここがGNN特有！)
        # パディング後の長さ (max_dialog_len) に合わせた隣接行列を作ります
        
        # 隣接行列 (adj): 「どの発話からどの発話へ情報を流すか」
        # d[3]: 話者IDリスト
        adj = self.get_adj_v1([d[3] for d in batch], max_dialog_len)
        # 話者マスク (s_mask): 「同一話者か、別話者か」
        s_mask, s_mask_onehot = self.get_s_mask([d[3] for d in batch], max_dialog_len)
        
        # 発話ごとの長さ情報
        lengths = torch.LongTensor([d[4] for d in batch])
        
        # 話者ID (パディング: -1)
        speakers = pad_sequence([torch.LongTensor(d[3]) for d in batch], batch_first = True, padding_value = -1)
        
        # 発話テキスト (リストのまま)
        utterances = [d[5] for d in batch]

        return features_text, features_audio, labels_soft, adj, s_mask, s_mask_onehot, lengths, speakers, utterances