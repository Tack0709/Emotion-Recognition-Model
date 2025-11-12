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
    def __init__(self, dataset_name='IEMOCAP', split='train', speaker_vocab=None, args=None, dev_ratio=0.1):
        self.speaker_vocab = speaker_vocab
        self.args = args
        datapath = os.path.join('../data', dataset_name) 

        print(f"Loading {split} data...")

        # 1. IDをキーとする辞書を先に読み込む
        self.bert_features = np.load(os.path.join(datapath, 'bert-base-diag.npy'), allow_pickle=True).item()
        self.w2v2_features = np.load(os.path.join(datapath, 'w2v2-ft-diag.npy'), allow_pickle=True).item()
        self.soft_labels = np.load(os.path.join(datapath, 'IEMOCAP-softlabel-sum.npy'), allow_pickle=True).item()
        self.hard_labels = np.load(os.path.join(datapath, 'IEMOCAP-hardlabel.npy'), allow_pickle=True).item()

        # 発話IDの順序 (セッションIDがキーの辞書)
        order_path = os.path.join(datapath, 'order.pkl') 
        with open(order_path, 'rb') as f: 
            self.order = pickle.load(f) # .pklからロード
            
        # 発話テキスト (発話IDがキーの辞書)
        text_dict_path = os.path.join(datapath, 'text_dict.pkl')
        if os.path.exists(text_dict_path):
             with open(text_dict_path, 'rb') as f:
                self.text_dict = pickle.load(f)
        else:
            self.text_dict = {}

        # #############################################################
        # ⬇️ 変更点: .scp 読み込みを削除し、固定分割ロジックを実装
        # #############################################################
        
        # 2. self.order のキー（セッションID）に基づいてデータを分割
        all_session_ids = sorted(list(self.order.keys()))
        train_sessions = []
        dev_sessions = []
        test_sessions = []

        for ses_id in all_session_ids:
            if ses_id.startswith('Ses05'): # Session 5 は Test
                test_sessions.append(ses_id)
            elif ses_id.startswith(('Ses01', 'Ses02', 'Ses03', 'Ses04')): # Session 1-4 は Train/Dev
                train_sessions.append(ses_id)
        
        # Session 1-4 から dev_ratio (例: 1割) をランダムに dev に振り分け
        # (注: 元の ERC-SLT22 は 'cv' (検証) 用に 1/5 を固定で分けていた)
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
        ses_id = self.session_list[index]

        # 2. 発話IDのリストを取得
        # (変更点: start=0, end=max でセッション全体を取得)
        utt_ids = self.order[ses_id] 

        features_text_list = []
        features_audio_list = []
        labels_soft_list = []
        speakers_list = []
        utterances_list = []

        # 3. 発話IDをキーにしてデータを収集 (処理は同一)
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
            speakers_list.append(self.speaker_vocab['stoi'].get(speaker_char, 0)) 

            # 3e. 発話テキスト
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
        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                cnt = 0
                for j in range(i - 1, -1, -1): 
                    a[i,j] = 1
                    if speaker[j] == s: 
                        cnt += 1
                        if cnt==self.args.windowp:
                            break
            adj.append(a)
        return torch.stack(adj)

    def get_s_mask(self, speakers, max_dialog_len):
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype = torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
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
        batch = [d for d in batch if d[4] > 0]
        if not batch:
            return None, None, None, None, None, None, None, None, None

        max_dialog_len = max([d[4] for d in batch])
        
        features_text = pad_sequence([d[0] for d in batch], batch_first = True) 
        features_audio = pad_sequence([d[1] for d in batch], batch_first = True) 
        labels_soft = pad_sequence([d[2] for d in batch], batch_first = True, padding_value = -1.0) 

        adj = self.get_adj_v1([d[3] for d in batch], max_dialog_len)
        s_mask, s_mask_onehot = self.get_s_mask([d[3] for d in batch], max_dialog_len)
        
        lengths = torch.LongTensor([d[4] for d in batch])
        speakers = pad_sequence([torch.LongTensor(d[3]) for d in batch], batch_first = True, padding_value = -1)
        utterances = [d[5] for d in batch]

        return features_text, features_audio, labels_soft, adj, s_mask, s_mask_onehot, lengths, speakers, utterances