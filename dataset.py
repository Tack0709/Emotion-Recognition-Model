import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import json
import numpy as np
import random
from pandas import DataFrame
import os # os.path を使用

class MultimodalDAGDataset(Dataset):

    def __init__(self, dataset_name='IEMOCAP', split='train', speaker_vocab=None, args=None):
        self.speaker_vocab = speaker_vocab
        self.args = args
        datapath = os.path.join('../data', dataset_name)

        # 1. ERC-SLT22 スタイルで、IDをキーとする辞書を先に読み込む
        # (注: DAG-ERC の read() とは異なる)
        
        # 特徴量 (発話IDがキーの辞書)
        self.bert_features = np.load(os.path.join(datapath, 'bert-base-diag.npy'), allow_pickle=True).item()
        self.w2v2_features = np.load(os.path.join(datapath, 'w2v2-ft-diag.npy'), allow_pickle=True).item()
        
        # ラベル (発話IDがキーの辞書)
        self.soft_labels = np.load(os.path.join(datapath, 'IEMOCAP-softlabel-sum.npy'), allow_pickle=True).item()
        self.hard_labels = np.load(os.path.join(datapath, 'IEMOCAP-hardlabel.npy'), allow_pickle=True).item()

        # 発話IDの順序 (セッションIDがキーの辞書)
        with open(os.path.join(datapath, 'order.json'), 'r') as f:
            self.order = json.load(f)
            
        # (新規) 発話テキスト (発話IDがキーの辞書)
        # (注: この 'text_dict.pkl' は別途作成されている前提)
        with open(os.path.join(datapath, 'text_dict.pkl'), 'rb') as f:
            self.text_dict = pickle.load(f)

        # 2. 処理対象のセッションリストを .scp ファイルから読み込む
        scp_path = os.path.join(datapath, f'iemocap_diag_{split}.scp') # 例: iemocap_diag_train.scp
        if not os.path.exists(scp_path):
             # .scp がなければ all (DAG-ERC の split 名に近い)
             scp_path = os.path.join(datapath, f'iemocap_diag_{split}-all.scp') 
        
        self.scp_file = pd.read_csv(scp_path, delimiter='\t', header=0).values.tolist()
        
        self.len = len(self.scp_file)

    def __getitem__(self, index):
        # 1. セッションID と範囲を取得 (ERC-SLT22 方式)
        ses_id, start, end = self.scp_file[index]

        # 2. 発話IDのリストを取得
        utt_ids = self.order[ses_id][start:end]

        features_text_list = []
        features_audio_list = []
        labels_soft_list = []
        speakers_list = []
        utterances_list = []

        # 3. 発話IDをキーにしてデータを収集
        for utt_id in utt_ids:
            
            # 3a. 'xxx' (hard_label==5) を除外
            hard_label = self.hard_labels.get(utt_id, -1)
            if hard_label == 5:
                continue

            # 3b. 特徴量
            features_text_list.append(self.bert_features.get(utt_id))
            features_audio_list.append(self.w2v2_features.get(utt_id))

            # 3c. ソフトラベル (分布に正規化)
            soft_label_vec = self.soft_labels.get(utt_id)
            if soft_label_vec is None or hard_label == -1: # ラベル無し
                labels_soft_list.append(np.array([-1.0] * 5)) # パディング用
            else:
                label_sum = np.sum(soft_label_vec)
                if label_sum > 0:
                    labels_soft_list.append(soft_label_vec / label_sum)
                else:
                    labels_soft_list.append(np.array([-1.0] * 5)) # パディング用
            
            # 3d. 話者ID/性別 (発話ID '..._F000' や '..._M000' から抽出)
            # (注: これは IEMOCAP の命名規則に依存)
            speaker_char = utt_id.split('_')[-1][0] # 'F' or 'M'
            speakers_list.append(self.speaker_vocab['stoi'].get(speaker_char, 0)) # 0: default

            # 3e. 発話テキスト
            utterances_list.append(self.text_dict.get(utt_id, ""))

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


    def collate_fn(self, data):
        # (DAG-ERC の collate_fn とほぼ同一)
        # data: (features_text, features_audio, labels_soft, speakers, length, utterances)
        
        max_dialog_len = max([d[4] for d in data])
        if max_dialog_len == 0: # 空のダイアログを除外
            return None, None, None, None, None, None, None, None, None
        
        features_text = pad_sequence([d[0] for d in data], batch_first = True) # (B, N, D_t)
        features_audio = pad_sequence([d[1] for d in data], batch_first = True) # (B, N, D_a)
        
        # ソフトラベルのパディング (B, N, C)
        labels_soft = pad_sequence([d[2] for d in data], batch_first = True, padding_value = -1.0) # (B, N, C)

        adj = self.get_adj_v1([d[3] for d in data], max_dialog_len)
        s_mask, s_mask_onehot = self.get_s_mask([d[3] for d in data], max_dialog_len)
        lengths = torch.LongTensor([d[4] for d in data])
        speakers = pad_sequence([torch.LongTensor(d[3]) for d in data], batch_first = True, padding_value = -1)
        utterances = [d[5] for d in data]

        return features_text, features_audio, labels_soft, adj, s_mask, s_mask_onehot, lengths, speakers, utterances