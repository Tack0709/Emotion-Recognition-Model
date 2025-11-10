import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import json
import numpy as np
import random
from pandas import DataFrame

# ERC-SLT22のラベル定義
# 5クラス: neu, hap/exc, sad, ang, oth
# 'xxx' (インデックス5) は除外対象
LABEL_MAP = {
    'neu': 0,
    'hap': 1,
    'exc': 1, # happyに統合
    'sad': 2,
    'ang': 3,
    'oth': 4
}
# 'xxx' は スキップ

class MultimodalDAGDataset(Dataset):

    def __init__(self, dataset_name='IEMOCAP', split='train', speaker_vocab=None, args=None):
        self.speaker_vocab = speaker_vocab
        self.args = args
        self.data = self.read(dataset_name, split)
        self.len = len(self.data)

    def read(self, dataset_name, split):
        # ERC-SLT22 のように、NPYファイルから読み込むことを想定
        # (注: ここでは DAG-ERC のJSON読み込み を修正して適応)
        
        # 1. 特徴量ファイル (JSON)
        with open('../data/%s/%s_data_roberta.json.feature'%(dataset_name, split), encoding='utf-8') as f:
            raw_data_text = json.load(f) # DAG-ERCのテキスト特徴量
            
        # (仮) 音声特徴量も同様にJSONから読み込むと仮定 (実際はNPYかもしれない)
        # ここでは text 特徴量を audio 特徴量としてダミーで使用する
        # 実際には '../data/%s/%s_data_wav2vec.json.feature' などを読み込む
        with open('../data/%s/%s_data_roberta.json.feature'%(dataset_name, split), encoding='utf-8') as f:
            raw_data_audio = json.load(f) 
            
        # 2. ソフトラベルファイル (ERC-SLT22 準拠のNPY)
        soft_labels_all = np.load('../data/%s/IEMOCAP-softlabel-sum.npy' % dataset_name, allow_pickle=True).item()
        
        # 3. ハードラベル (xxx 除外のため)
        hard_labels_all = np.load('../data/%s/IEMOCAP-hardlabel.npy' % dataset_name, allow_pickle=True).item()


        dialogs = []
        # テキストと音声のデータを dialogue ID で紐付ける (ここでは同一と仮定)
        for d_text, d_audio in zip(raw_data_text, raw_data_audio):
            utterances = []
            labels_soft = []
            labels_hard_check = [] # 'xxx' チェック用
            speakers = []
            features_text = []
            features_audio = []

            valid_dialog = True
            for i, (u_text, u_audio) in enumerate(zip(d_text, d_audio)):
                utt_name = u_text['id'] # (仮) 発話IDが 'id' にあると仮定
                
                # 'xxx' (インデックス5) を除外
                hard_label_idx = hard_labels_all.get(utt_name, -1)
                if hard_label_idx == 5: # 5 は 'xxx'
                    continue 

                # ソフトラベル取得 (5クラス)
                soft_label_vec = soft_labels_all.get(utt_name)
                if soft_label_vec is None:
                    soft_label_vec = np.array([0.0] * 5)
                    label_sum = 0
                else:
                    label_sum = np.sum(soft_label_vec)

                # 合計1の分布に正規化
                if label_sum > 0:
                    soft_label_dist = soft_label_vec / label_sum
                else:
                    # ラベルがない場合はパディング（-1）と同様の扱い（後でマスク）
                    # ここでは [0,0,0,0,0] とし、ハードラベル側で-1にする
                    soft_label_dist = np.array([0.0] * 5)
                
                # パディングラベル（-1）の処理 (DAG-ERC 準拠)
                # 元のコードではハードラベルを 'stoi' で変換している
                # ここではソフトラベルを使うため、ハードラベルが-1 (ラベル無し) の場合のみソフトラベルもマスク対象（-1）とする
                if hard_label_idx == -1: # ラベル無し
                   labels_soft.append(np.array([-1.0] * 5)) # マスク用
                else:
                   labels_soft.append(soft_label_dist)

                utterances.append(u_text['text'])
                speakers.append(self.speaker_vocab['stoi'][u_text['speaker']])
                features_text.append(u_text['cls']) # (B, D_t)
                features_audio.append(u_audio['cls']) # (B, D_a) (ダミー)

            if len(utterances) > 0:
                dialogs.append({
                    'utterances': utterances,
                    'labels_soft': labels_soft,
                    'speakers': speakers,
                    'features_text': features_text,
                    'features_audio': features_audio
                })
                
        random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]['features_text']), \
               torch.FloatTensor(self.data[index]['features_audio']), \
               torch.FloatTensor(self.data[index]['labels_soft']), \
               self.data[index]['speakers'], \
               len(self.data[index]['labels_soft']), \
               self.data[index]['utterances']

    def __len__(self):
        return self.len

    # get_adj_v1 と get_s_mask は DAG-ERC からそのままコピー
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
        # data: (features_text, features_audio, labels_soft, speakers, length, utterances)
        max_dialog_len = max([d[4] for d in data])
        
        features_text = pad_sequence([d[0] for d in data], batch_first = True) # (B, N, D_t)
        features_audio = pad_sequence([d[1] for d in data], batch_first = True) # (B, N, D_a)
        
        # ソフトラベルのパディング (B, N, C)
        # パディング値は -1.0 とする
        labels_soft = pad_sequence([d[2] for d in data], batch_first = True, padding_value = -1.0) # (B, N, C)

        adj = self.get_adj_v1([d[3] for d in data], max_dialog_len)
        s_mask, s_mask_onehot = self.get_s_mask([d[3] for d in data], max_dialog_len)
        lengths = torch.LongTensor([d[4] for d in data])
        speakers = pad_sequence([torch.LongTensor(d[3]) for d in data], batch_first = True, padding_value = -1)
        utterances = [d[5] for d in data]

        return features_text, features_audio, labels_soft, adj, s_mask, s_mask_onehot, lengths, speakers, utterances