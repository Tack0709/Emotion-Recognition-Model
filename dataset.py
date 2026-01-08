import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import numpy as np
import random
import os

class MultimodalDAGDataset(Dataset):
    def __init__(self, split='train', speaker_vocab=None, args=None, data_dir='output_data', dev_ratio=0.1, test_session=5, nma=False):
        self.speaker_vocab = speaker_vocab
        self.args = args
        self.nma = nma
        # self.nma_only = nma_only  <-- 削除: 常に全データをロードするため不要
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

        print(f"Loading {split} data (NMA flag={self.nma})...")

        self.bert_features = np.load(os.path.join(data_dir, 'bert-base-diag.npy'), allow_pickle=True).item()
        self.w2v2_features = np.load(os.path.join(data_dir, 'w2v2-ft-diag.npy'), allow_pickle=True).item()
        self.soft_labels = np.load(os.path.join(data_dir, 'IEMOCAP-softlabel-sum.npy'), allow_pickle=True).item()
        self.hard_labels = np.load(os.path.join(data_dir, 'IEMOCAP-hardlabel.npy'), allow_pickle=True).item()
        
        with open(os.path.join(data_dir, 'order.pkl'), 'rb') as f:
            self.order = pickle.load(f)
        
        text_path = os.path.join(data_dir, 'text_dict.pkl')
        if os.path.exists(text_path):
            with open(text_path, 'rb') as f:
                self.text_dict = pickle.load(f)
        else:
            self.text_dict = {}

        all_session_ids = sorted(list(self.order.keys()))
        train_sessions = []
        test_sessions = []
        
        target_test_prefix = f"Ses0{test_session}"

        for ses_id in all_session_ids:
            if ses_id.startswith(target_test_prefix): 
                test_sessions.append(ses_id)
            else:
                train_sessions.append(ses_id)
        
        random.seed(args.seed if args else 42)
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

    def __getitem__(self, index):
        ses_id = self.session_list[index]
        utt_ids = self.order[ses_id]

        features_text_list = []
        features_audio_list = []
        labels_soft_list = []
        speakers_list = []
        utterances_list = []
        hard_labels_list = [] # ★追加

        for utt_id in utt_ids:
            hard_label = self.hard_labels.get(utt_id, -1)
            
            # ==========================================================
            # ★修正: フィルタリングを削除し、全てのデータをロードする
            # ==========================================================
            # if not self.nma and hard_label == 5: continue
            # if self.nma_only and hard_label != 5: continue
            # ==========================================================

            bert_feat = self.bert_features.get(utt_id)
            w2v2_feat = self.w2v2_features.get(utt_id)
            
            if bert_feat is None or w2v2_feat is None:
                continue

            features_text_list.append(bert_feat)
            features_audio_list.append(w2v2_feat)

            # ソフトラベルの処理
            soft_label_vec = self.soft_labels.get(utt_id)
            if soft_label_vec is None or hard_label == -1:
                labels_soft_list.append(np.array([-1.0] * 5))
            else:
                label_sum = np.sum(soft_label_vec)
                if label_sum > 0:
                    labels_soft_list.append(soft_label_vec / label_sum)
                else:
                    labels_soft_list.append(np.array([-1.0] * 5))
            
            speaker_char = utt_id.split('_')[-1][0]
            speakers_list.append(self.speaker_vocab['stoi'].get(speaker_char, 0))

            utterances_list.append(self.text_dict.get(utt_id, ""))
            hard_labels_list.append(hard_label) # ★追加

        if not speakers_list:
             return None 

        return torch.FloatTensor(np.array(features_text_list)), \
               torch.FloatTensor(np.array(features_audio_list)), \
               torch.FloatTensor(np.array(labels_soft_list)), \
               speakers_list, \
               len(speakers_list), \
               utterances_list, \
               torch.LongTensor(np.array(hard_labels_list)) # ★追加: ハードラベルを返す

    def __len__(self):
        return self.len

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
        batch = [d for d in batch if d is not None]
        if not batch:
            return None, None, None, None, None, None, None, None, None, None # +1 for hard_labels

        max_dialog_len = max([d[4] for d in batch])
        
        features_text = pad_sequence([d[0] for d in batch], batch_first = True)
        features_audio = pad_sequence([d[1] for d in batch], batch_first = True)
        labels_soft = pad_sequence([d[2] for d in batch], batch_first = True, padding_value = -1.0)

        adj = self.get_adj_v1([d[3] for d in batch], max_dialog_len)
        s_mask, s_mask_onehot = self.get_s_mask([d[3] for d in batch], max_dialog_len)
        
        lengths = torch.LongTensor([d[4] for d in batch])
        speakers = pad_sequence([torch.LongTensor(d[3]) for d in batch], batch_first = True, padding_value = -1)
        utterances = [d[5] for d in batch]
        
        # ★追加: ハードラベルのパディング（パディング値は -1 とする）
        hard_labels = pad_sequence([d[6] for d in batch], batch_first = True, padding_value = -1)

        return features_text, features_audio, labels_soft, adj, s_mask, s_mask_onehot, lengths, speakers, utterances, hard_labels