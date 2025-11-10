import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
from tqdm import tqdm
import json

def train_or_eval_model(model, loss_function, dataloader, epoch, cuda, args, optimizer=None, train=False):
    losses, preds, labels = [], [], []
    
    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        # 1. データのアンパック (dataset.py の collate_fn に合わせる)
        features_text, features_audio, labels_soft, adj, s_mask, s_mask_onehot, lengths, speakers, utterances = data

        if cuda:
            features_text = features_text.cuda()
            features_audio = features_audio.cuda()
            labels_soft = labels_soft.cuda()
            adj = adj.cuda()
            s_mask = s_mask.cuda()
            s_mask_onehot = s_mask_onehot.cuda()
            lengths = lengths.cuda()

        # 2. モデル実行 (log_softmax が返る)
        log_prob = model(features_text, features_audio, adj, s_mask, s_mask_onehot, lengths) # (B, N, C)
        n_classes = log_prob.size(2)

        # 3. マスク処理と損失計算 (ソフトラベル対応)
        # labels_soft のパディング値は -1.0
        # (B, N, C) -> (B, N) マスク作成 (どれか一つでも -1 ならマスク)
        mask = (labels_soft.select(2, 0) != -1.0).unsqueeze(2) # (B, N, 1)

        # マスクを適用 (B*N, C)
        log_prob_masked = log_prob.masked_select(mask).view(-1, n_classes)
        labels_soft_masked = labels_soft.masked_select(mask).view(-1, n_classes)
        
        # 4. 損失計算 (KLDivLoss)
        # loss_function は run.py で nn.KLDivLoss(reduction='sum') として定義
        loss = loss_function(log_prob_masked, labels_soft_masked)
        
        # 発話数で正規化
        num_utterances = mask.sum()
        if num_utterances > 0:
            loss = loss / num_utterances
        
        # 5. 評価指標のための変換
        # 予測 (ハードラベル)
        pred = torch.argmax(log_prob, dim=2).cpu().numpy().tolist() # (B, N)
        # 正解 (ソフトラベル -> ハードラベル)
        # (注: パディング箇所 (-1) は argmax でも問題ない値 (-1 or 0) になるはずだが、後でマスクするのでOK)
        label_hard = torch.argmax(labels_soft, dim=2).cpu().numpy().tolist() # (B, N)
        
        # パディング除外用の正解ラベル (DAG-ERC と同様)
        # labels_soft が (-1, -1, ...) の場合、label_hard は 0 になる。
        # 正確なマスクのために、-1 でパディングされたハードラベルも用意する
        label_hard_padded = torch.where(
            labels_soft.sum(dim=2) < -0.5, # パディング (-1*C) かどうか
            torch.tensor(-1, device=labels_soft.device, dtype=torch.long),
            torch.argmax(labels_soft, dim=2)
        ).cpu().numpy().tolist()


        preds += pred
        labels += label_hard_padded # パディング(-1)を含むハードラベル
        losses.append(loss.item())

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

    if preds != []:
        new_preds = []
        new_labels = []
        # DAG-ERC と同じパディング除外ロジック
        for i, label in enumerate(labels):
            for j, l in enumerate(label):
                if l != -1:
                    new_labels.append(l)
                    new_preds.append(preds[i][j])
    else:
        return float('nan'), float('nan'), [], [], float('nan')

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)
    
    # 5クラス (0-4) の F1スコア
    avg_fscore = round(f1_score(new_labels, new_preds, average='weighted', labels=list(range(5))) * 100, 2)
    
    return avg_loss, avg_accuracy, labels, preds, avg_fscore