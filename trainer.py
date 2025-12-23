import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def train_or_eval_model(model, loss_function, dataloader, epoch, cuda, args, optimizer=None, scheduler=None, train=False):
    losses = []
    preds = []
    labels = []
    nlls = []
    
    # ★追加: 保存用リスト
    raw_probs = []       # 予測確率分布
    raw_soft_labels = [] # 正解ソフトラベル
    text_data = []       # 発話テキスト

    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        # dataset.pyのcollate_fnの戻り値に合わせて展開
        # features_text, features_audio, labels_soft, adj, s_mask, s_mask_onehot, lengths, speakers, utterances
        features_text = data[0].cuda() if cuda else data[0]
        features_audio = data[1].cuda() if cuda else data[1]
        labels_soft = data[2].cuda() if cuda else data[2]
        adj = data[3].cuda() if cuda else data[3]
        s_mask = data[4].cuda() if cuda else data[4]
        s_mask_onehot = data[5].cuda() if cuda else data[5]
        lengths = data[6].cuda() if cuda else data[6]
        
        # ★追加: テキストデータの取得 (dataset.pyのcollate_fnの9番目の戻り値)
        batch_utterances = data[8]
        
        # モデル入力
        prob = model(features_text, features_audio, adj, s_mask, s_mask_onehot, lengths)
        
        # NLL損失の計算
        log_prob = torch.log(prob + 1e-10)
        
        # パディング(-1.0)を除外するためのマスク
        l_s = labels_soft
        valid_mask = (l_s.select(2, 0) != -1.0).unsqueeze(2)
        
        # Loss計算 (SoftNLL)
        loss = loss_function(
            log_prob.masked_select(valid_mask).view(-1, 5), 
            l_s.masked_select(valid_mask).view(-1, 5)
        )
        
        # 平均化 (有効な発話数で割る)
        loss = loss / valid_mask.sum() if valid_mask.sum() > 0 else torch.tensor(0.0).to(loss.device)
        
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            
            if scheduler:
                scheduler.step()

        losses.append(loss.item())
        
        # 評価指標計算用のデータ収集
        # Soft Label -> Hard Label 変換
        p_hard = torch.argmax(prob, 2)
        l_hard = torch.where(l_s.sum(2) < -0.5, torch.tensor(-1).to(l_s.device), torch.argmax(l_s, 2))
        
        # GPUからCPUへ戻してリスト化
        # ★ここで詳細データも収集します
        prob_cpu = prob.detach().cpu().numpy()
        ls_cpu = l_s.detach().cpu().numpy()
        
        # GPUからCPUへ戻してリスト化
        for i, seq in enumerate(l_hard.cpu().tolist()):
            for j, l in enumerate(seq):
                if l != -1: # パディング以外
                    labels.append(l)
                    preds.append(p_hard[i][j].item())
                    
                    # NLL記録 (1発話ごと)
                    single_log_prob = log_prob[i][j]
                    single_label = l_s[i][j]
                    single_nll = -torch.sum(single_label * single_log_prob).item()
                    nlls.append(single_nll)
                    
                    # ★追加: 詳細データを保存 (train以外、つまり検証・テスト時のみ推奨)
                    if not train:
                        raw_probs.append(prob_cpu[i][j])
                        raw_soft_labels.append(ls_cpu[i][j])
                        text_data.append(batch_utterances[i][j])

    if len(preds) > 0:
        avg_loss = np.mean(losses)
        avg_nll = np.mean(nlls)
        f1 = f1_score(labels, preds, average='weighted') * 100
        acc = accuracy_score(labels, preds) * 100 # Accuracy計算
    else:
        avg_loss = 0.0
        avg_nll = 0.0
        f1 = 0.0
        acc = 0.0

    # 戻り値に acc を含める
    return avg_loss, avg_nll, acc, labels, preds, f1, raw_probs, raw_soft_labels, text_data