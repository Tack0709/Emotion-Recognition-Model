import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def train_or_eval_model(model, loss_function, dataloader, epoch, cuda, args, optimizer=None, scheduler=None, train=False):
    losses = []
    preds = []
    labels = []
    nlls = []
    kls = []  # ★追加: KLダイバージェンス記録用
    
    raw_probs = []
    raw_soft_labels = []
    text_data = []

    # KLダイバージェンス計算用 (reduction='none'で個別に計算)
    kl_criterion = nn.KLDivLoss(reduction='none')

    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        features_text = data[0].cuda() if cuda else data[0]
        features_audio = data[1].cuda() if cuda else data[1]
        labels_soft = data[2].cuda() if cuda else data[2]
        adj = data[3].cuda() if cuda else data[3]
        s_mask = data[4].cuda() if cuda else data[4]
        s_mask_onehot = data[5].cuda() if cuda else data[5]
        lengths = data[6].cuda() if cuda else data[6]
        batch_utterances = data[8]
        labels_hard_gt = data[9].cuda() if cuda else data[9]
        
        outputs = model(features_text, features_audio, adj, s_mask, s_mask_onehot, lengths)
        
        # --- Loss計算 ---
        valid_mask = (labels_soft.select(2, 0) != -1.0).unsqueeze(2) 
        
        if args.edl_r2:
            # EDLの場合の確率計算
            valid_alpha = outputs.masked_select(valid_mask).view(-1, 5) 
            valid_targets = labels_soft.masked_select(valid_mask).view(-1, 5)
            loss = loss_function(valid_alpha, valid_targets, epoch + 1)
            
            alpha = outputs
            sum_alpha = torch.sum(alpha, dim=2, keepdim=True)
            prob = alpha / (sum_alpha + 1e-10)
            log_prob = torch.log(prob + 1e-10)
        else:
            # 通常の場合
            prob = outputs
            log_prob = torch.log(prob + 1e-10)
            valid_log_prob = log_prob.masked_select(valid_mask).view(-1, 5)
            valid_targets = labels_soft.masked_select(valid_mask).view(-1, 5)
            loss_sum = loss_function(valid_log_prob, valid_targets)
            loss = loss_sum / valid_mask.sum() if valid_mask.sum() > 0 else torch.tensor(0.0).to(outputs.device)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()

        losses.append(loss.item())
        
        # --- 評価指標計算 ---
        p_hard = torch.argmax(prob, 2)
        l_hard = torch.where(labels_soft.sum(2) < -0.5, torch.tensor(-1).to(labels_soft.device), torch.argmax(labels_soft, 2))
        
        prob_cpu = prob.detach().cpu().numpy()
        ls_cpu = labels_soft.detach().cpu().numpy()
        
        # ★追加: バッチ全体のKLを計算 (B, N, C) -> (B, N)
        # input=log_prob, target=labels_soft
        # labels_softが確率分布でない場合(カウント等)は正規化が必要
        labels_soft_norm = labels_soft
        if args.edl_r2: # EDLの場合はカウントが入る可能性があるので正規化
             ls_sum = labels_soft.sum(dim=2, keepdim=True)
             labels_soft_norm = labels_soft / (ls_sum + 1e-10)

        # KL(P || Q) = sum(P * (log P - log Q)) = sum(target * (log target - input))
        # PyTorchのKLDivLossは target * (log(target) - input) を計算する
        # ここでは input が log_prob なのでそのまま渡せる
        batch_kl_loss = kl_criterion(log_prob, labels_soft_norm).sum(dim=2) # クラス方向の合計 (B, N)
        
        for i, seq in enumerate(l_hard.cpu().tolist()):
            for j, l in enumerate(seq):
                if l != -1: # パディング以外
                    
                    true_label_id = labels_hard_gt[i][j].item()
                    
                    if args.nma:
                        if true_label_id != 5: continue
                    else:
                        pass # Defaultモード

                    labels.append(l)
                    preds.append(p_hard[i][j].item())
                    
                    single_log_prob = log_prob[i][j]
                    single_label = labels_soft[i][j]
                    
                    # NLL計算
                    if args.edl_r2:
                        label_sum = single_label.sum()
                        single_label_norm = single_label / label_sum if label_sum > 0 else single_label
                        single_nll = -torch.sum(single_label_norm * single_log_prob).item()
                    else:
                        single_nll = -torch.sum(single_label * single_log_prob).item()
                    
                    nlls.append(single_nll)
                    
                    # ★追加: KLの取得
                    kl_val = batch_kl_loss[i][j].item()
                    kls.append(kl_val)
                    
                    if not train:
                        raw_probs.append(prob_cpu[i][j])
                        raw_soft_labels.append(ls_cpu[i][j])
                        text_data.append(batch_utterances[i][j])

    if len(preds) > 0:
        avg_loss = np.mean(losses)
        avg_nll = np.mean(nlls)
        avg_kl = np.mean(kls) # ★追加
        f1 = f1_score(labels, preds, average='weighted') * 100
        acc = accuracy_score(labels, preds) * 100
    else:
        avg_loss = np.mean(losses) if losses else 0.0
        avg_nll = 0.0
        avg_kl = 0.0 # ★追加
        f1 = 0.0
        acc = 0.0

    # 戻り値に avg_kl を追加 (4番目)
    return avg_loss, avg_nll, acc, avg_kl, labels, preds, f1, raw_probs, raw_soft_labels, text_data