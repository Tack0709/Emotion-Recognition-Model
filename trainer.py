import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def train_or_eval_model(model, loss_function, dataloader, epoch, cuda, args, optimizer=None, scheduler=None, train=False):
    losses = []
    preds = []
    labels = []
    nlls = []
    
    raw_probs = []
    raw_soft_labels = []
    text_data = []

    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        # アンパック (dataset.pyの修正に合わせて最後に labels_hard_gt を受け取る)
        features_text = data[0].cuda() if cuda else data[0]
        features_audio = data[1].cuda() if cuda else data[1]
        labels_soft = data[2].cuda() if cuda else data[2]
        adj = data[3].cuda() if cuda else data[3]
        s_mask = data[4].cuda() if cuda else data[4]
        s_mask_onehot = data[5].cuda() if cuda else data[5]
        lengths = data[6].cuda() if cuda else data[6]
        batch_utterances = data[8]
        # ★追加: 真のハードラベル
        labels_hard_gt = data[9].cuda() if cuda else data[9]
        
        outputs = model(features_text, features_audio, adj, s_mask, s_mask_onehot, lengths)
        
        # --- Loss計算 (ここでは全データを使用) ---
        valid_mask = (labels_soft.select(2, 0) != -1.0).unsqueeze(2) 
        
        if args.edl_r2:
            valid_alpha = outputs.masked_select(valid_mask).view(-1, 5) 
            valid_targets = labels_soft.masked_select(valid_mask).view(-1, 5)
            loss = loss_function(valid_alpha, valid_targets, epoch + 1)
            
            alpha = outputs
            sum_alpha = torch.sum(alpha, dim=2, keepdim=True)
            prob = alpha / (sum_alpha + 1e-10)
            log_prob = torch.log(prob + 1e-10)
        else:
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
        
        # --- 評価指標計算用の集計 (フィルタリング適用) ---
        p_hard = torch.argmax(prob, 2)
        # ソフトラベルのArgmaxを「評価上の正解ラベル」とする (XXXの場合もこれで0~4の正解が決まる)
        l_hard = torch.where(labels_soft.sum(2) < -0.5, torch.tensor(-1).to(labels_soft.device), torch.argmax(labels_soft, 2))
        
        prob_cpu = prob.detach().cpu().numpy()
        ls_cpu = labels_soft.detach().cpu().numpy()
        
        for i, seq in enumerate(l_hard.cpu().tolist()):
            for j, l in enumerate(seq):
                if l != -1: # パディング以外
                    
                    # ★追加: フィルタリングロジック
                    true_label_id = labels_hard_gt[i][j].item()
                    
                    if args.nma:
                        # NMAモード: XXX(5) のみ評価する -> 5以外ならスキップ
                        if true_label_id != 5:
                            continue
                    else:
                        # Defaultモード: Loaderから来るデータ(XXX以外)を全て評価
                        # (dataset.pyで既にXXXは除外されているので、ここではチェック不要)
                        pass

                    # 集計に追加
                    labels.append(l)
                    preds.append(p_hard[i][j].item())
                    
                    single_log_prob = log_prob[i][j]
                    single_label = labels_soft[i][j]
                    
                    # === 【修正箇所】 NLL計算時のみ、投票数を確率に正規化する ===
                    # 学習(Loss)では投票数を使うが、表示用NLLは確率分布との距離で見たい場合
                    if args.edl_r2:
                        label_sum = single_label.sum()
                        if label_sum > 0:
                            single_label_norm = single_label / label_sum
                        else:
                            single_label_norm = single_label
                        
                        single_nll = -torch.sum(single_label_norm * single_log_prob).item()
                    else:
                        # 既存のロジック (元々確率値が入っている場合)
                        single_nll = -torch.sum(single_label * single_log_prob).item()
                    
                    nlls.append(single_nll)
                    
                    if not train:
                        raw_probs.append(prob_cpu[i][j])
                        raw_soft_labels.append(ls_cpu[i][j])
                        text_data.append(batch_utterances[i][j])

    if len(preds) > 0:
        avg_loss = np.mean(losses)
        avg_nll = np.mean(nlls)
        f1 = f1_score(labels, preds, average='weighted') * 100
        acc = accuracy_score(labels, preds) * 100
    else:
        # 該当データがない場合（例: NMAモードだがXXXデータが1つもない場合など）
        avg_loss = np.mean(losses) if losses else 0.0
        avg_nll = 0.0
        f1 = 0.0
        acc = 0.0

    return avg_loss, avg_nll, acc, labels, preds, f1, raw_probs, raw_soft_labels, text_data