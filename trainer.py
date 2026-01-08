import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def train_or_eval_model(model, loss_function, dataloader, epoch, cuda, args, optimizer=None, scheduler=None, train=False):
    # MA (Standard) Metrics
    losses = []
    preds = []
    labels = []
    nlls = []
    
    # NMA (XXX) Metrics -- only used if args.nma is True
    nma_preds = []
    nma_labels = [] 
    nma_nlls = []
    
    # 詳細データ保存用 (MA)
    raw_probs = []       
    raw_soft_labels = [] 
    text_data = []       

    # 詳細データ保存用 (NMA) ★追加
    nma_probs = []
    nma_softs = []
    nma_texts = []

    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        # dataloader unpack
        # 0:text, 1:audio, 2:soft_label, 3:adj, 4:s_mask, 5:s_mask_onehot, 6:lengths, 7:speakers, 8:utterances, 9:hard_labels
        features_text = data[0].cuda() if cuda else data[0]
        features_audio = data[1].cuda() if cuda else data[1]
        labels_soft = data[2].cuda() if cuda else data[2]
        adj = data[3].cuda() if cuda else data[3]
        s_mask = data[4].cuda() if cuda else data[4]
        s_mask_onehot = data[5].cuda() if cuda else data[5]
        lengths = data[6].cuda() if cuda else data[6]
        batch_utterances = data[8]
        hard_labels = data[9].cuda() if cuda else data[9]
        
        # モデル入力
        outputs = model(features_text, features_audio, adj, s_mask, s_mask_onehot, lengths)
        
        # --- マスク作成 ---
        # MA Mask: パディングでない かつ XXX(5) でない
        valid_mask_ma = (labels_soft.select(2, 0) != -1.0) & (hard_labels != 5)
        
        # --- 損失計算 (MAデータのみ) ---
        if args.edl_r2:
            # === EDL Mode ===
            valid_alpha = outputs.masked_select(valid_mask_ma.unsqueeze(2)).view(-1, 5)
            valid_targets = labels_soft.masked_select(valid_mask_ma.unsqueeze(2)).view(-1, 5)
            
            if valid_alpha.size(0) > 0:
                loss = loss_function(valid_alpha, valid_targets, epoch + 1)
            else:
                loss = torch.tensor(0.0).to(outputs.device)
            
            # Probability計算
            alpha = outputs
            sum_alpha = torch.sum(alpha, dim=2, keepdim=True)
            prob = alpha / (sum_alpha + 1e-10)
            log_prob = torch.log(prob + 1e-10)

        else:
            # === Normal Mode ===
            prob = outputs
            log_prob = torch.log(prob + 1e-10)
            
            valid_log_prob = log_prob.masked_select(valid_mask_ma.unsqueeze(2)).view(-1, 5)
            valid_targets = labels_soft.masked_select(valid_mask_ma.unsqueeze(2)).view(-1, 5)
            
            if valid_mask_ma.sum() > 0:
                loss_sum = loss_function(valid_log_prob, valid_targets)
                loss = loss_sum / valid_mask_ma.sum()
            else:
                loss = torch.tensor(0.0).to(outputs.device)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()

        if valid_mask_ma.sum() > 0:
            losses.append(loss.item())
        
        # --- 評価データの収集 ---
        p_hard = torch.argmax(prob, 2)
        l_hard = hard_labels
        
        prob_cpu = prob.detach().cpu().numpy()
        ls_cpu = labels_soft.detach().cpu().numpy()
        
        # 1発話ごとに処理
        for i, seq in enumerate(l_hard.cpu().tolist()):
            for j, l in enumerate(seq):
                if l == -1: # Padding
                    continue
                
                # NLL計算 (Soft Label vs Log Prob)
                single_log_prob = log_prob[i][j]
                single_label = labels_soft[i][j]
                single_nll = -torch.sum(single_label * single_log_prob).item()

                if l == 5: 
                    # === NMA (XXX) Case ===
                    if args.nma:
                        nma_labels.append(l) # 5
                        nma_preds.append(p_hard[i][j].item())
                        nma_nlls.append(single_nll)
                        # ★追加: 分布推定データの保存
                        if not train:
                            nma_probs.append(prob_cpu[i][j])
                            nma_softs.append(ls_cpu[i][j])
                            nma_texts.append(batch_utterances[i][j])
                else:
                    # === MA (Standard) Case ===
                    labels.append(l)
                    preds.append(p_hard[i][j].item())
                    nlls.append(single_nll)
                    
                    if not train:
                        raw_probs.append(prob_cpu[i][j])
                        raw_soft_labels.append(ls_cpu[i][j])
                        text_data.append(batch_utterances[i][j])

    # --- MA Metrics Calculation ---
    if len(preds) > 0:
        avg_loss = np.mean(losses) if losses else 0.0
        avg_nll = np.mean(nlls)
        f1 = f1_score(labels, preds, average='weighted') * 100
        acc = accuracy_score(labels, preds) * 100
    else:
        avg_loss = 0.0
        avg_nll = 0.0
        f1 = 0.0
        acc = 0.0
        
    # --- NMA Metrics Calculation ---
    nma_results = {}
    if args.nma and len(nma_preds) > 0:
        nma_results['nll'] = np.mean(nma_nlls)
        nma_results['count'] = len(nma_preds)
        nma_results['preds'] = nma_preds
        nma_results['labels'] = nma_labels # 全て5
        # ★追加: 分布データ
        nma_results['probs'] = nma_probs
        nma_results['softs'] = nma_softs
        nma_results['texts'] = nma_texts

    return avg_loss, avg_nll, acc, labels, preds, f1, raw_probs, raw_soft_labels, text_data, nma_results