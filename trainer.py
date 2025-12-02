import torch, numpy as np
from sklearn.metrics import f1_score, accuracy_score

def train_or_eval_model(model, loss_function, dataloader, epoch, cuda, args, optimizer=None, train=False):
    if train: model.train()
    else: model.eval()
    
    losses = []     # KL損失（学習用）
    nll_losses = [] # NLL損失（比較・評価用） <--- 追加
    preds, labels = [], []
    
    for data in dataloader:
        f_t, f_a, l_s, adj, mask, mask_oh, lens, _, _ = data
        if f_t is None: continue
        
        if cuda:
            f_t, f_a, l_s, adj, mask, mask_oh, lens = f_t.cuda(), f_a.cuda(), l_s.cuda(), adj.cuda(), mask.cuda(), mask_oh.cuda(), lens.cuda()
        
        if train: optimizer.zero_grad()
        
        # 1. モデル出力 (確率分布)
        prob = model(f_t, f_a, adj, mask, mask_oh, lens)
        
        # 2. 対数確率に変換
        log_prob = torch.log(prob + 1e-10)

        # 3. マスク作成
        valid_mask = (l_s.select(2, 0) != -1.0).unsqueeze(2)
        
        # --- 損失の計算 ---
        
        # A. KLダイバージェンス (学習用: 予測と正解の「距離」)
        # PyTorchのKLDivLossは「log_prob」と「prob(正解)」を受け取る
        loss_kl = loss_function(
            log_prob.masked_select(valid_mask).view(-1, 5), 
            l_s.masked_select(valid_mask).view(-1, 5)
        )
        
        # B. NLL / クロスエントロピー (比較用: -sum(P * logQ)) <--- 追加
        # 手動で計算するのが確実です: - Σ (正解 * log(予測))
        # マスクされたデータだけを取り出して計算
        selected_log_prob = log_prob.masked_select(valid_mask).view(-1, 5)
        selected_target = l_s.masked_select(valid_mask).view(-1, 5)
        
        # 要素ごとの積をとって合計し、マイナスをかける
        loss_nll = -torch.sum(selected_target * selected_log_prob)
        
        # 正規化 (発話数で割る)
        num_valid = valid_mask.sum()
        if num_valid > 0:
            loss_kl = loss_kl / num_valid
            loss_nll = loss_nll / num_valid # <--- 追加
        else:
            loss_kl = torch.tensor(0.0).to(loss_kl.device)
            loss_nll = torch.tensor(0.0).to(loss_kl.device) # <--- 追加
        
        if train:
            loss_kl.backward() # 学習はKLで行う（勾配はNLLと同じなのでOK）
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            
        losses.append(loss_kl.item())
        nll_losses.append(loss_nll.item()) # <--- 追加
        
        # 評価用データ蓄積
        p_hard = torch.argmax(prob, 2)
        l_hard_padded = torch.where(l_s.sum(2) < -0.5, torch.tensor(-1).to(l_s.device), torch.argmax(l_s, 2))
        
        for i, l_seq in enumerate(l_hard_padded.cpu().tolist()):
            for j, l in enumerate(l_seq):
                if l != -1:
                    labels.append(l)
                    preds.append(p_hard[i][j].item())

    avg_loss_kl = np.mean(losses) if losses else 0.0
    avg_loss_nll = np.mean(nll_losses) if nll_losses else 0.0 # <--- 追加
    f1 = f1_score(labels, preds, average='weighted', labels=list(range(5))) * 100 if labels else 0.0
    
    # 戻り値に NLL を追加して返す
    # (KL, NLL, Accuracy(今回は0), labels, preds, F1)
    return avg_loss_kl, avg_loss_nll, 0.0, labels, preds, f1