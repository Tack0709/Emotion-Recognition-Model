import torch, numpy as np
from sklearn.metrics import f1_score, accuracy_score

def train_or_eval_model(model, loss_function, dataloader, epoch, cuda, args, optimizer=None, train=False):
    if train: model.train()
    else: model.eval()
    
    losses = []
    preds, labels = [], []
    
    for data in dataloader:
        f_t, f_a, l_s, adj, mask, mask_oh, lens, _, _ = data
        if f_t is None: continue
        
        if cuda:
            f_t, f_a, l_s, adj, mask, mask_oh, lens = f_t.cuda(), f_a.cuda(), l_s.cuda(), adj.cuda(), mask.cuda(), mask_oh.cuda(), lens.cuda()
        
        if train: optimizer.zero_grad()
        
        # 1. モデル出力 (確率分布)
        prob = model(f_t, f_a, adj, mask, mask_oh, lens)
        
        # 2. 対数確率に変換 (SoftNLLLossへの入力用)
        log_prob = torch.log(prob + 1e-10)

        # 3. マスク作成
        valid_mask = (l_s.select(2, 0) != -1.0).unsqueeze(2)
        
        # 4. 損失計算 (SoftNLLLoss)
        # ここで計算される loss がそのまま NLL になります
        loss = loss_function(
            log_prob.masked_select(valid_mask).view(-1, 5), 
            l_s.masked_select(valid_mask).view(-1, 5)
        )
        
        # 正規化
        loss = loss / valid_mask.sum() if valid_mask.sum() > 0 else torch.tensor(0.0).to(loss.device)
        
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            
        losses.append(loss.item())
        
        # 評価用 (ハードラベル)
        p_hard = torch.argmax(prob, 2)
        l_hard_padded = torch.where(l_s.sum(2) < -0.5, torch.tensor(-1).to(l_s.device), torch.argmax(l_s, 2))
        
        for i, l_seq in enumerate(l_hard_padded.cpu().tolist()):
            for j, l in enumerate(l_seq):
                if l != -1:
                    labels.append(l)
                    preds.append(p_hard[i][j].item())

    avg_nll = np.mean(losses) if losses else 0.0
    acc = accuracy_score(labels, preds) * 100 if labels else 0.0
    f1 = f1_score(labels, preds, average='weighted', labels=list(range(5))) * 100 if labels else 0.0
    
    # 戻り値: (Loss(=NLL), NLL(同じ), Acc, Labels, Preds, F1)
    # run.py のアンパック (6個) に合わせるため、avg_nll を2回返します
    return avg_nll, avg_nll, acc, labels, preds, f1