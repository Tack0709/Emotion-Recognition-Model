import torch, numpy as np
from sklearn.metrics import f1_score, accuracy_score

# <--- 修正: scheduler 引数を追加
def train_or_eval_model(model, loss_function, dataloader, epoch, cuda, args, optimizer=None, scheduler=None, train=False):
    if train: model.train()
    else: model.eval()
    
    losses, nll_losses, preds, labels = [], [], [], []
    
    for data in dataloader:
        f_t, f_a, l_s, adj, mask, mask_oh, lens, _, _ = data
        if f_t is None: continue
        
        if cuda:
            f_t, f_a, l_s, adj, mask, mask_oh, lens = f_t.cuda(), f_a.cuda(), l_s.cuda(), adj.cuda(), mask.cuda(), mask_oh.cuda(), lens.cuda()
        
        if train: optimizer.zero_grad()
        
        prob = model(f_t, f_a, adj, mask, mask_oh, lens)
        log_prob = torch.log(prob + 1e-10)
        valid_mask = (l_s.select(2, 0) != -1.0).unsqueeze(2)
        
        # KL損失 (学習用)
        loss_kl = loss_function(log_prob.masked_select(valid_mask).view(-1, 5), l_s.masked_select(valid_mask).view(-1, 5))
        loss_kl = loss_kl / valid_mask.sum() if valid_mask.sum() > 0 else torch.tensor(0.0).to(loss_kl.device)
        
        # NLL (評価用)
        sel_p, sel_logq = l_s.masked_select(valid_mask).view(-1, 5), log_prob.masked_select(valid_mask).view(-1, 5)
        loss_nll = -torch.sum(sel_p * sel_logq) / valid_mask.sum() if valid_mask.sum() > 0 else torch.tensor(0.0).to(loss_kl.device)

        if train:
            loss_kl.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            
            # <--- 追加: スケジューラの更新
            if scheduler:
                scheduler.step()
            
        losses.append(loss_kl.item())
        nll_losses.append(loss_nll.item())
        
        p_hard = torch.argmax(prob, 2)
        l_hard = torch.where(l_s.sum(2) < -0.5, torch.tensor(-1).to(l_s.device), torch.argmax(l_s, 2))
        for i, seq in enumerate(l_hard.cpu().tolist()):
            for j, l in enumerate(seq):
                if l != -1: labels.append(l); preds.append(p_hard[i][j].item())

    avg_kl, avg_nll = (np.mean(losses) if losses else 0.0), (np.mean(nll_losses) if nll_losses else 0.0)
    acc = accuracy_score(labels, preds) * 100 if labels else 0.0
    f1 = f1_score(labels, preds, average='weighted', labels=list(range(5))) * 100 if labels else 0.0
    
    return avg_kl, avg_nll, acc, labels, preds, f1