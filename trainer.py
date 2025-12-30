import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def train_or_eval_model(model, loss_function, dataloader, epoch, cuda, args, optimizer=None, scheduler=None, train=False):
    losses = []
    preds = []
    labels = []
    nlls = []
    
    # 詳細データ保存用
    raw_probs = []       # 予測確率分布 (またはAlpha)
    raw_soft_labels = [] # 正解ソフトラベル
    text_data = []       # 発話テキスト

    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        # dataloader unpack
        # 0:text, 1:audio, 2:soft_label, 3:adj, 4:s_mask, 5:s_mask_onehot, 6:lengths, 7:speakers, 8:utterances
        features_text = data[0].cuda() if cuda else data[0]
        features_audio = data[1].cuda() if cuda else data[1]
        labels_soft = data[2].cuda() if cuda else data[2]
        adj = data[3].cuda() if cuda else data[3]
        s_mask = data[4].cuda() if cuda else data[4]
        s_mask_onehot = data[5].cuda() if cuda else data[5]
        lengths = data[6].cuda() if cuda else data[6]
        batch_utterances = data[8]
        
        # モデル入力
        # args.edl_r2=True -> Alpha (exp) が返る
        # args.edl_r2=False -> Prob (softmax) が返る
        outputs = model(features_text, features_audio, adj, s_mask, s_mask_onehot, lengths)
        
        # マスク作成 (パディング除外)
        # labels_soft: (B, Seq, Class)
        # 有効な部分だけを取り出すためのマスク
        valid_mask = (labels_soft.select(2, 0) != -1.0).unsqueeze(2) # (B, Seq, 1)
        
        # --- 損失計算 & 確率変換 ---
        if args.edl_r2:
            # === EDL Mode ===
            # outputs は Alpha。Loss関数には Alpha を渡す。
            # masked_selectで1次元化してから渡す
            valid_alpha = outputs.masked_select(valid_mask).view(-1, 5) # 5 is n_classes(assume IEMOCAP)
            valid_targets = labels_soft.masked_select(valid_mask).view(-1, 5)
            
            # Loss計算 (Epoch数を渡す)
            loss = loss_function(valid_alpha, valid_targets, epoch + 1)
            
            # 評価用に Probability に変換 (Alpha / Sum(Alpha))
            # outputs自体は勾配計算用に保持、probは評価用
            alpha = outputs
            sum_alpha = torch.sum(alpha, dim=2, keepdim=True)
            prob = alpha / (sum_alpha + 1e-10) # 確率化
            
            # NLL記録用の log_prob
            log_prob = torch.log(prob + 1e-10)

        else:
            # === Normal Mode ===
            # outputs は Probability。Loss関数には LogProb を渡す。
            prob = outputs
            log_prob = torch.log(prob + 1e-10)
            
            valid_log_prob = log_prob.masked_select(valid_mask).view(-1, 5)
            valid_targets = labels_soft.masked_select(valid_mask).view(-1, 5)
            
            # Loss計算 (SoftNLLLoss)
            # ※ SoftNLLLossの実装が sum reduction なら、マスク後の有効発話数で割る必要あり
            loss_sum = loss_function(valid_log_prob, valid_targets)
            loss = loss_sum / valid_mask.sum() if valid_mask.sum() > 0 else torch.tensor(0.0).to(outputs.device)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()

        losses.append(loss.item())
        
        # --- 評価指標計算用のデータ収集 ---
        # Hard Label 予測
        p_hard = torch.argmax(prob, 2) # probは上で計算済み
        # 正解 Hard Label (パディングは -1 に戻す)
        l_hard = torch.where(labels_soft.sum(2) < -0.5, torch.tensor(-1).to(labels_soft.device), torch.argmax(labels_soft, 2))
        
        # GPUからCPUへ戻してリスト化
        prob_cpu = prob.detach().cpu().numpy()
        ls_cpu = labels_soft.detach().cpu().numpy()
        
        for i, seq in enumerate(l_hard.cpu().tolist()):
            for j, l in enumerate(seq):
                if l != -1: # パディング以外
                    labels.append(l)
                    preds.append(p_hard[i][j].item())
                    
                    # NLL記録 (1発話ごと)
                    # EDLでもNormalでも log_prob と label を使って計算
                    single_log_prob = log_prob[i][j]
                    single_label = labels_soft[i][j]
                    # single_labelがOne-hotでない(Soft)場合も考慮してsumをとる
                    single_nll = -torch.sum(single_label * single_log_prob).item()
                    nlls.append(single_nll)
                    
                    # 詳細データを保存 (train以外推奨)
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
        avg_loss = 0.0
        avg_nll = 0.0
        f1 = 0.0
        acc = 0.0

    return avg_loss, avg_nll, acc, labels, preds, f1, raw_probs, raw_soft_labels, text_data