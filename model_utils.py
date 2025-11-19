import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def mask_logic(alpha, adj):
    '''
    隣接行列に基づいてアテンションスコアをマスクする関数
    adj: (B, N, N) - 接続があれば1, なければ0
    '''
    # 接続がない部分(0)を非常に小さな値(-infに近い値)にしてSoftmaxで0にする
    return alpha - (1 - adj) * 1e30

class GAT_dialoggcn_v1(nn.Module):
    '''
    DAG-ERCで使用されるグラフアテンション層 (RGCNタイプ)
    '''
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size * 2, 1)
        # リレーションごとの変換行列 (Wr0: 自己/他者, Wr1: パディング用)
        self.Wr0 = nn.Linear(hidden_size, hidden_size, bias = False)
        self.Wr1 = nn.Linear(hidden_size, hidden_size, bias = False)

    def forward(self, Q, K, V, adj, s_mask):
        '''
        Q: クエリ (現在の発話)
        K, V: キー、バリュー (過去の発話群)
        adj: 隣接行列
        s_mask: 話者関係マスク (同一話者=1, 他者=0)
        '''
        N = K.size()[1]
        # Qを拡張して (B, N, D) にする
        Q = Q.unsqueeze(1).expand(-1, N, -1) 
        
        # 特徴量の結合とアテンションスコアの計算
        X = torch.cat((Q, K), dim = 2) # (B, N, 2D)
        alpha = self.linear(X).permute(0, 2, 1) # (B, 1, N)
        
        # 隣接行列によるマスク
        adj = adj.unsqueeze(1)
        alpha = mask_logic(alpha, adj)
        
        # アテンション重み
        attn_weight = F.softmax(alpha, dim = 2) # (B, 1, N)

        # リレーション（話者関係）に応じた値の変換
        V0 = self.Wr0(V) # 関係あり用
        V1 = self.Wr1(V) # 関係なし用
        
        s_mask = s_mask.unsqueeze(2).float()
        V_transformed = V0 * s_mask + V1 * (1 - s_mask)

        # 重み付き和
        attn_sum = torch.bmm(attn_weight, V_transformed).squeeze(1) 

        return attn_weight, attn_sum

# 他のアテンションタイプ (Linear, Dot) も必要ならここに追加しますが、
# run.py のデフォルトは 'rgcn' なので上記があれば動作します。

class attentive_node_features(nn.Module):
    '''
    GNNの出力に対して、ダイアログ全体または過去の情報を使った
    アテンションプーリングを行う層 (Nodal Attention)
    '''
    def __init__(self, hidden_size):
        super().__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)

    def forward(self, features, lengths, nodal_att_type):
        '''
        features: (B, N, D)
        nodal_att_type: 'global' or 'past' or None
        '''
        if nodal_att_type is None:
            return features

        batch_size = features.size(0)
        max_seq_len = features.size(1)
        
        # パディングマスク作成 (有効な発話=1, パディング=0)
        padding_mask = torch.zeros(batch_size, max_seq_len).to(features.device)
        for i, l in enumerate(lengths):
            padding_mask[i, :l] = 1

        if nodal_att_type == 'global':
            # 全発話に対するアテンション
            mask = padding_mask.unsqueeze(1)
        elif nodal_att_type == 'past':
            # 過去の発話のみに対するアテンション（因果マスク）
            causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).to(features.device)
            mask = padding_mask.unsqueeze(1) * causal_mask.unsqueeze(0)
        else:
            return features

        # アテンション計算
        x = self.transform(features)
        scores = torch.bmm(x, features.permute(0, 2, 1)) # (B, N, N)
        
        # マスク適用（0の部分を-infに）
        scores = scores.masked_fill(mask == 0, -1e9)
        
        alpha = F.softmax(torch.tanh(scores), dim=2) 
        attn_pool = torch.bmm(alpha, features) # (B, N, D)

        return attn_pool