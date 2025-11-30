import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np, itertools, random, copy, math
from model_utils import * # DAG-ERCのmodel_utils.pyをそのまま使用

# ERC-SLT22 から Bilinear Pooling を導入
class bilinear_pooling(nn.Module):
    """
    低ランク近似を用いたバイリニアプーリング層。
    テキストと音声の特徴量を融合させる。
    """
    def __init__(self, text_dim, audio_dim, D=256, O=256): # 入力次元を個別に設定
        super(bilinear_pooling, self).__init__()
        # 低ランク近似用の射影行列
        # liner層を使用して、テキストと音声の特徴量をD次元に射影
        self.U1 = nn.Linear(text_dim, D, bias=False)
        self.U2 = nn.Linear(audio_dim, D, bias=False)
        self.P = nn.Linear(D, O)
        # ショートカット接続用の行列
        self.V1 = nn.Linear(text_dim, O, bias=False)
        self.V2 = nn.Linear(audio_dim, O, bias=False)
        self.output_dim = O

    def forward(self, e1, e2): # e1: text (B, N, D_t), e2: audio (B, N, D_a)
        # 相互作用項 + ショートカット項
        c_ = self.P(torch.sigmoid(self.U1(e1)) * torch.sigmoid(self.U2(e2)))
        c = c_ + self.V1(e1) + self.V2(e2)
        return c

class DAGERC_multimodal(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)
        self.gnn_layers = args.gnn_layers

        # 1. 融合層 (Bilinear Pooling)
        # args.text_dim と args.audio_dim を args から渡す必要がある
        self.fusion_layer = bilinear_pooling(
            args.text_dim, 
            args.audio_dim, 
            D=args.fusion_dim_D, 
            O=args.fusion_dim_O
        )
        
        # 2. GNNへの入力射影層
        self.fc1 = nn.Linear(self.fusion_layer.output_dim, args.hidden_dim)

        # 3. GNN部分 (DAG-ERC と同様)
        if not args.no_rel_attn:
            self.rel_attn = True
        else:
            self.rel_attn = False

        if self.args.attn_type == 'rgcn':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GAT_dialoggcn_v1(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        # (他のGATタイプも同様にDAG-ERCからコピー)
        # ...

        grus_c = []
        for _ in range(args.gnn_layers):
            grus_c += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_c = nn.ModuleList(grus_c)

        grus_p = []
        for _ in range(args.gnn_layers):
            grus_p += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_p = nn.ModuleList(grus_p)
        
        fcs = []
        for _ in range(args.gnn_layers):
            fcs += [nn.Linear(args.hidden_dim * 2, args.hidden_dim)]
        self.fcs = nn.ModuleList(fcs)

        self.nodal_att_type = args.nodal_att_type
        
        # GNNの出力次元 + 融合特徴量の次元 (skip connection)
        in_dim = args.hidden_dim * (args.gnn_layers + 1) + self.fusion_layer.output_dim

        # 4. 出力層
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [self.dropout]
        layers += [nn.Linear(args.hidden_dim, num_class)] # num_class = 5

        self.out_mlp = nn.Sequential(*layers)

        self.attentive_node_features = attentive_node_features(in_dim)

    def forward(self, features_text, features_audio, adj, s_mask, s_mask_onehot, lengths):
        '''
        :param features_text: (B, N, D_t)
        :param features_audio: (B, N, D_a)
        :param adj: (B, N, N)
        :param s_mask: (B, N, N)
        :param s_mask_onehot: (B, N, N, 2)
        :return:
        '''
        num_utter = features_text.size()[1]

        # 1. 融合
        features_fused = self.fusion_layer(features_text, features_audio) # (B, N, D_f)

        # 2. GNN入力
        H0 = F.relu(self.fc1(features_fused)) # (B, N, D_h)
        H = [H0]

        # 3. GNN実行 (DAG-ERC と同様)
        for l in range(self.args.gnn_layers):
            C = self.grus_c[l](H[l][:,0,:]).unsqueeze(1) 
            M = torch.zeros_like(C).squeeze(1) 
            P = self.grus_p[l](M, H[l][:,0,:]).unsqueeze(1)  
            H1 = C+P
            for i in range(1, num_utter):
                if self.args.attn_type == 'rgcn':
                    #H1は直前の発話までのかくれ状態が積みあがってる
                    _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i], s_mask[:,i,:i])
                # (他のGATタイプも同様)
                # ...
                
                C = self.grus_c[l](H[l][:,i,:], M).unsqueeze(1)
                P = self.grus_p[l](M, H[l][:,i,:]).unsqueeze(1)   
                H_temp = C+P
                #バッチサイズ×発話数（これの分積みあがる）×隠れ層次元
                H1 = torch.cat((H1 , H_temp), dim = 1)  
            H.append(H1)
        
        # 4. Skip connection (融合特徴量)
        H.append(features_fused) 
        
        H = torch.cat(H, dim = 2) 

        H = self.attentive_node_features(H, lengths, self.nodal_att_type) 

        # 5. 出力
        logits = self.out_mlp(H) # (B, N, C)

        # NLL (KLDivLoss) のために log_softmax を返す
        return F.log_softmax(logits, dim=2)