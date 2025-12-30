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
    """
    768次元のテキスト/音声特徴を受け取り、
    - 両モダリティ利用時は bilinear_pooling で O(=512既定) 次元に融合
    - 単一モダリティ時は 768 -> 512 -> ReLU で圧縮
    その後 512 -> 300 (hidden_dim) に射影し、DAG-ERC GNN へ渡す。
    """
    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        self.use_text = getattr(args, 'use_text', True)
        self.use_audio = getattr(args, 'use_audio', True)
        if not (self.use_text or self.use_audio):
            raise ValueError("少なくとも1つのモダリティが必要です。")

        self.dropout = nn.Dropout(args.dropout)
        self.gnn_layers = args.gnn_layers

        if self.use_text and self.use_audio:
            self.fusion_mode = 'bilinear'
            self.fusion_layer = bilinear_pooling(
                args.text_dim,   # 768 (BERT CLS)
                args.audio_dim,  # 768 (Wav2Vec mean pooling)
                D=args.fusion_dim_D,  # 低ランク内部次元 (256)
                O=args.fusion_dim_O   # 出力融合次元 (512)
            )
            fusion_out_dim = self.fusion_layer.output_dim  # 512
        else:
            self.fusion_mode = 'single'
            input_dim = args.text_dim if self.use_text else args.audio_dim  # 768
            self.single_modal_proj = nn.Sequential(
                nn.Linear(input_dim, args.single_modal_dim),  # 768 -> 512
                nn.ReLU()
            )
            fusion_out_dim = args.single_modal_dim  # 512

        # 融合後 (512) -> hidden_dim (=300) に圧縮して GNN へ入力
        self.fc1 = nn.Linear(fusion_out_dim, args.hidden_dim)

        # --- GNN (各層の隠れ次元は hidden_dim=300 を維持) ---
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
        
        # GNN の各層出力 (hidden_dim=300) と最初の融合特徴 (512) を連結
        in_dim = args.hidden_dim * (args.gnn_layers + 1) + fusion_out_dim
        
        # Simple NN の場合は GNN をバイパスするため、入力次元が変わる
        if getattr(args, 'simple_nn', False):
            # 融合特徴量(512) のみを使用
            in_dim = fusion_out_dim

        # 4. 出力層
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [self.dropout]
        layers += [nn.Linear(args.hidden_dim, num_class)] # num_class = 5

        self.out_mlp = nn.Sequential(*layers)

        self.attentive_node_features = attentive_node_features(in_dim)

    def forward(self, features_text, features_audio, adj, s_mask, s_mask_onehot, lengths):
        # features_text/audio: (B, N, 768)
        num_utter = features_text.size()[1]

        if self.fusion_mode == 'bilinear':
            # 双方利用時: 768_text + 768_audio -> 512
            features_fused = self.fusion_layer(features_text, features_audio)
        else:
            # 単一モダリティ時: 768 -> 512
            base_feat = features_text if self.use_text else features_audio
            features_fused = self.single_modal_proj(base_feat)

        # ==========================================
        # ★追加: Simple NN モード (Context-free)
        # ==========================================
        if getattr(self.args, 'simple_nn', False):
            # GNNを通さず、融合特徴量を直接出力層へ
            # H = features_fused
            logits = self.out_mlp(features_fused)
            
            # --- 出力層の分岐 (EDL or Softmax) ---
            if getattr(self.args, 'edl_r2', False):
                return torch.exp(logits) # Evidence
            else:
                return F.softmax(logits, dim=2)

        # ==========================================
        # GNN使用モード (通常DAG-ERC または Standard GNN)
        # ==========================================
        
        # 512 -> 300 (hidden_dim)　GNN入力
        H0 = F.relu(self.fc1(features_fused)) # (B, N, 300)
        H = [H0]

        # ==========================================
        # ★追加: Standard GNN モード
        # ==========================================
        if getattr(self.args, 'standard_gnn', False):
            # 隣接行列を「全結合(全部1)」にする
            # これにより、DAG（過去→未来のみ）の制約がなくなり、
            # 全ての発話が相互に接続された普通のGNNとして動作します。
            adj = torch.ones_like(adj)
            
            # 話者マスクも無効化（全て1）して、話者関係の区別をなくす
            s_mask = torch.ones_like(s_mask)
        
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
        # Skip connection: GNN各層(300)と融合特徴(512)を結合 ⇒ (B, N, in_dim)
        H.append(features_fused) 
        
        H = torch.cat(H, dim = 2) 

        H = self.attentive_node_features(H, lengths, self.nodal_att_type) 

        # 5. 出力
        logits = self.out_mlp(H) # (B, N, num_class)

        # ==========================================
        # ★追加: EDL(R2) 使用時の分岐
        # ==========================================
        if getattr(self.args, 'edl_r2', False):
            # EDLの場合: 証拠 (Evidence) を返す
            # Evidence = exp(logits) で非負の値を保証
            # (場合によっては +1 することもあるが、loss.pyの仕様に合わせる)
            return torch.exp(logits)
        
        # 通常の場合: 確率 (Probability) を返す
        return F.softmax(logits, dim=2)