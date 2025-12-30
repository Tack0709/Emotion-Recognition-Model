import torch
import torch.nn as nn
import torch.nn.functional as F

class EDL_R2_Loss(nn.Module):
    def __init__(self, annealing_step=10, lambda_coef=0.8, device='cuda'):
        """
        EDL*(R2) Loss Function
        Args:
            annealing_step (int): 正則化の重みを徐々に増やすためのステップ数（総エポック数など）
            lambda_coef (float): 正則化の最大係数 (IEMOCAP: 0.8, CREMA-D: 0.2)
        """
        super().__init__()
        self.annealing_step = annealing_step
        self.lambda_coef = lambda_coef
        self.device = device

    def forward(self, alpha, target, epoch_num):
        """
        Args:
            alpha: モデル出力 (Batch, Class)。Evidence + 1 (正の値)。
            target: 正解ラベル (Batch, Class)。Soft Label (確率分布) 推奨。
            epoch_num: 現在のエポック数 (1-indexed想定)
        """
        eps = 1e-8
        
        # 証拠の総和 S
        S = torch.sum(alpha, dim=1, keepdim=True) # (B, 1)

        # --- 1. NLL項 (Eq 11) ---
        # Log-Likelihood of Dirichlet: sum( y_k * (log(S) - log(alpha_k)) )
        log_term = torch.log(S + eps) - torch.log(alpha + eps)
        loss_nll = torch.sum(target * log_term, dim=1).mean()

        # --- 2. R2 正則化項 (Eq 13) ---
        # KL( target || E[eta] )
        # E[eta] = alpha / S
        expected_p = alpha / (S + eps)
        log_expected_p = torch.log(expected_p + eps)
        
        # KLDivLoss (reduction='batchmean' はバッチサイズで割ってくれる)
        loss_r2 = F.kl_div(log_expected_p, target, reduction='batchmean')

        # --- 3. Annealing ---
        if self.annealing_step > 0:
            annealing_coef = min(1.0, epoch_num / self.annealing_step)
        else:
            annealing_coef = 1.0
            
        total_loss = loss_nll + (self.lambda_coef * annealing_coef * loss_r2)
        
        return total_loss