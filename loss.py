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
            target: 正解ラベル (Batch, Class)。
                    Datasetの修正により、EDL(R2)使用時は「投票数 (Counts)」が渡され、
                    SoftNLL使用時は「確率値 (Probs)」が渡される可能性がある。
            epoch_num: 現在のエポック数 (1-indexed想定)
        """
        eps = 1e-8
        
        # 証拠の総和 S
        S = torch.sum(alpha, dim=1, keepdim=True) # (B, 1)

        # --- 1. NLL項 (Eq 11) ---
        # Log-Likelihood of Dirichlet: sum( y_k * (log(S) - log(alpha_k)) )
        # targetが投票数(Counts)の場合、y_kとしてそのまま重みになるため正しい挙動となる。
        # targetが確率値(Probs)の場合も、従来の重み付けとして機能する。
        log_term = torch.log(S + eps) - torch.log(alpha + eps)
        loss_nll = torch.sum(target * log_term, dim=1).mean()

        # --- 2. R2 正則化項 (Eq 13) ---
        # KL( target || E[eta] )
        # KLダイバージェンスの計算には、ターゲットが確率分布(合計1)である必要がある。
        # そのため、targetが投票数(Counts)の場合は正規化を行う。
        
        target_sum = torch.sum(target, dim=1, keepdim=True)
        # ゼロ除算回避 (合計が0のデータが万が一あった場合用)
        target_prob = target / (target_sum + 1e-8)
        
        # E[eta] = alpha / S
        expected_p = alpha / (S + eps)
        log_expected_p = torch.log(expected_p + eps)
        
        # KLDivLoss (reduction='batchmean')
        # target_prob は確率分布化されているため安全に計算可能
        loss_r2 = F.kl_div(log_expected_p, target_prob, reduction='batchmean')

        # --- 3. Annealing ---
        if self.annealing_step > 0:
            annealing_coef = min(1.0, epoch_num / self.annealing_step)
        else:
            annealing_coef = 1.0
            
        total_loss = loss_nll + (self.lambda_coef * annealing_coef * loss_r2)
        
        return total_loss