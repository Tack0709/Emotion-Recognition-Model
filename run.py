import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np, argparse, time, random, logging
import torch
import torch.nn as nn
from transformers import Adafactor
from torch.optim.lr_scheduler import LambdaLR
import copy

from dataloader import get_multimodal_loaders
from model import DAGERC_multimodal
from trainer import train_or_eval_model
from loss import EDL_R2_Loss

# --- (中略: get_logger, seed_everything, SoftNLLLoss はそのまま) ---
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    return logger

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class SoftNLLLoss(nn.Module):
    def __init__(self, reduction='sum'):
        super().__init__()
        self.reduction = reduction
    def forward(self, log_probs, targets):
        loss = -torch.sum(targets * log_probs, dim=1)
        if self.reduction == 'sum': return loss.sum()
        elif self.reduction == 'mean': return loss.mean()
        return loss

# --- Noam Scheduler ---
def get_noam_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    def lr_lambda(current_step):
        current_step = max(1, current_step)
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return (num_warmup_steps ** 0.5) * (current_step ** -0.5)
    return LambdaLR(optimizer, lr_lambda, last_epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # --- (引数定義はそのまま) ---
    parser.add_argument('--data_dir', default='output_data', type=str, help='path to data directory')
    parser.add_argument('--eval_metric', type=str, default='loss', choices=['f1', 'loss'], 
                        help='Metric to select best model (f1: Maximize F1, loss: Minimize NLL)')
    parser.add_argument('--test_session', type=int, default=5, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--log_file_name', type=str, default=None)
    parser.add_argument('--dev_ratio', type=float, default=0.2)
    parser.add_argument('--nma', action='store_true', help='Include "xxx" labels and train with soft labels only')
    parser.add_argument('--train_ma_test_nma', action='store_true', help='Train on MA data (exclude XXX), but Test on NMA data (include XXX)')
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--mlp_layers', type=int, default=2)
    parser.add_argument('--gnn_layers', type=int, default=4)
    parser.add_argument('--attn_type', type=str, default='rgcn', choices=['dotprod','linear','bilinear', 'rgcn'])
    parser.add_argument('--no_rel_attn',  action='store_true')
    parser.add_argument('--windowp', type=int, default=1)
    parser.add_argument('--windowf', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=5.0)
    parser.add_argument('--lr', type=float, default=8.84e-4) # 参考文献の値
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--nodal_att_type', type=str, default=None, choices=['global','past'])
    parser.add_argument('--text_dim', type=int, default=768)
    parser.add_argument('--audio_dim', type=int, default=768)
    parser.add_argument('--fusion_dim_D', type=int, default=256)
    parser.add_argument('--fusion_dim_O', type=int, default=512)
    parser.add_argument('--single_modal_dim', type=int, default=512)
    parser.add_argument('--dataset_name', default='IEMOCAP', type=str)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--modality', type=str, default='multimodal')
    parser.add_argument('--simple_nn', action='store_true', help='Use Simple NN (No GNN, Context-free)')
    parser.add_argument('--standard_gnn', action='store_true', help='Use Standard GNN (Fully connected)')
    parser.add_argument('--edl_r2', action='store_true', help='Use EDL(R2) Loss and Dirichlet distribution')
 
    args = parser.parse_args()
    seed_everything(args.seed)
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    args.use_text = args.modality in ('multimodal', 'text')
    args.use_audio = args.modality in ('multimodal', 'audio')
    
    # --- モード設定 ---
    if args.train_ma_test_nma:
        train_nma_flag = False
        test_nma_flag = True
        mode_dir_name = 'ma2nma'
        logger_suffix = "" 
    else:
        train_nma_flag = args.nma
        test_nma_flag = args.nma
        mode_dir_name = 'nma' if args.nma else 'default'
        logger_suffix = "_nma" if args.nma else ""

    # --- アーキテクチャ設定 ---
    if args.edl_r2:
        arch_name = "edl_r2" 
    elif args.simple_nn:
        arch_name = "simple_nn"
    elif args.standard_gnn:
        arch_name = "standard_gnn"
    else:
        arch_name = "dag_erc"
    
    base_save_dir = os.path.join('saved_models', f'seed{args.seed}', mode_dir_name)
    if args.edl_r2:
        save_dir = os.path.join(base_save_dir, 'edl_r2')
    elif args.modality == 'multimodal' and arch_name != 'dag_erc':
        save_dir = os.path.join(base_save_dir, arch_name)
    else:
        save_dir = os.path.join(base_save_dir, args.modality)
    os.makedirs(save_dir, exist_ok=True)
     
    if args.log_file_name:
        log_file_path = os.path.join(save_dir, args.log_file_name)
    else:
        log_file_path = os.path.join(save_dir, f'logging_{args.eval_metric}_fold{args.test_session}_seed{args.seed}{logger_suffix}.log')

    logger = get_logger(log_file_path)
    logger.info(f'Log file: {log_file_path}')
    logger.info(f'Architecture: {arch_name}')
    logger.info(f'Optimizer: Adafactor + Noam Scheduler (Warmup 200)') # ログにも記載
    logger.info(args)

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    
    train_loader, valid_loader, test_loader, speaker_vocab, label_vocab = get_multimodal_loaders(
        data_dir=args.data_dir, 
        batch_size=batch_size, 
        num_workers=0, 
        args=args,
        test_session=args.test_session,
        dev_ratio=args.dev_ratio,
        nma=False,
        train_nma=train_nma_flag,
        test_nma=test_nma_flag
    )
    
    n_classes = len(label_vocab['itos'])
    model = DAGERC_multimodal(args, n_classes)
    if cuda: model.cuda()

    if args.edl_r2:
        criterion = EDL_R2_Loss(annealing_step=args.epochs, lambda_coef=0.8, device='cuda' if cuda else 'cpu')
    else:
        criterion = SoftNLLLoss(reduction='sum')

    # Optimizer & Scheduler (Adafactor + Noam)
    optimizer = Adafactor(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-2, 
        scale_parameter=False, 
        relative_step=False, 
        warmup_init=False
    )
    scheduler = get_noam_schedule_with_warmup(optimizer, num_warmup_steps=200)

    if args.eval_metric == 'loss':
        best_val_score = float('inf')
    else:
        best_val_score = -float('inf')
        
    best_test_f1 = 0.0
    best_test_nll = 0.0 
    best_test_acc = 0.0
    best_test_kl = 0.0 # ★追加
    best_epoch = -1

    analysis_data = None
    
    for e in range(n_epochs):
        start_time = time.time()

        args.nma = train_nma_flag 
        # ★修正: 戻り値 t_kl を受け取る
        t_loss, t_nll, t_acc, t_kl, _, _, t_f1, _, _, _ = train_or_eval_model(
            model, criterion, train_loader, e, cuda, args, optimizer, scheduler, True
        )
        
        # ★修正: 戻り値 v_kl を受け取る
        v_loss, v_nll, v_acc, v_kl, _, _, v_f1, _, _, _ = train_or_eval_model(
            model, criterion, valid_loader, e, cuda, args
        )
        
        args.nma = test_nma_flag
        # ★修正: 戻り値 test_kl を受け取る
        test_loss, test_nll, test_acc, test_kl, test_labels, test_preds, test_f1, test_probs, test_softs, test_texts = train_or_eval_model(
            model, criterion, test_loader, e, cuda, args
        )

        # ★修正: ログにKLを追加
        logger.info(
            f"Ep {e+1}: "
            f"Train [NLL {t_nll:.4f} KL {t_kl:.4f} F1 {t_f1:.2f}] | "
            f"Val [NLL {v_nll:.4f} KL {v_kl:.4f} F1 {v_f1:.2f}] | "
            f"Test [NLL {test_nll:.4f} KL {test_kl:.4f} F1 {test_f1:.2f}] | "
            f"Time {time.time() - start_time:.1f}s"
        )

        is_best = False
        if args.eval_metric == 'loss':
            if v_nll < best_val_score:
                best_val_score = v_nll
                is_best = True
        else:
            if v_f1 > best_val_score:
                best_val_score = v_f1
                is_best = True
        
        if is_best:
            best_epoch = e + 1
            best_test_f1 = test_f1
            best_test_nll = test_nll
            best_test_acc = test_acc
            best_test_kl = test_kl # ★追加
            
            analysis_data = {
                'epoch': best_epoch,
                'fold': args.test_session,
                'true_ids': test_labels,
                'pred_ids': test_preds,
                'pred_probs': test_probs,
                'true_softs': test_softs,
                'texts': test_texts,
                'kl_div': test_kl # ★追加
            }

    logger.info('finish training!')
    logger.info(f"Best Epoch: {best_epoch}")
    
    if args.eval_metric == 'loss':
        logger.info(f"Best Val NLL: {best_val_score:.4f}")
    else:
        logger.info(f"Best Val F1: {best_val_score:.2f}")

    logger.info(f"Test F1 at Best Val: {best_test_f1:.2f}")
    logger.info(f"Test NLL at Best Val: {best_test_nll:.4f}")
    logger.info(f"Test KL at Best Val: {best_test_kl:.4f}") # ★追加
    logger.info(f"Test Acc at Best Val: {best_test_acc:.2f}")
    
    if analysis_data is not None:
        result_filename = f'test_results_fold{args.test_session}.npy'
        result_path = os.path.join(save_dir, result_filename)
        np.save(result_path, analysis_data)
        logger.info(f"Saved best analysis results to {result_path}")