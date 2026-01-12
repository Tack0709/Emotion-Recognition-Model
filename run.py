import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np, argparse, time, random, logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import copy

from dataloader import get_multimodal_loaders
from model import DAGERC_multimodal
from trainer import train_or_eval_model
from loss import EDL_R2_Loss

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
        # targets: Soft Labels (Probabilities)
        # log_probs: Log Probabilities
        loss = -torch.sum(targets * log_probs, dim=1)
        if self.reduction == 'sum': return loss.sum()
        elif self.reduction == 'mean': return loss.mean()
        return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', default='output_data', type=str, help='path to data directory')
    parser.add_argument('--eval_metric', type=str, default='loss', choices=['f1', 'loss'], 
                        help='Metric to select best model (f1: Maximize F1, loss: Minimize NLL)')
    parser.add_argument('--test_session', type=int, default=5, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--log_file_name', type=str, default=None)
    parser.add_argument('--dev_ratio', type=float, default=0.2)
    parser.add_argument('--nma', action='store_true', help='Include "xxx" labels and train with soft labels only')
    
    # ★追加: 学習はMA、テストはNMAで行うためのフラグ
    parser.add_argument('--train_ma_test_nma', action='store_true', help='Train on MA data (exclude XXX), but Test on NMA data (include XXX)')

    # GNN & Model params
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--mlp_layers', type=int, default=2)
    parser.add_argument('--gnn_layers', type=int, default=4)
    parser.add_argument('--attn_type', type=str, default='rgcn', choices=['dotprod','linear','bilinear', 'rgcn'])
    parser.add_argument('--no_rel_attn',  action='store_true')
    parser.add_argument('--windowp', type=int, default=1)
    parser.add_argument('--windowf', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=5.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--nodal_att_type', type=str, default=None, choices=['global','past'])

    # Dimensions
    parser.add_argument('--text_dim', type=int, default=768)
    parser.add_argument('--audio_dim', type=int, default=768)
    parser.add_argument('--fusion_dim_D', type=int, default=256)
    parser.add_argument('--fusion_dim_O', type=int, default=512)
    parser.add_argument('--single_modal_dim', type=int, default=512,
                        help='単一モダリティ時に 768 次元を圧縮する中間次元')

    parser.add_argument('--dataset_name', default='IEMOCAP', type=str)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--modality', type=str, default='multimodal', help='保存先や処理をモダリティ別に分岐')
    
    # アブレーション実験用のフラグ
    parser.add_argument('--simple_nn', action='store_true', help='Use Simple NN (No GNN, Context-free)')
    parser.add_argument('--standard_gnn', action='store_true', help='Use Standard GNN (Fully connected)')
    
    # EDL(R2)用フラグ
    parser.add_argument('--edl_r2', action='store_true', help='Use EDL(R2) Loss and Dirichlet distribution')
 
    args = parser.parse_args()
    seed_everything(args.seed)
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    args.use_text = args.modality in ('multimodal', 'text')
    args.use_audio = args.modality in ('multimodal', 'audio')
    if not (args.use_text or args.use_audio):
        raise ValueError("少なくとも1つのモダリティを有効にしてください。")
    
    # --- モード設定の決定 ---
    if args.train_ma_test_nma:
        # 学習はMA(False)、テストはNMA(True)
        train_nma_flag = False
        test_nma_flag = True
        mode_dir_name = 'ma2nma'
        logger_suffix = "" 
    else:
        # 通常モード（両方同じ）
        train_nma_flag = args.nma
        test_nma_flag = args.nma
        mode_dir_name = 'nma' if args.nma else 'default'
        logger_suffix = "_nma" if args.nma else ""

    # --- アーキテクチャ名の決定 ---
    if args.edl_r2:
        if args.modality != 'multimodal' or args.simple_nn or args.standard_gnn:
            raise ValueError("エラー: --edl_r2 は multimodal かつ DAG-ERC (simple_nn/standard_gnnなし) の時のみ使用可能です。")
        arch_name = "edl_r2" 
    elif args.simple_nn:
        arch_name = "simple_nn"
    elif args.standard_gnn:
        arch_name = "standard_gnn"
    else:
        arch_name = "dag_erc"
    
    # --- 保存先ディレクトリの構築 ---
    # base: saved_models/seedXXX/[default|nma|ma2nma]/
    base_save_dir = os.path.join('saved_models', f'seed{args.seed}', mode_dir_name)

    # フォルダ分けルール
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
    logger.info(f'Mode Directory: {mode_dir_name}')
    logger.info(f'Evaluation Metric: {args.eval_metric}')
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
        nma=False, # default引数は無視されます
        train_nma=train_nma_flag, # ★学習用設定
        test_nma=test_nma_flag    # ★テスト用設定
    )
    
    n_classes = len(label_vocab['itos'])
    print('n_classes:', n_classes)

    print('building model..')
    model = DAGERC_multimodal(args, n_classes)
    if cuda: model.cuda()

    # --- 損失関数の定義 ---
    if args.edl_r2:
        logger.info("Using EDL(R2) Loss")
        criterion = EDL_R2_Loss(annealing_step=args.epochs, lambda_coef=0.8, device='cuda' if cuda else 'cpu')
    else:
        criterion = SoftNLLLoss(reduction='sum')

    optimizer = AdamW(model.parameters() , lr=args.lr, weight_decay=1e-2)

    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * 0.1)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    if args.eval_metric == 'loss':
        best_val_score = float('inf')
    else:
        best_val_score = -float('inf')
        
    best_test_f1 = 0.0
    best_test_nll = 0.0 
    best_test_acc = 0.0
    best_epoch = -1

    analysis_data = None
    
    for e in range(n_epochs):
        start_time = time.time()

        # ★学習・検証フェーズの設定を適用
        args.nma = train_nma_flag 
        t_loss, t_nll, t_acc, _, _, t_f1, _, _, _ = train_or_eval_model(
            model, criterion, train_loader, e, cuda, args, optimizer, scheduler, True
        )
        v_loss, v_nll, v_acc, _, _, v_f1, _, _, _ = train_or_eval_model(
            model, criterion, valid_loader, e, cuda, args
        )
        
        # ★テストフェーズの設定を適用
        args.nma = test_nma_flag
        test_loss, test_nll, test_acc, test_labels, test_preds, test_f1, test_probs, test_softs, test_texts = train_or_eval_model(
            model, criterion, test_loader, e, cuda, args
        )

        logger.info(
            f"Ep {e+1}: "
            f"Train [NLL {t_nll:.4f} F1 {t_f1:.2f} Acc {t_acc:.2f}] | "
            f"Val [NLL {v_nll:.4f} F1 {v_f1:.2f} Acc {v_acc:.2f}] | "
            f"Test [NLL {test_nll:.4f} F1 {test_f1:.2f} Acc {test_acc:.2f}] | "
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
            
            analysis_data = {
                'epoch': best_epoch,
                'fold': args.test_session,
                'true_ids': test_labels,
                'pred_ids': test_preds,
                'pred_probs': test_probs,
                'true_softs': test_softs,
                'texts': test_texts
            }

    logger.info('finish training!')
    logger.info(f"Best Epoch: {best_epoch}")
    
    if args.eval_metric == 'loss':
        logger.info(f"Best Val NLL: {best_val_score:.4f}")
    else:
        logger.info(f"Best Val F1: {best_val_score:.2f}")

    logger.info(f"Test F1 at Best Val: {best_test_f1:.2f}")
    logger.info(f"Test NLL at Best Val: {best_test_nll:.4f}")
    logger.info(f"Test Acc at Best Val: {best_test_acc:.2f}")
    
    if analysis_data is not None:
        result_filename = f'test_results_fold{args.test_session}.npy'
        result_path = os.path.join(save_dir, result_filename)
    
        np.save(result_path, analysis_data)
        logger.info(f"Saved best analysis results to {result_path}")