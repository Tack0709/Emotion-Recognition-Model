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
        loss = -torch.sum(targets * log_probs, dim=1)
        if self.reduction == 'sum': return loss.sum()
        elif self.reduction == 'mean': return loss.mean()
        return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', default='output_data', type=str)
    parser.add_argument('--eval_metric', type=str, default='loss', choices=['f1', 'loss'])
    parser.add_argument('--test_session', type=int, default=5, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--log_file_name', type=str, default=None)
    parser.add_argument('--dev_ratio', type=float, default=0.2)
    parser.add_argument('--nma', action='store_true', help='Test on NMA data as well')

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

    parser.add_argument('--text_dim', type=int, default=768)
    parser.add_argument('--audio_dim', type=int, default=768)
    parser.add_argument('--fusion_dim_D', type=int, default=256)
    parser.add_argument('--fusion_dim_O', type=int, default=512)
    parser.add_argument('--single_modal_dim', type=int, default=512)

    parser.add_argument('--dataset_name', default='IEMOCAP', type=str)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--modality', type=str, default='multimodal')
    
    parser.add_argument('--simple_nn', action='store_true')
    parser.add_argument('--standard_gnn', action='store_true')
    parser.add_argument('--edl_r2', action='store_true')
 
    args = parser.parse_args()
    seed_everything(args.seed)
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    args.use_text = args.modality in ('multimodal', 'text')
    args.use_audio = args.modality in ('multimodal', 'audio')
    if not (args.use_text or args.use_audio):
        raise ValueError("少なくとも1つのモダリティを有効にしてください。")
    
    if args.edl_r2:
        if args.modality != 'multimodal' or args.simple_nn or args.standard_gnn:
            raise ValueError("エラー: --edl_r2 は multimodal かつ DAG-ERC の時のみ使用可能です。")
        arch_name = "edl_r2" 
    elif args.simple_nn:
        arch_name = "simple_nn"
    elif args.standard_gnn:
        arch_name = "standard_gnn"
    else:
        arch_name = "dag_erc"
    
    mode_dir = 'nma' if args.nma else 'default'
    base_save_dir = os.path.join('saved_models', f'seed{args.seed}', mode_dir)

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
        nma_suffix = "_nma" if args.nma else ""
        log_file_path = os.path.join(save_dir, f'logging_{args.eval_metric}_fold{args.test_session}_seed{args.seed}{nma_suffix}.log')

    logger = get_logger(log_file_path)
    logger.info(f'Log file: {log_file_path}')
    logger.info(f'Architecture: {arch_name}')
    logger.info(f'Evaluation Metric: {args.eval_metric}')
    logger.info(args)

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    
    # =========================================================================
    # ★修正: ローダーの取得
    # =========================================================================
    
    logger.info("Loading Data...")
    # nmaフラグは渡すが、dataset.py内でのフィルタリングは無効化されている前提
    train_loader, valid_loader, test_loader, speaker_vocab, label_vocab = get_multimodal_loaders(
        data_dir=args.data_dir, 
        batch_size=batch_size, 
        num_workers=0, 
        args=args,
        test_session=args.test_session,
        dev_ratio=args.dev_ratio,
        nma=args.nma 
    )
    
    # =========================================================================

    n_classes = len(label_vocab['itos'])
    print('n_classes:', n_classes)

    print('building model..')
    model = DAGERC_multimodal(args, n_classes)
    if cuda: model.cuda()

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
        
    best_epoch = -1
    
    best_res_ma = None
    best_res_nma = None

    for e in range(n_epochs):
        start_time = time.time()

        # Train (trainer内でXXXのLossはゼロ化される)
        t_loss, t_nll, t_acc, _, _, t_f1, _, _, _, _ = train_or_eval_model(
            model, criterion, train_loader, e, cuda, args, optimizer, scheduler, True
        )
        
        # Validation
        v_loss, v_nll, v_acc, _, _, v_f1, _, _, _, _ = train_or_eval_model(
            model, criterion, valid_loader, e, cuda, args
        )
        
        # Test
        ma_loss, ma_nll, ma_acc, ma_labels, ma_preds, ma_f1, ma_probs, ma_softs, ma_texts, nma_results = train_or_eval_model(
            model, criterion, test_loader, e, cuda, args
        )
        
        # NMAのログ文字列作成
        nma_log_str = ""
        if args.nma and nma_results:
            nma_log_str = f" | NMA [NLL {nma_results['nll']:.4f} Cnt {nma_results['count']}]"
        
        # ログ表示
        logger.info(
            f"Ep {e+1}: "
            f"Tr [NLL {t_nll:.3f} Acc {t_acc:.1f}] | "
            f"Val [NLL {v_nll:.3f} Acc {v_acc:.1f}] | "
            f"MA [NLL {ma_nll:.3f} F1 {ma_f1:.1f} Acc {ma_acc:.1f}]"
            f"{nma_log_str} | "
            f"Time {time.time() - start_time:.1f}s"
        )

        # ベストモデル更新判定 (MAのみに基づく)
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
            
            # MAの結果を保存
            best_res_ma = {
                'epoch': best_epoch, 'fold': args.test_session,
                'f1': ma_f1, 'nll': ma_nll, 'acc': ma_acc,
                'true_ids': ma_labels, 'pred_ids': ma_preds,
                'pred_probs': ma_probs, 'true_softs': ma_softs, 'texts': ma_texts
            }
            
            # NMAの結果も保存
            if args.nma and nma_results:
                best_res_nma = {
                    'epoch': best_epoch, 'fold': args.test_session,
                    'nll': nma_results['nll'], 
                    'count': nma_results['count'],
                    'true_ids': nma_results['labels'], # ★追加: ラベル（全て5）
                    'pred_ids': nma_results['preds'],
                    'pred_probs': nma_results['probs'], # ★追加: 分布
                    'true_softs': nma_results['softs'], # ★追加: ソフトラベル
                    'texts': nma_results['texts']       # ★追加: テキスト
                }

    logger.info('finish training!')
    logger.info(f"Best Epoch: {best_epoch}")
    
    # 最終結果の表示
    if best_res_ma:
        logger.info(f"Best Val Score ({args.eval_metric}): {best_val_score:.4f}")
        logger.info(f"Test F1 at Best Val: {best_res_ma['f1']:.2f}")
        logger.info(f"Test NLL at Best Val: {best_res_ma['nll']:.4f}")
        logger.info(f"Test Acc at Best Val: {best_res_ma['acc']:.2f}")
        
        # 保存
        ma_path = os.path.join(save_dir, f'test_results_fold{args.test_session}_MA.npy')
        np.save(ma_path, best_res_ma)
        logger.info(f"Saved MA results: {ma_path}")
        
    if best_res_nma:
        logger.info(f"NMA NLL at Best Val: {best_res_nma['nll']:.4f}")
        
        # 保存
        nma_path = os.path.join(save_dir, f'test_results_fold{args.test_session}_NMA.npy')
        np.save(nma_path, best_res_nma)
        logger.info(f"Saved NMA results: {nma_path}")