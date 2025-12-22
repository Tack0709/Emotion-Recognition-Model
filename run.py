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
    
    parser.add_argument('--data_dir', default='output_data', type=str, help='path to data directory')
    parser.add_argument('--eval_metric', type=str, default='loss', choices=['f1', 'loss'], 
                        help='Metric to select best model (f1: Maximize F1, loss: Minimize NLL)')
    parser.add_argument('--test_session', type=int, default=5, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--log_file_name', type=str, default=None)
    parser.add_argument('--dev_ratio', type=float, default=0.2)
    parser.add_argument('--nma', action='store_true', help='Include "xxx" labels and train with soft labels only')

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
 
    args = parser.parse_args()
    seed_everything(args.seed)
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    args.use_text = args.modality in ('multimodal', 'text')
    args.use_audio = args.modality in ('multimodal', 'audio')
    if not (args.use_text or args.use_audio):
        raise ValueError("少なくとも1つのモダリティを有効にしてください。")
    
    mode_dir = 'nma' if args.nma else 'default'
    save_dir = os.path.join('saved_models', f'seed{args.seed}', mode_dir, args.modality)
    os.makedirs(save_dir, exist_ok=True)
     
    if args.log_file_name:
        log_file_path = os.path.join(save_dir, args.log_file_name)
    else:
        nma_suffix = "_nma" if args.nma else ""
        log_file_path = os.path.join(save_dir, f'logging_{args.eval_metric}_fold{args.test_session}_seed{args.seed}{nma_suffix}.log')

    logger = get_logger(log_file_path)
    logger.info(f'Log file: {log_file_path}')
    logger.info(f'Evaluation Metric: {args.eval_metric}')
    logger.info(f'Validation Ratio: {args.dev_ratio}')
    logger.info(f'NMA Mode: {args.nma}')
    logger.info(f'Modality: {args.modality} (text={args.use_text}, audio={args.use_audio})')
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
        nma=args.nma
    )
    
    n_classes = len(label_vocab['itos'])
    print('n_classes:', n_classes)

    print('building model..')
    model = DAGERC_multimodal(args, n_classes)
    if cuda: model.cuda()

    loss_fn = SoftNLLLoss(reduction='sum')
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

    for e in range(n_epochs):
        start_time = time.time()

        t_loss, t_nll, t_acc, _, _, t_f1 = train_or_eval_model(
            model, loss_fn, train_loader, e, cuda, args, optimizer, scheduler, True
        )
        v_loss, v_nll, v_acc, _, _, v_f1 = train_or_eval_model(
            model, loss_fn, valid_loader, e, cuda, args
        )
        test_loss, test_nll, test_acc, _, _, test_f1 = train_or_eval_model(
            model, loss_fn, test_loader, e, cuda, args
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

    logger.info('finish training!')
    logger.info(f"Best Epoch: {best_epoch}")
    
    if args.eval_metric == 'loss':
        logger.info(f"Best Val NLL: {best_val_score:.4f}")
    else:
        logger.info(f"Best Val F1: {best_val_score:.2f}")

    logger.info(f"Test F1 at Best Val: {best_test_f1:.2f}")
    logger.info(f"Test NLL at Best Val: {best_test_nll:.4f}")
    logger.info(f"Test Acc at Best Val: {best_test_acc:.2f}")