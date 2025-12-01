import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.optim import AdamW
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

if __name__ == '__main__':

    path = './saved_models/'
    if not os.path.exists(path): os.makedirs(path)

    parser = argparse.ArgumentParser()
    
    # データ設定
    parser.add_argument('--data_dir', default='output_data', type=str, help='path to data directory')

    # 評価指標設定
    parser.add_argument('--eval_metric', type=str, default='f1', choices=['f1', 'loss'], 
                        help='Metric to select best model (f1: Maximize F1, loss: Minimize Loss)')
    parser.add_argument('--log_file_name', type=str, default=None, 
                        help='Name of the log file. If None, generated automatically based on metric and seed.')

    # GNN & Model params
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')
    parser.add_argument('--gnn_layers', type=int, default=4, help='Number of gnn layers.')
    parser.add_argument('--attn_type', type=str, default='rgcn', choices=['dotprod','linear','bilinear', 'rgcn'], help='Feature size.')
    parser.add_argument('--no_rel_attn',  action='store_true', default=False, help='no relation for edges' )
    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--windowp', type=int, default=1)
    parser.add_argument('--windowf', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=50, metavar='E', help='number of epochs')
    parser.add_argument('--tensorboard', action='store_true', default=False)
    parser.add_argument('--nodal_att_type', type=str, default=None, choices=['global','past'])

    # Dimensions
    parser.add_argument('--text_dim', type=int, default=768, help='Text feature dimension (BERT-base)')
    parser.add_argument('--audio_dim', type=int, default=768, help='Audio feature dimension (Wav2Vec 2.0-base)')
    parser.add_argument('--fusion_dim_D', type=int, default=256, help='Bilinear D dim')
    parser.add_argument('--fusion_dim_O', type=int, default=512, help='Bilinear Output dim (to fc1)')

    parser.add_argument('--dataset_name', default='IEMOCAP', type=str)
    parser.add_argument('--seed', type=int, default=100)

    args = parser.parse_args()
    print(args)
    
    seed_everything(args.seed)
    
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    
    # #############################################################
    # ⬇️ 変更点: ログファイル名にシード値を含めるように修正
    # #############################################################
    if args.log_file_name:
        # 指定があればそれを使う
        log_file_path = os.path.join(path, args.log_file_name)
    else:
        # 指定がなければ自動生成: logging_{指標}_seed{シード値}.log
        # 例: logging_f1_seed100.log
        log_file_path = os.path.join(path, f'logging_{args.eval_metric}_seed{args.seed}.log')

    logger = get_logger(log_file_path)
    logger.info(f'Log file: {log_file_path}')
    logger.info(f'Start training on {args.data_dir}')
    logger.info(f'Evaluation Metric: {args.eval_metric}')
    logger.info(f'Seed: {args.seed}') # シード値もログに記録
    logger.info(args)
    # #############################################################

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    
    train_loader, valid_loader, test_loader, speaker_vocab, label_vocab = get_multimodal_loaders(
        data_dir=args.data_dir, 
        batch_size=batch_size, 
        num_workers=0, 
        args=args
    )
    
    n_classes = len(label_vocab['itos'])
    print('n_classes:', n_classes)

    print('building model..')
    model = DAGERC_multimodal(args, n_classes)

    if cuda:
        model.cuda()

    loss_function = nn.KLDivLoss(reduction='sum')
    optimizer = AdamW(model.parameters() , lr=args.lr)

    if args.eval_metric == 'loss':
        best_val_score = float('inf')
    else:
        best_val_score = -float('inf')
        
    best_test_f1 = 0.0
    best_test_loss = 0.0
    best_epoch = -1

    for e in range(n_epochs):
        start_time = time.time()

        # train
        t_loss, t_acc, _, _, t_f1 = train_or_eval_model(
            model, loss_function, train_loader, e, cuda, args, optimizer, True
        )
        # valid
        v_loss, v_acc, _, _, v_f1 = train_or_eval_model(
            model, loss_function, valid_loader, e, cuda, args
        )
        # test
        test_loss, test_acc, _, _, test_f1 = train_or_eval_model(
            model, loss_function, test_loader, e, cuda, args
        )

        logger.info(
            f"Ep {e+1}: "
            f"Train [Loss {t_loss:.4f} F1 {t_f1:.2f}] | "
            f"Val [Loss {v_loss:.4f} F1 {v_f1:.2f}] | "
            f"Test [Loss {test_loss:.4f} F1 {test_f1:.2f}] | "
            f"Time {time.time() - start_time:.1f}s"
        )

        is_best = False
        if args.eval_metric == 'loss':
            if v_loss < best_val_score:
                best_val_score = v_loss
                is_best = True
        else:
            if v_f1 > best_val_score:
                best_val_score = v_f1
                is_best = True
        
        if is_best:
            best_epoch = e + 1
            best_test_f1 = test_f1
            best_test_loss = test_loss
            # torch.save(model.state_dict(), os.path.join(path, f'best_model_seed{args.seed}.pt'))

    logger.info('finish training!')
    logger.info(f"Best Epoch: {best_epoch}")
    
    if args.eval_metric == 'loss':
        logger.info(f"Best Val Loss: {best_val_score:.4f}")
    else:
        logger.info(f"Best Val F1: {best_val_score:.2f}")

    logger.info(f"Test F1 at Best Val: {best_test_f1:.2f}")
    logger.info(f"Test Loss at Best Val: {best_test_loss:.4f}")