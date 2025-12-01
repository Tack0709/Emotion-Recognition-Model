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
    logger.addHandler(sh)
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
    
    # データセットディレクトリ指定 (dataset_nameの代わりにこれを使う)
    parser.add_argument('--data_dir', default='output_data', type=str, help='path to data directory')

    # Training params評価指標の選択
    parser.add_argument('--eval_metric', type=str, default='f1', choices=['f1', 'loss'], 
                        help='Metric to select best model (f1: Maximize F1, loss: Minimize NLL)')
    
    # GNN & Model params
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of gnn layers.')
    parser.add_argument('--attn_type', type=str, default='rgcn', choices=['dotprod','linear','bilinear', 'rgcn'], help='Feature size.')
    parser.add_argument('--no_rel_attn',  action='store_true', default=False, help='no relation for edges' )
    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--windowp', type=int, default=1)
    parser.add_argument('--windowf', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=30, metavar='E', help='number of epochs')
    parser.add_argument('--tensorboard', action='store_true', default=False)
    parser.add_argument('--nodal_att_type', type=str, default=None, choices=['global','past'])

    # --- 修正: 次元数を768に変更 ---
    parser.add_argument('--text_dim', type=int, default=768, help='Text feature dimension (BERT-base)')
    parser.add_argument('--audio_dim', type=int, default=768, help='Audio feature dimension (Wav2Vec 2.0-base)')
    parser.add_argument('--fusion_dim_D', type=int, default=256, help='Bilinear D dim')
    parser.add_argument('--fusion_dim_O', type=int, default=512, help='Bilinear Output dim (to fc1)')

    # dataset_name はログ出力用に残すが、読み込みには data_dir を使う
    parser.add_argument('--dataset_name', default='IEMOCAP', type=str)
    parser.add_argument('--seed', type=int, default=100)

    args = parser.parse_args()
    print(args)
    
    seed_everything(args.seed)
    
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    
    logger = get_logger(os.path.join(path, 'logging.log'))
    logger.info(f'Start training on {args.data_dir}')
    logger.info(args)

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    
    # 修正: data_dir を渡す
    train_loader, valid_loader, test_loader, speaker_vocab, label_vocab = get_multimodal_loaders(
        data_dir=args.data_dir, 
        batch_size=batch_size, 
        num_workers=0, 
        args=args
    )
    
    n_classes = len(label_vocab['itos']) # 5クラス
    print('n_classes:', n_classes)

    print('building model..')
    model = DAGERC_multimodal(args, n_classes)

    if cuda:
        model.cuda()

    loss_fn = nn.KLDivLoss(reduction='sum')
    optimizer = AdamW(model.parameters() , lr=args.lr)

    if args.eval_metric == 'loss':
        best_val_score = float('inf')
    else:
        best_val_score = 0.0
        
    best_test_score_at_best_val = 0.0 # F1スコア記録用
    best_test_loss_at_best_val = float('inf') # Loss記録用
    
    all_fscore = []

    for e in range(args.epochs):
        start = time.time()
        
        t_loss, t_acc, _, _, t_f1 = train_or_eval_model(model, loss_fn, train_loader, e, args.cuda, args, optimizer, True)
        v_loss, v_acc, _, _, v_f1 = train_or_eval_model(model, loss_fn, valid_loader, e, args.cuda, args)
        test_loss, test_acc, _, _, test_f1 = train_or_eval_model(model, loss_fn, test_loader, e, args.cuda, args)

        # <--- 追加: ログにLossとF1の両方を表示
        logger.info(
            f"Ep {e+1}: "
            f"Train [Loss {t_loss:.4f} F1 {t_f1:.2f}] | "
            f"Val [Loss {v_loss:.4f} F1 {v_f1:.2f}] | "
            f"Test [Loss {test_loss:.4f} F1 {test_f1:.2f}] | "
            f"Time {time.time()-start:.1f}s"
        )

        # -----------------------------------------------------------------
        # <--- 変更: ベストモデルの更新判定ロジック
        # -----------------------------------------------------------------
        update_best = False
        if args.eval_metric == 'loss':
            # Lossが小さいほど良い
            if v_loss < best_val_score:
                best_val_score = v_loss
                update_best = True
        else:
            # F1が大きいほど良い (デフォルト)
            if v_f1 > best_val_score:
                best_val_score = v_f1
                update_best = True
        
        if update_best:
            # ベスト更新時のテストスコアを記録
            best_test_score_at_best_val = test_f1
            best_test_loss_at_best_val = test_loss
            
            # (必要ならここで torch.save(model.state_dict(), ...) でモデル保存)

    logger.info('finish training!')
    
    # -----------------------------------------------------------------
    # <--- 変更: 最終結果の表示
    # -----------------------------------------------------------------
    if args.eval_metric == 'loss':
        logger.info(f"Best Val Loss: {best_val_score:.4f}")
        logger.info(f"Test Loss at Best Val: {best_test_loss_at_best_val:.4f}")
        logger.info(f"Test F1 at Best Val: {best_test_score_at_best_val:.2f}")
    else:
        logger.info(f"Best Val F1: {best_val_score:.2f}")
        logger.info(f"Test F1 at Best Val: {best_test_score_at_best_val:.2f}")