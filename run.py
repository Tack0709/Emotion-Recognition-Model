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

    loss_function = nn.KLDivLoss(reduction='sum')
    optimizer = AdamW(model.parameters() , lr=args.lr)

    all_fscore = []

    for e in range(n_epochs):
        start_time = time.time()

        t_loss, t_acc, _, _, t_f1 = train_or_eval_model(model, loss_function, train_loader, e, cuda, args, optimizer, True)
        v_loss, v_acc, _, _, v_f1 = train_or_eval_model(model, loss_function=loss_function, dataloader=valid_loader, epoch=e, cuda=cuda, args=args)
        test_loss, test_acc, _, _, test_f1 = train_or_eval_model(model, loss_function=loss_function, dataloader=test_loader, epoch=e, cuda=cuda, args=args)

        all_fscore.append([v_f1, test_f1])

        logger.info( 'Epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
            format(e + 1, t_loss, t_acc, t_f1, v_loss, v_acc, v_f1, test_loss, test_acc, test_f1, round(time.time() - start_time, 2)))

    logger.info('finish training!')
    all_fscore = sorted(all_fscore, key=lambda x: x[0], reverse=True) # Valid基準
    logger.info('Best Val F1: {}'.format(all_fscore[0][1]))
    logger.info('Best Test F1 (at Best Val): {}'.format(all_fscore[0][1])) # all_fscore[0][1] is Test F1 when Val F1 is max