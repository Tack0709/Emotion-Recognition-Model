import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
# (get_logger, seed_everything は DAG-ERC と同様)
# ...

# 変更: dataset.py と dataloader.py を新しいものに置き換える
from dataset import MultimodalDAGDataset # 修正版 dataset.py
from dataloader import get_multimodal_loaders # 修正版 dataloader.py (後述)
from model import DAGERC_multimodal # 修正版 model.py
from trainer import train_or_eval_model # 修正版 trainer.py

from transformers import AdamW
import copy

seed = 100
# (get_logger, seed_everything の実装 ...)

if __name__ == '__main__':

    path = './saved_models/'

    parser = argparse.ArgumentParser()
    # DAG-ERC の引数
    parser.add_argument('--hidden_dim', type = int, default=300)
    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of gnn layers.')
    parser.add_argument('--attn_type', type=str, default='rgcn', choices=['dotprod','linear','bilinear', 'rgcn'], help='Feature size.')
    parser.add_argument('--no_rel_attn',  action='store_true', default=False, help='no relation for edges' )
    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--dataset_name', default='IEMOCAP', type= str)
    parser.add_argument('--windowp', type=int, default=1)
    parser.add_argument('--windowf', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate') # (調整が必要)
    parser.add_argument('--dropout', type=float, default=0.2, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=30, metavar='E', help='number of epochs')
    parser.add_argument('--tensorboard', action='store_true', default=False)
    parser.add_argument('--nodal_att_type', type=str, default=None, choices=['global','past'])

    # 追加: マルチモーダルと融合層のための引数
    parser.add_argument('--text_dim', type=int, default=1024, help='Text feature dimension (e.g., RoBERTa large)')
    parser.add_argument('--audio_dim', type=int, default=1024, help='Audio feature dimension (e.g., Wav2Vec 2.0 large)')
    parser.add_argument('--fusion_dim_D', type=int, default=256, help='Bilinear D dim')
    parser.add_argument('--fusion_dim_O', type=int, default=512, help='Bilinear Output dim (to fc1)')

    args = parser.parse_args()
    print(args)
    
    seed_everything()
    
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    # (ロガー設定 ...)

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    
    # 変更: 新しいデータローダーを使用
    # (注: dataloader.py も MultimodalDAGDataset を使うように修正が必要)
    train_loader, valid_loader, test_loader, speaker_vocab, label_vocab = get_multimodal_loaders(
        dataset_name=args.dataset_name, batch_size=batch_size, num_workers=0, args=args
    )
    
    n_classes = len(label_vocab['itos']) # 5クラス
    print('n_classes:', n_classes)

    print('building model..')
    model = DAGERC_multimodal(args, n_classes) # 修正版モデル

    if cuda:
        model.cuda()

    # 変更: 損失関数を KLDivLoss に (モデル出力が log_softmax のため)
    # ERC-SLT22 同様 'sum' を使い、trainer で正規化
    loss_function = nn.KLDivLoss(reduction='sum')
    
    optimizer = AdamW(model.parameters() , lr=args.lr)

    best_fscore = 0.
    all_fscore = []

    for e in range(n_epochs):
        start_time = time.time()

        # trainer の戻り値に合わせる (Fスコアのみ)
        train_loss, train_acc, _, _, train_fscore = train_or_eval_model(model, loss_function,
                                                                        train_loader, e, cuda,
                                                                        args, optimizer, True)
        valid_loss, valid_acc, _, _, valid_fscore= train_or_eval_model(model, loss_function,
                                                                        valid_loader, e, cuda, args)
        test_loss, test_acc, test_label, test_pred, test_fscore= train_or_eval_model(model,loss_function, 
                                                                        test_loader, e, cuda, args)

        all_fscore.append([valid_fscore, test_fscore])

        print( 'Epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
            format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc,
            test_fscore, round(time.time() - start_time, 2)))

        e += 1

    print('finish training!')
    all_fscore = sorted(all_fscore, key=lambda x: x[0], reverse=True) # Valid基準
    print('Best F-Score based on validation:{}'.format(all_fscore[0][1]))
    print('Best F-Score based on test:{}'.format(max([f[1] for f in all_fscore])))