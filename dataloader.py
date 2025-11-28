from dataset import MultimodalDAGDataset # 修正版 dataset.py
import pickle
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import os
import numpy as np

def get_train_valid_sampler(trainset):
    """
    データセットのインデックスサンプラーを返す。
   
    """
    size = len(trainset)
    idx = list(range(size))
    # (注: trainset.session_list のインデックス 0~size-1 をシャッフルする)
    return SubsetRandomSampler(idx)

def load_vocab(data_dir):
    """
    話者語彙とラベル語彙をロード（または定義）する。
    """
    datapath = data_dir
    
    # 話者語彙（性別）をロード
    speaker_vocab_path = os.path.join(datapath, 'speaker_vocab.pkl')
    if os.path.exists(speaker_vocab_path):
        speaker_vocab = pickle.load(open(speaker_vocab_path, 'rb'))
    else:
        print("Warning: speaker_vocab.pkl not found. Using default {'M': 0, 'F': 1}")
        speaker_vocab = {'stoi': {'M': 0, 'F': 1}}
    
    # ラベル語彙（5クラス）を定義
    label_vocab = {
        'stoi': {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3, 'oth': 4},
        'itos': {0: 'neu', 1: 'hap', 2: 'sad', 3: 'ang', 4: 'oth'}
    }

    return speaker_vocab, label_vocab

def get_multimodal_loaders(data_dir='output_data', batch_size=32, num_workers=0, pin_memory=False, args = None):
    """
    修正版 MultimodalDAGDataset を使用して、
    train/valid/test 用の DataLoader を作成する。
   
    """
    print('building vocab.. ')
    speaker_vocab, label_vocab = load_vocab(data_dir)
    
    print('building datasets..')
    # 修正: MultimodalDAGDataset を使用
    # 'split'引数に基づき、dataset内部で固定分割が行われる
    trainset = MultimodalDAGDataset(split='train', speaker_vocab=speaker_vocab, args=args, data_dir=data_dir)
    devset = MultimodalDAGDataset(split='dev', speaker_vocab=speaker_vocab, args=args, data_dir=data_dir)
    testset = MultimodalDAGDataset(split='test', speaker_vocab=speaker_vocab, args=args, data_dir=data_dir)
    
    train_sampler = get_train_valid_sampler(trainset)
    valid_sampler = get_train_valid_sampler(devset)

    # collate_fn は dataset インスタンスのメソッドを指定する
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn, #
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(devset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=devset.collate_fn, #
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn, #
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader, speaker_vocab, label_vocab